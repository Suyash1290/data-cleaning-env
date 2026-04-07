from models.core import Observation, Action, Reward, EpisodeState
from data.generator import generate_dataset
from graders.fix_grader import normalize_value
from tasks.hard import run_sql_against_data
from tasks.base import MAX_STEPS

class DataCleaningEnv:
    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        self.max_steps = MAX_STEPS.get(difficulty, 20)
        self.state = None

    def reset(self, seed: int = 42) -> Observation:
        dirty_df, clean_df, manifest = generate_dataset(seed, 100, self.difficulty)
        
        self.state = EpisodeState(
            dirty_df=dirty_df,
            clean_df=clean_df,
            manifest=manifest,
            cells_fixed={},
            issues_found=[],
            step_count=0,
            history=[],
            max_steps=self.max_steps,
            done=False
        )
        return self._get_obs()

    def _get_obs(self) -> Observation:
        sample = self.state.dirty_df.head(5).replace({float('nan'): None}).to_dict(orient="records")
        columns = [{"name": str(c), "dtype": str(t)} for c, t in self.state.dirty_df.dtypes.items()]
        return Observation(
            table_sample=sample,
            columns=columns,
            step=self.state.step_count,
            max_steps=self.state.max_steps
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool]:
        if self.state.done:
            return self._get_obs(), Reward(score=0.0, breakdown={"error": 1.0}), True

        reward_score = -0.01
        breakdown = {"step_cost": -0.01}

        # Check loop penalty
        action_dict = action.model_dump() if hasattr(action, "model_dump") else action.dict()
        if len(self.state.history) >= 3:
            last_3 = self.state.history[-3:]
            if all(past == action_dict for past in last_3):
                reward_score -= 0.10
                breakdown["loop_penalty"] = -0.10

        self.state.history.append(action_dict)
        self.state.step_count += 1

        if action.type == "finish":
            self.state.done = True
        elif action.type == "label_issue":
            key = f"{action.row}|{action.col}"
            if key in self.state.issues_found:
                reward_score -= 0.05
                breakdown["duplicate_label"] = -0.05
            else:
                found = False
                for issue in self.state.manifest:
                    if issue["row"] == action.row and issue["col"] == action.col:
                        if issue["issue_type"] == action.issue_type:
                            self.state.issues_found.append(key)
                            reward_score += 0.15
                            breakdown["correct_label"] = 0.15
                            found = True
                            break
                        else:
                            reward_score -= 0.05
                            breakdown["wrong_type_label"] = -0.05
                            found = True
                            break
                if not found:
                    reward_score -= 0.05
                    breakdown["false_positive_label"] = -0.05

        elif action.type == "fix_cell":
            if action.row is None or action.col is None or action.row >= len(self.state.dirty_df) or action.col not in self.state.dirty_df.columns:
                pass # invalid row/col -> 0.0 penalty according to spec
            else:
                key = f"{action.row}|{action.col}"
                manifest_issue = next((m for m in self.state.manifest if m["row"] == action.row and m["col"] == action.col), None)
                if manifest_issue:
                    submitted = normalize_value(action.value, action.col)
                    expected = normalize_value(manifest_issue["expected"], action.col)
                    if submitted == expected:
                        reward_score += 0.10
                        breakdown["correct_fix"] = 0.10
                        self.state.cells_fixed[key] = action.value
                        self.state.dirty_df.at[action.row, action.col] = action.value
                    else:
                        reward_score -= 0.03
                        breakdown["wrong_fix"] = -0.03
                else:
                    reward_score -= 0.05
                    breakdown["red_herring_fix"] = -0.05

        elif action.type == "submit_sql":
            if action.sql:
                passed, err = run_sql_against_data(self.state.dirty_df, action.sql)
                if err:
                    reward_score -= 0.05
                    breakdown["sql_error"] = -0.05
                else:
                    rules_passed = sum(passed)
                    reward_score += rules_passed * 0.20
                    breakdown["sql_passed"] = rules_passed * 0.20
            else:
                reward_score -= 0.05

        elif action.type == "drop_row":
            if action.row is not None and action.row < len(self.state.dirty_df):
                manifest_issue = next((m for m in self.state.manifest if m["row"] == action.row and m["issue_type"] == "duplicate"), None)
                if manifest_issue:
                    self.state.dirty_df = self.state.dirty_df.drop(action.row).reset_index(drop=True)
                    reward_score += 0.10
                    breakdown["correct_drop"] = 0.10
                else:
                    reward_score -= 0.08
                    breakdown["wrong_drop"] = -0.08

        elif action.type == "inspect_column":
            reward_score -= 0.02
            breakdown["inspect_cost"] = -0.02

        if self.state.step_count >= self.state.max_steps:
            self.state.done = True

        self._cumulative_score = getattr(self, "_cumulative_score", 0.001)
        if self._cumulative_score + reward_score <= 0.001:
            reward_score = 0.001 - self._cumulative_score
        elif self._cumulative_score + reward_score >= 0.999:
            reward_score = 0.999 - self._cumulative_score
            
        self._cumulative_score += reward_score

        return self._get_obs(), Reward(score=reward_score, breakdown=breakdown), self.state.done
