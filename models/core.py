from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional

class Observation(BaseModel):
    table_sample: list
    columns: List[Dict[str, str]]
    step: int
    max_steps: int

class Action(BaseModel):
    type: str # e.g., 'label_issue', 'fix_cell', 'drop_row', 'submit_sql', 'inspect_column', 'validate_step', 'finish'
    row: Optional[int] = None
    col: Optional[str] = None
    value: Optional[Any] = None
    sql: Optional[str] = None
    issue_type: Optional[str] = None

class Reward(BaseModel):
    score: float
    breakdown: Dict[str, float]

class EpisodeState(BaseModel):
    dirty_df: Any # pd.DataFrame under the hood
    clean_df: Any
    manifest: list # list of dicts detailing true issues
    cells_fixed: dict # dict of formatted "row|col" -> submitted_value
    issues_found: list # list of dicts of labeled issues
    step_count: int
    history: list # list of dicts of past actions
    max_steps: int
    done: bool
