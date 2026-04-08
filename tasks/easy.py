from tasks.base import MAX_STEPS

def get_config():
    return {"difficulty": "easy", "max_steps": MAX_STEPS["easy"]}

def grade(state) -> float:
    from graders.fix_grader import grade as fix_grade
    return fix_grade(state)
