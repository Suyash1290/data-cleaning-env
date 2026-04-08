from tasks.base import MAX_STEPS

def get_config():
    return {"difficulty": "medium", "max_steps": MAX_STEPS["medium"]}

def grade(state) -> float:
    from graders.fix_grader import grade as fix_grade
    return fix_grade(state)
