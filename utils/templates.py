"""
FHIR template loading utilities.
"""

import json


def load_templates():
    """Reading FHIR Resource JSON Templates"""
    with open("./templates/Patient.json", "r") as f:
        patient_template = json.load(f)
    with open("./templates/Age.json", "r") as f:
        age_template = json.load(f)
    with open("./templates/Observation.json", "r") as f:
        obs_template = json.load(f)
    with open("./templates/Location.json", "r") as f:
        loc_template = json.load(f)
    with open("./templates/Condition.json", "r") as f:
        cond_template = json.load(f)
    with open("./templates/NutritionIntake.json", "r") as f:
        ni_template = json.load(f)
    with open("./templates/Encounter.json", "r") as f:
        enc_template = json.load(f)
    with open("./templates/Group.json", "r") as f:
        grp_template = json.load(f)
    return patient_template, age_template, obs_template, loc_template, cond_template, ni_template, enc_template, grp_template
