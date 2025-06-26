"""
Additional FHIR resource creation utilities (continued).
"""

import copy
import datetime
import pandas as pd
import json


def create_location(rid, val, loc_template):
    """Create FHIR Location resource."""
    coding_tmplt = {"code": "", "display": "", "system": "http://snomed.info/sct/"}
    cond = copy.deepcopy(loc_template)
    cond["id"] = rid
    cond["name"] = val["location name"]
    cond["position"]["longitude"] = val["longitude"]
    cond["position"]["latitude"] = val["latitude"]
    return cond


def create_group(rid, patient_ids, grp_template):
    """Create FHIR Group resource."""
    grp = copy.deepcopy(grp_template)
    grp["id"] = rid
    grp["name"] = rid
    grp["title"] = rid
    grp["status"] = "active"
    grp["type"] = "person"
    grp["publisher"] = "ICMR"
    grp["membership"] = "enumerated"
    grp["member"] = patient_ids
    grp["quantity"] = len(patient_ids)
    return grp


def create_Encounter(rid, idd, snm_cd, valset, val, lids, enc_template):
    """Create FHIR Encounter resource."""
    coding_tmplt = {"coding": "", "display": "", "system": "http://snomed.info/sct/"}
    cond = copy.deepcopy(enc_template)
    cond["id"] = rid
    cond["subject"]["reference"] = "Patient/" + str(idd)
    cond["plannedStartDate"] = str(val)
    for lid in lids:
        dct = {"location": {"reference": ""}}
        dct["location"]["reference"] = f"Location/{lid}"
        cond["location"].append(dct)
    return cond


def create_intake(rid, idd, snm_cd, additionalfhir, valset, val, text, obs_template, valueMap=None):
    """Create FHIR NutritionIntake resource."""
    coding_tmplt = {"code": "", "display": "", "system": "http://snomed.info/sct/"}
    timing_tmplt = {"repeat": {"when": ["MORN"]}}
    ni = copy.deepcopy(obs_template)
    ni["id"] = rid
    ni["subject"]["reference"] = "Patient/" + str(idd)
    
    # Handle string vs. list format for snm_cd and safely parse
    try:
        if isinstance(snm_cd, str):
            concepts = snm_cd.split(",")
        elif isinstance(snm_cd, list):
            concepts = snm_cd
        else:
            # Default concept if format is unexpected
            concepts = [{"SCT_ID": "261665006", "SCT_Name": "Unknown"}]
    except:
        concepts = [{"SCT_ID": "261665006", "SCT_Name": "Unknown"}]
    
    ni["code"]["coding"] = []
    
    # Process concepts based on format
    for concept in concepts:
        try:
            ct = copy.deepcopy(coding_tmplt)
            
            if isinstance(concept, str) and "(" in concept:
                # Parse format like "Name (Code)"
                parts = concept.split("(")
                ct["display"] = parts[0].strip()
                ct["code"] = parts[1].strip(")").strip()
            elif isinstance(concept, dict):
                # Handle dictionary format
                if 'SCT_ID' in concept:
                    if isinstance(concept['SCT_ID'], list):
                        ct["code"] = str(concept['SCT_ID'][0])
                    else:
                        ct["code"] = str(concept['SCT_ID'])
                    
                    if 'SCT_Name' in concept:
                        ct["display"] = concept['SCT_Name']
                    else:
                        ct["display"] = "Unknown"
                else:
                    # Default if no SCT_ID
                    ct["code"] = "261665006"
                    ct["display"] = "Unknown"
            else:
                # Default for unsupported format
                ct["code"] = "261665006"
                ct["display"] = "Unknown"
                
            ni["code"]["coding"].append(ct)
        except Exception as e:
            print(f"Error processing concept in create_intake: {e}")
            # Add default if error
            ct = copy.deepcopy(coding_tmplt)
            ct["code"] = "261665006"
            ct["display"] = "Error"
            ni["code"]["coding"].append(ct)

    # Set text if provided
    if text and isinstance(text, str):
        ni["code"]["text"] = text

    # Process additional FHIR properties
    if additionalfhir and not isinstance(additionalfhir, float):
        # Apply timing info
        ni["effectiveTiming"] = timing_tmplt
        
        # Convert to string to handle various input types
        additionalfhir_str = str(additionalfhir)
        
        if "MORN" in additionalfhir_str:
            ni["effectiveTiming"]["repeat"]["when"][0] = "MORN"
        elif "AFT" in additionalfhir_str:
            ni["effectiveTiming"]["repeat"]["when"][0] = "AFT"
        elif "EVE" in additionalfhir_str:
            ni["effectiveTiming"]["repeat"]["when"][0] = "EVE"
        elif "NIGHT" in additionalfhir_str:
            ni["effectiveTiming"]["repeat"]["when"][0] = "NIGHT"

    return ni


def create_loc(idd, val, loc_template):
    """Create FHIR location for patient."""
    loc = copy.deepcopy(loc_template)
    loc["subject"]["reference"] = "Patient/" + str(idd)
    loc["name"] = val
    return loc