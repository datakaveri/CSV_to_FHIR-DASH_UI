"""
FHIR resource creation utilities.
"""

import copy
import datetime
import pandas as pd
from utils.templates import load_templates


def create_condition(rid, idd, snm_cd, valset, val, cond_template, valueMap=None):
    """Create FHIR Condition resource."""
    coding_tmplt = {"code": "", "display": "", "system": "http://snomed.info/sct/"}
    cond = copy.deepcopy(cond_template)
    cond["id"] = rid
    cond["subject"]["reference"] = "Patient/" + str(idd)
    
    # Make sure snm_cd is a list and handle empty cases
    if not snm_cd or len(snm_cd) == 0:
        # Default coding for empty entities
        ct = copy.deepcopy(coding_tmplt)
        ct["code"] = "261665006"  # Unknown (qualifier value)
        ct["display"] = "Unknown"
        cond["code"]["coding"] = [ct]
    else:
        cond["code"]["coding"] = []
        for i in snm_cd:
            try:
                ct = copy.deepcopy(coding_tmplt)
                # Handle different formats of SCT_ID
                if isinstance(i, dict) and 'SCT_ID' in i:
                    if isinstance(i['SCT_ID'], list) and i['SCT_ID']:
                        ct["code"] = str(i['SCT_ID'][0])
                    else:
                        ct["code"] = str(i['SCT_ID'])
                    
                    # Safely handle SCT_Name
                    if 'SCT_Name' in i and i['SCT_Name']:
                        ct["display"] = i['SCT_Name']
                    else:
                        ct["display"] = "Unspecified"
                else:
                    # Unknown format, use defaults
                    ct["code"] = "261665006"
                    ct["display"] = "Unknown"
                
                cond["code"]["coding"].append(ct)
            except Exception as e:
                print(f"Error processing SNOMED entity: {e}, {i}")
                # Add default if there's an error
                ct = copy.deepcopy(coding_tmplt)
                ct["code"] = "261665006"
                ct["display"] = "Error processing entity"
                cond["code"]["coding"].append(ct)
    
    # Set verification status based on value and mapping
    if not pd.isna(val) and valueMap and str(val) in valueMap:
        cond["verificationStatus"]["coding"][0]["code"] = valueMap[str(val)]
    else:
        # Default based on value truthiness
        if val and not pd.isna(val):
            cond["verificationStatus"]["coding"][0]["code"] = "confirmed"
        else:
            cond["verificationStatus"]["coding"][0]["code"] = "refuted"
            
    return cond


def create_obsv(rid, idd, snm_cd, valset, val, obs_template, valueMap=None):
    """Create FHIR Observation resource."""
    coding_tmplt = {"code": "", "display": "", "system": "http://snomed.info/sct/"}
    obs = copy.deepcopy(obs_template)
    obs["id"] = rid
    obs["subject"]["reference"] = "Patient/" + str(idd)
    
    # Make sure snm_cd is a list and handle empty cases
    if not snm_cd or len(snm_cd) == 0:
        # Default coding for empty entities
        ct = copy.deepcopy(coding_tmplt)
        ct["code"] = "261665006"  # Unknown (qualifier value)
        ct["display"] = "Unknown"
        obs["code"]["coding"] = [ct]
    else:
        obs["code"]["coding"] = []
        for i in snm_cd:
            try:
                ct = copy.deepcopy(coding_tmplt)
                # Handle different formats of SCT_ID
                if isinstance(i, dict) and 'SCT_ID' in i:
                    if isinstance(i['SCT_ID'], list) and i['SCT_ID']:
                        ct["code"] = str(i['SCT_ID'][0])
                    else:
                        ct["code"] = str(i['SCT_ID'])
                    
                    # Safely handle SCT_Name
                    if 'SCT_Name' in i and i['SCT_Name']:
                        ct["display"] = i['SCT_Name']
                    else:
                        ct["display"] = "Unspecified"
                else:
                    # Unknown format, use defaults
                    ct["code"] = "261665006"
                    ct["display"] = "Unknown"
                
                obs["code"]["coding"].append(ct)
            except Exception as e:
                print(f"Error processing SNOMED entity: {e}, {i}")
                # Add default if there's an error
                ct = copy.deepcopy(coding_tmplt)
                ct["code"] = "261665006"
                ct["display"] = "Error processing entity"
                obs["code"]["coding"].append(ct)

    # Process value mapping if provided
    if valueMap is not None:
        if "valueInteger" in valset:
            val_str = str(val)
            if val_str in valueMap:
                obs["valueInteger"] = valueMap[val_str]
            else:
                obs["valueInteger"] = 0
        elif "valueBoolean" in valset:
            val_str = str(val).replace(".0", "")
            if val_str in valueMap:
                obs["valueBoolean"] = valueMap[val_str]
            else:
                obs["valueBoolean"] = False
        elif "valueDateTime" in valset:
            if val in valueMap:
                obs["valueDateTime"] = valueMap[val]
            else:
                obs["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        return obs

    # Handle different value types with sensible defaults
    if "valueInteger" in valset:
        if pd.isna(val):
            obs["valueInteger"] = 0
        else:
            try:
                obs["valueInteger"] = int(float(val))
            except (ValueError, TypeError):
                obs["valueInteger"] = 0
    elif "valueBoolean" in valset:
        if pd.isna(val):
            obs["valueBoolean"] = False
        elif val == 0 or val == "0" or val == "0.0" or val == "False" or val == "false" or val == "no" or val == "No":
            obs["valueBoolean"] = False
        else:
            obs["valueBoolean"] = True
    elif "valueDateTime" in valset:
        if pd.isna(val):
            obs["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        else:
            try:
                if isinstance(val, (datetime.datetime, pd.Timestamp)):
                    obs["valueDateTime"] = val.strftime("%Y-%m-%dT%H:%M:%S")
                else:
                    # Try to parse string to datetime
                    dt = pd.to_datetime(val)
                    obs["valueDateTime"] = dt.strftime("%Y-%m-%dT%H:%M:%S")
            except:
                # Fallback to current time if parsing fails
                obs["valueDateTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    elif "valueString" in valset:
        if pd.isna(val):
            obs["valueString"] = ""
        else:
            obs["valueString"] = str(val)
    else:
        # If we can't determine the value type, default to string
        if pd.isna(val):
            obs["valueString"] = ""
        else:
            obs["valueString"] = str(val)
    
    return obs


def load_fhir_data(df, dataset_name):
    """Load and convert DataFrame to FHIR bundle format."""
    import uuid
    import json
    import traceback
    from utils.snomed_processing import Patient
    from utils.fhir_resources import create_group, create_intake
    
    patient_template, age_template, obs_template, loc_template, cond_template, ni_template, enc_template, grp_template = load_templates()
    fullbundle = []
    baseurl = "https://adarv.icmr.gov.in/add_mtzion/base/"
    location_data = []
    patient_ids = []
    gndr = None
    
    # Remove 'dataset_' prefix from the dataset_name if it exists
    if dataset_name and dataset_name.startswith('dataset_'):
        group_id = dataset_name[len('dataset_'):]
    else:
        group_id = dataset_name
    
    # Safely find gender column
    for col in df.columns:
        try:
            # Use getattr with default empty dict to avoid KeyError
            attrs = getattr(df[col], 'attrs', {})
            if 'FHIR_Resource' in attrs and attrs['FHIR_Resource'] == 'Patient.Gender':
                gndr = col
                break
        except Exception as e:
            print(f"Error checking column {col} for gender: {e}")
    
    sample_bundle = []
    bundle_count = 1
    for index, data in df.iterrows():
        bundle = {"resourceType": "Bundle", "type": "transaction", "entry": []}
        resource_template = {"request": {"method": "PUT"}, "fullUrl": baseurl, "resource": {}}
        pat = Patient()
        patid = lid = str(uuid.uuid1())
        pat.set_id(patid)
        patient_ids.append({
                "entity": {"reference" : f"Patient/{patid}"}
            })
        if gndr:
            pat.set_gender(data[gndr])
        rpt = copy.deepcopy(resource_template)
        rpt["fullUrl"] = baseurl+"Patient/" + pat._id
        rpt["resource"] = pat.create_patient(patient_template)
        bundle["entry"].append(rpt)

        for k in df.columns:
            rpt = copy.deepcopy(resource_template)
            rid = str(uuid.uuid1())
            try:
                # Safe attribute access with defaults
                attrs = getattr(df[k], 'attrs', {})
                fhir_resource = attrs.get('FHIR_Resource', 'observation')
                entities = attrs.get('Entities', [])
                value_set = attrs.get('valueSet', 'valueString')
                val_mod = attrs.get('valueModifier', None)
                
                # Skip Gender as it's handled separately
                if fhir_resource == 'Patient.Gender':
                    continue

                if fhir_resource == 'condition':
                    cd = create_condition(rid, pat._id, entities, value_set, data[k],
                                        cond_template, val_mod)
                    rpt["fullUrl"] = baseurl+"Condition/" + rid
                    rpt["resource"] = cd
                    bundle["entry"].append(rpt)
    
                elif fhir_resource == 'observation': 
                    obs = create_obsv(rid, pat._id, entities, value_set, data[k],
                                    obs_template, val_mod)
                    if obs is None:
                        continue
                    rpt["fullUrl"] = baseurl+ "Observation/" + rid
                    rpt["resource"] = obs
                    bundle["entry"].append(rpt)

                elif fhir_resource == 'observation (intake)':
                    additional_props = attrs.get('Additional FHIR Properties', "")
                    itk = create_intake(rid, pat._id, entities, additional_props,
                                    value_set, data[k], k, obs_template, val_mod)
                    rpt["fullUrl"] = baseurl+"Observation/" + rid
                    rpt["resource"] = itk
                    bundle["entry"].append(rpt)
    
            except Exception as e:
                print(f"Error processing column {k}:")
                print(e)
                print(traceback.format_exc())
                continue
    
        for entry in bundle["entry"]:
            resource_data = entry["resource"]
            url = f"http://65.0.127.208:30007/fhir/{resource_data['resourceType']}/{resource_data['id']}"
            # r = requests.put(url,
            #             headers= {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"},
            #             data=json.dumps(resource_data))
            # print(r.status_code)
            # print(r.json())
    
        fullbundle.append(bundle)
        if bundle_count == 1:
            sample_bundle.append(bundle)
            bundle_count += 1

    # Create the Group resource with the cleaned ID (without dataset_ prefix)
    url = f"http://65.0.127.208:30007/fhir/Group/{group_id}"
    resource_data = create_group(group_id, patient_ids, grp_template)
    
    # Create a bundle entry for the Group resource
    group_bundle_entry = {
        "request": {"method": "PUT"}, 
        "fullUrl": baseurl + "Group/" + group_id,
        "resource": resource_data
    }
    
    # Add the Group resource to its own transaction bundle
    group_bundle = {
        "resourceType": "Bundle", 
        "type": "transaction", 
        "entry": [group_bundle_entry]
    }
    
    # Add the Group bundle to the full bundle
    fullbundle.append(group_bundle)
    
    # Add the Group resource to the sample bundle too, if it exists
    if sample_bundle:
        sample_bundle.append(group_bundle)
    
    with open("bundle.json", "w") as f:
        json.dump(fullbundle, f, indent=4)

    return fullbundle, sample_bundle