"""
SNOMED code processing and sentiment analysis utilities.
"""

import copy
import pandas as pd
from transformers import pipeline


def mark_condition_resources(list_of_dicts):
    """Mark condition resources in list of dictionaries."""
    for outer_dict in list_of_dicts:
        for key, value in outer_dict.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                if any(inner_dict.get('FHIR_Resource') == 'condition' for inner_dict in value):
                    for inner_dict in value:
                        inner_dict['FHIR_Resource'] = 'condition'
    
    return list_of_dicts


def process_snmd_cds(list_of_dicts):
    """Process SNOMED codes and assign FHIR resources."""
    for outer_dict in list_of_dicts:
        updates = {}
        for key, value in outer_dict.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                if any(inner_dict.get('FHIR_Resource') == 'condition' for inner_dict in value):
                    updates['FHIR_Resource'] = 'condition'
                elif any(inner_dict.get('FHIR_Resource') == 'Patient.Age' for inner_dict in value):
                    updates['FHIR_Resource'] = 'Patient.Age'
                elif any(inner_dict.get('FHIR_Resource') == 'Patient.Gender' for inner_dict in value):
                    updates['FHIR_Resource'] = 'Patient.Gender'
                else:
                    updates['FHIR_Resource'] = 'observation'
        
        outer_dict.update(updates)
    return list_of_dicts


def check_sentiment(text):
    """Check sentiment of text using transformer pipeline."""
    sentiment_analysis = pipeline("sentiment-analysis")
    result = sentiment_analysis(text)
    return result


def value_modifier(df):
    """Modify values based on sentiment analysis."""
    for col in df.columns:
        if df[col].attrs['FHIR_Resource'] == 'observation' and df[col].attrs['valueSet'] == 'valueBoolean':
            value_modifier = {}
            unique_values = df[col].unique()
            for value in unique_values:
                value = str(value)
                if not pd.isna(value):
                    sentiment = check_sentiment(str(value))
                    if sentiment[0]['label'] == 'POSITIVE':
                        value_modifier[value] = True
                    else:
                        value_modifier[value] = False
                else:
                    value_modifier[value] = False
            df[col].attrs['valueModifier'] = value_modifier
        elif df[col].attrs['FHIR_Resource'] == 'condition':
            value_modifier = {}
            unique_values = df[col].unique()
            for value in unique_values:
                value = str(value)
                if not pd.isna(value):
                    sentiment = check_sentiment(str(value))
                    if sentiment[0]['label'] == 'POSITIVE':
                        value_modifier[value] = 'confirmed'
                    else:
                        value_modifier[value] = 'refuted'
                else:
                    value_modifier[value] = 'refuted'
            df[col].attrs['valueModifier'] = value_modifier
    return df


class Patient:
    """Patient class for FHIR patient resource creation."""
    def __init__(self):
        self._id = ""
        self._gender = ""

    def set_id(self, idd):
        """Set patient ID."""
        self._id = str(idd)

    def set_gender(self, gender):
        """Set patient gender."""
        if type(gender) == str:
            if (gender.lower() == 'male'):
                self._gender = "male"
            if (gender.lower() == 'female'):
                self._gender = "female"
        else:
            if (gender == 1):
                self._gender = "male"
            if (gender == 2):
                self._gender = "female"

    def create_patient(self, patient_template):
        """Create patient resource from template."""
        pat = copy.deepcopy(patient_template)
        pat["id"] = self._id
        pat["gender"] = self._gender
        return pat
