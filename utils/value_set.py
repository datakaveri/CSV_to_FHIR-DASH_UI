"""
Value set creation utilities for FHIR resources.
"""

import pandas as pd


def create_valueset(df):
    """Create value sets for DataFrame columns based on their data types."""
    for col in df.columns:
        # Initialize attrs dictionary if it doesn't exist
        if not hasattr(df[col], 'attrs'):
            df[col].attrs = {}
            
        # if df[col].attrs["FHIR_Resource"] == "condition":
        #     df[col].attrs['valueSet'] = 'valueBoolean'
        # elif df[col].attrs["FHIR_Resource"] == "observation":
        unique_values = df[col].dropna().unique()
        
        if pd.api.types.is_bool_dtype(df[col].dropna()):
            df[col].attrs['valueSet'] = 'valueBoolean'
        elif pd.api.types.is_string_dtype(df[col].dropna()):
            if set(unique_values).issubset({'yes', 'no', 'Yes', 'No', 'YES', 'NO'}):
                df[col].attrs['valueSet'] = 'valueBoolean'
            else:
                df[col].attrs['valueSet'] = 'valueString'
        elif pd.api.types.is_datetime64_any_dtype(df[col].dropna()):
            df[col].attrs['valueSet'] = 'valueDateTime'
        elif pd.api.types.is_integer_dtype(df[col].dropna()) or pd.api.types.is_float_dtype(df[col].dropna()):
            df[col].attrs['valueSet'] = 'valueInteger'
        else:
            df[col].attrs['valueSet'] = 'valueString'
    return df
