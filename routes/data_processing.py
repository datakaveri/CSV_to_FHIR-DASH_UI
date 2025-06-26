"""
File processing and data manipulation utilities.
"""

import os
import pandas as pd
import traceback


def process_file_from_path(file_path, filename):
    """Process a file from a full file path"""
    try:
        # Read the file based on extension
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return None, f"Unsupported file format: {filename}"
        
        # Extract base filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        return df, base_filename
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        traceback.print_exc()
        return None, f"Error processing file: {str(e)}"
