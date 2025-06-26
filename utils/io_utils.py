"""
Input/Output utilities and file processing functions.
"""

import re
import os
import json
import requests
import pandas as pd
import base64
import io


def sanitize_for_json(text):
    """Sanitize text to be safely included in JSON strings."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # First, handle any encoding issues
    try:
        text = text.encode('utf-8', errors='replace').decode('utf-8')
    except:
        pass
    
    # Remove or replace all potentially problematic characters
    # Replace control characters and special whitespace
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control characters
    text = re.sub(r'[\u00a0\u2000-\u200f\u2028-\u202f\u205f-\u206f]', ' ', text)  # Replace special spaces
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Now escape JSON-problematic characters
    text = (text.replace('\\', '\\\\')
                .replace('"', '\\"')
                .replace('\n', '\\n')
                .replace('\r', '\\r')
                .replace('\t', '\\t')
                .replace('\b', '\\b')
                .replace('\f', '\\f'))
    
    return text


def search_and_select_top_match(entity, col_name, fhir_resource, category=""):
    """
    Search for SNOMED codes for an entity and select the top result automatically
    Returns a dict with the mapping information or None if no match found
    """
    try:
        # API endpoint for SNOMED search
        API_ENDPOINT = "https://entitymapper.adarv.in/search"
        
        # Make API request to search for SNOMED codes
        payload = {
            "keywords": entity,
            "threshold": 0.5,
            "limit": 10
        }
        
        # Make request to the API
        response = requests.post(API_ENDPOINT, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            results = json.loads(response.text)
            
            # Extract the first result if available
            if results and len(results) > 0:
                top_result = results[0]
                return {
                    "entity": entity,
                    "col_name": col_name,
                    "fhir_resource": fhir_resource,
                    "category": category,
                    "snomed_id": top_result.get("ConceptID", ""),
                    "snomed_name": top_result.get("ConceptID_name", ""),
                    "score": top_result.get("score", 0)
                }
            else:
                return None
        else:
            print(f"API request failed with status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error searching SNOMED for entity {entity}: {e}")
        return None


def process_file_from_path(file_path, filename):
    """Process a file from a full file path"""
    try:
        # Read the file based on extension
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return None, f"Unsupported file type: {filename}"
        
        # Extract base filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        return df, base_filename
        
    except Exception as e:
        return None, f"Error processing file: {str(e)}"


def process_upload_content(contents, filename):
    """Process uploaded file content."""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, f"Unsupported file type: {filename}"
        
        return df, None
        
    except Exception as e:
        return None, f"Error processing upload: {str(e)}"
