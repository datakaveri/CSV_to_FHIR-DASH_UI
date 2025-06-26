"""
Text and string utility functions.
"""

import json
import re


def safe_json_loads(json_string, default=None):
    """Safely load JSON string with fallback."""
    if not json_string:
        return default or {}
    
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default or {}


def truncate_text(text, max_length=50):
    """Truncate text to specified length with ellipsis."""
    if not text:
        return ""
    
    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str
    
    return text_str[:max_length - 3] + "..."


def get_sample_data_string(df, column, max_samples=3):
    """Get a string representation of sample data from a column."""
    if df is None or column not in df.columns:
        return ""
    
    sample_values = df[column].dropna().head(max_samples).tolist()
    if not sample_values:
        return "No data"
    
    sample_strings = [truncate_text(v, 20) for v in sample_values]
    return ', '.join(sample_strings)


def validate_snomed_code(code):
    """Basic validation for SNOMED CT codes."""
    if not code:
        return False
    
    # SNOMED CT codes are typically numeric and 6-18 digits long
    code_str = str(code).strip()
    if not code_str.isdigit():
        return False
    
    if len(code_str) < 6 or len(code_str) > 18:
        return False
    
    return True


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


def split_entities(entity_text):
    """Utility function to split entities from comma-separated text"""
    if not entity_text or entity_text == "None":
        return []
    
    # First, clean and normalize the string
    entity_text = entity_text.strip()
    
    # Improved entity splitting
    raw_entities = re.split(r',\s*', entity_text)
    
    # Clean up and deduplicate each entity
    entities = []
    for entity in raw_entities:
        entity = entity.strip()
        if entity and entity not in entities:
            entities.append(entity)
    
    return entities
