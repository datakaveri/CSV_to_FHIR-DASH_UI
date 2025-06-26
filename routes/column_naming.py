"""
Column name generation utilities.
"""

from utils.embedding import remove_punctuation
from .db_operations import get_suggested_names_from_adarv_dict


def generate_column_names(processed_columns, columns_list):
    """
    Generate column names using the following priority:
    1. API suggested_name (highest priority)
    2. ADARV data dictionary suggested names
    3. Entity-based naming logic
    4. Fallback naming
    """
    # First, get suggested names from ADARV dict table (for fallback)
    adarv_suggested_names = get_suggested_names_from_adarv_dict(columns_list)
    
    column_renamed_mapping = {}
    used_names = set()
    
    print(f"DEBUG: Starting column name generation for {len(columns_list)} columns")
    print(f"DEBUG: Initial used_names: {used_names}")
    
    # FIRST PASS: Use API suggested_name where available (highest priority)
    for col_name in columns_list:
        if col_name in processed_columns:
            col_data = processed_columns[col_name]
            entities = col_data['entities']
            
            # Check if any entity has a suggested_name
            for entity in entities:
                suggested_name = entity.get('suggested_name', '').strip()
                if suggested_name:
                    print(f"DEBUG: Processing {col_name}, original suggested_name: '{suggested_name}'")
                    print(f"DEBUG: used_names before processing: {used_names}")
                    
                    # Use the suggested name as-is without numbering for duplicates
                    column_renamed_mapping[col_name] = suggested_name
                    used_names.add(suggested_name)
                    print(f"DEBUG: Using API suggested name for {col_name}: '{suggested_name}'")
                    print(f"DEBUG: used_names after adding: {used_names}")
                    break  # Use the first suggested name found
    
    # SECOND PASS: Use ADARV suggested names for unmapped columns
    for col_name in columns_list:
        # Skip if already named by API suggested_name
        if col_name in column_renamed_mapping:
            continue
            
        if col_name in adarv_suggested_names:
            suggested_name = adarv_suggested_names[col_name]
            
            # Use ADARV suggested name as-is without numbering for duplicates
            column_renamed_mapping[col_name] = suggested_name
            used_names.add(suggested_name)
            print(f"Using ADARV suggested name for {col_name}: {suggested_name}")
    
    # THIRD PASS: Generate names for columns with unified API mappings (entity-based naming)
    for col_name in columns_list:
        # Skip if already named by API suggested_name or ADARV
        if col_name in column_renamed_mapping:
            continue
            
        # If we have API entities for this column, generate based on entities
        if col_name in processed_columns:
            col_data = processed_columns[col_name]
            entities = col_data['entities']
            
            if entities:
                # Get FHIR resource from first entity
                fhir_resource = entities[0]['fhir_resource']
                
                # Extract prefix from FHIR resource (same logic as fhirconv.py)
                if fhir_resource.lower().startswith('condition'):
                    prefix = 'con'
                elif fhir_resource.lower().startswith('patient.gender'):
                    prefix = 'gen'
                elif fhir_resource.lower().startswith('patient.age'):
                    prefix = 'age'
                elif fhir_resource.lower().startswith('patient'):
                    prefix = 'pat'
                else:
                    prefix = fhir_resource[:3].lower()
                
                # Generate name based on entities (same logic as fhirconv.py)
                if len(entities) == 1:
                    # Single entity - use first 3 chars of concept name
                    concept_name = entities[0].get('snomed_name', '').strip()
                    if concept_name:
                        # Get first word and take first 3 chars
                        first_word = concept_name.split(',')[0].strip()
                        clean_name = first_word.lower().replace(' ', '').replace('(', '').replace(')', '')
                        abbr_col = f"{prefix}_{clean_name[:3]}"
                    else:
                        abbr_col = f"{prefix}_{col_name[:3].lower()}"
                        
                else:
                    # Multiple entities - use first 3 chars of each concept name
                    entity_parts = []
                    for entity in entities[:3]:  # Limit to first 3 entities
                        concept_name = entity.get('snomed_name', '').strip()
                        if concept_name:
                            first_word = concept_name.split(',')[0].strip()
                            clean_name = first_word.lower().replace(' ', '').replace('(', '').replace(')', '')
                            entity_parts.append(clean_name[:3])
                    
                    if entity_parts:
                        abbr_col = f"{prefix}_{'_'.join(entity_parts)}"
                    else:
                        abbr_col = f"{prefix}_{col_name[:3].lower()}"
                
                # Handle duplicates
                final_name = abbr_col
                counter = 1
                while final_name in used_names:
                    final_name = f"{abbr_col}_{counter}"
                    counter += 1
                
                column_renamed_mapping[col_name] = final_name
                used_names.add(final_name)
                print(f"Generated name for {col_name}: {final_name}")
    
    # FOURTH PASS: Generate fallback names for columns that don't have any mappings
    for col_name in columns_list:
        if col_name not in column_renamed_mapping:
            # Create a simple fallback abbreviation
            vowels = "aeiouAEIOU0123456789"
            no_vowels = "".join([char for char in col_name if char.lower() not in vowels])
            no_vowels = remove_punctuation(no_vowels.replace(" ", ""))
            
            # Create a simple abbreviation
            abbr_col = no_vowels[:6].lower() if no_vowels else col_name[:6].lower()
            
            # Handle duplicates
            final_name = abbr_col
            counter = 1
            while final_name in used_names:
                final_name = f"{abbr_col}_{counter}"
                counter += 1
            
            column_renamed_mapping[col_name] = final_name
            used_names.add(final_name)
    
    print(f"DEBUG: Final column_renamed_mapping: {column_renamed_mapping}")
    return column_renamed_mapping
