"""
Column name utilities and abbreviation creation.
"""

from utils.embedding import remove_punctuation


def create_col_abbrs(adarv_snmd_cds, snmd_cds):
    """
    Create abbreviated column names using SNOMED names and FHIR resource type.
    Format: first_three_of_FHIR_resource _ first_three_of_entity1_SNOMED _ first_three_of_entity2_SNOMED ...
    
    For multi-entity columns, the abbreviation is calculated after all entities have been mapped to SNOMED.
    """
    abbr_col_names = {}
    duplicate_col_check = []
    duplicate_counter = 1
    
    # First, handle ADARV mappings
    for col in adarv_snmd_cds:
        orig_col_name = col['Original_Column_Name']
        
        # Skip if we don't have entities with SCT_Name
        if not col.get('Entities'):
            continue
            
        # Extract FHIR resource prefix (first 3 letters)
        fhir_prefix = col['FHIR_Resource'][:3].lower()
        
        # Extract SNOMED terms for each entity
        snomed_prefixes = []
        for entity in col['Entities']:
            if 'SCT_Name' in entity and entity['SCT_Name']:
                # Use first 3 characters of SNOMED name
                snomed_name = entity['SCT_Name'].strip()
                if snomed_name:
                    snomed_prefixes.append(snomed_name[:3].lower())
        
        # Create abbreviated column name
        if snomed_prefixes:
            abbr_col = f"{fhir_prefix}_{'_'.join(snomed_prefixes)}"
            
            # Handle duplicates
            if abbr_col in duplicate_col_check:
                abbr_col = f"{abbr_col}{duplicate_counter}"
                duplicate_counter += 1
                
            abbr_col_names[orig_col_name] = abbr_col
            duplicate_col_check.append(abbr_col)
    
    # Then handle entries from snmd_cds
    for col_dict in snmd_cds:
        col_name = list(col_dict.keys())[0]
        
        # Skip if already processed from ADARV mappings
        if col_name in abbr_col_names:
            continue
        
        try:
            # Get FHIR resource prefix
            fhir_prefix = col_dict.get('FHIR_Resource', 'obs')[:3].lower()
            
            # Get SNOMED names for entities
            snomed_prefixes = []
            for entity in col_dict[col_name]:
                if isinstance(entity, dict) and 'SCT_Name' in entity and entity['SCT_Name']:
                    snomed_name = entity['SCT_Name'].strip()
                    if snomed_name:
                        snomed_prefixes.append(snomed_name[:3].lower())
            
            # Create abbreviated column name
            if snomed_prefixes:
                abbr_col = f"{fhir_prefix}_{'_'.join(snomed_prefixes)}"
                
                # Handle duplicates
                if abbr_col in duplicate_col_check:
                    abbr_col = f"{abbr_col}{duplicate_counter}"
                    duplicate_counter += 1
                    
                abbr_col_names[col_name] = abbr_col
                duplicate_col_check.append(abbr_col)
            else:
                # Fallback if no SNOMED names available
                abbr_col = f"{fhir_prefix}_{remove_punctuation(col_name)[:3].lower()}"
                if abbr_col in duplicate_col_check:
                    abbr_col = f"{abbr_col}{duplicate_counter}"
                    duplicate_counter += 1
                abbr_col_names[col_name] = abbr_col
                duplicate_col_check.append(abbr_col)
                
        except Exception as e:
            print(f"Error creating abbreviation for {col_name}: {e}")
            # Fallback to simple abbreviation
            abbr_col = f"col_{len(abbr_col_names)}"
            abbr_col_names[col_name] = abbr_col
    
    return abbr_col_names
