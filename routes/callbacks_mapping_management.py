"""
Mapping management callbacks for the Dash app.
Handles CRUD operations on SNOMED mappings.
"""

import json
import traceback
from dash import Input, Output, State
from dash.exceptions import PreventUpdate


def register_mapping_management_callbacks(app):
    """Register mapping management related callbacks."""
    
    @app.callback(
        [Output("snomed-mappings", "data", allow_duplicate=True),
         Output("renamed-columns", "data", allow_duplicate=True),
         Output("column-mapping-datatable", "data", allow_duplicate=True)],
        [Input("final-mapping-datatable", "active_cell")],
        [State("final-mapping-datatable", "data"),
         State("snomed-mappings", "data"),
         State("renamed-columns", "data"),
         State("column-mapping-datatable", "data")],
        prevent_initial_call=True
    )
    def delete_mapping_row(active_cell, table_data, mappings_json, renamed_cols_json, column_mapping_data):
        """Delete a mapping row when the Delete action is clicked."""
        # Check if a cell in the Action column was clicked
        if not active_cell or active_cell["column_id"] != "Action":
            raise PreventUpdate
        
        # Get the row that was clicked
        row_idx = active_cell["row"]
        row_data = table_data[row_idx]
        
        # Only proceed if the Action cell contains "Delete"
        if row_data["Action"] != "Delete":
            raise PreventUpdate
        
        try:
            # Extract the information we need
            col_name = row_data["Column"]
            entity = row_data["Entity"]
            
            # Load the current mappings
            mappings = json.loads(mappings_json)
            
            # Check if this column and entity exist in the mappings
            if col_name not in mappings or entity not in mappings[col_name]:
                raise PreventUpdate
                
            # Load existing renamed columns
            try:
                renamed_cols = json.loads(renamed_cols_json)
            except:
                renamed_cols = {}
            
            # Get the current renamed column name to understand its structure
            current_renamed = renamed_cols.get(col_name, col_name)
            
            # Get entity information before removing it
            entity_snomed_name = ""
            if col_name in mappings and entity in mappings[col_name]:
                entity_snomed_name = mappings[col_name][entity].get("snomed_name", "")
                if entity_snomed_name:
                    # Get first word and take first 3 chars (same logic used for naming)
                    first_word = entity_snomed_name.split(',')[0].strip()
                    entity_snomed_name = first_word.lower().replace(' ', '').replace('(', '').replace(')', '')
                    entity_snomed_name = entity_snomed_name[:3]  # Just the abbreviation
            
            # Remove this entity from the mappings
            del mappings[col_name][entity]
            
            # Handle renaming based on remaining entities
            if not mappings[col_name]:
                # If no more entities for this column, remove the column entirely
                del mappings[col_name]
                # Keep the original column name or mark as unmapped
                if col_name in renamed_cols:
                    del renamed_cols[col_name]
            else:
                # Only modify the name if we found a valid entity abbreviation to remove
                if entity_snomed_name and col_name in renamed_cols:
                    current_name = renamed_cols[col_name]
                    
                    # Simply remove the entity's abbreviation from the name
                    # Look for _abbreviation_ or _abbreviation at the end
                    part_to_remove = f"_{entity_snomed_name}_"
                    end_part = f"_{entity_snomed_name}"
                    
                    if part_to_remove in current_name:
                        # Replace the part in the middle with a single underscore
                        new_name = current_name.replace(part_to_remove, "_")
                        renamed_cols[col_name] = new_name
                    elif current_name.endswith(end_part):
                        # Remove from the end
                        new_name = current_name[:-len(end_part)]
                        renamed_cols[col_name] = new_name
                    # If we couldn't find the exact abbreviation, keep the current name
            
            # Update the SNOMED Match column in the column mapping table
            updated_column_mapping = column_mapping_data.copy()
            for row in updated_column_mapping:
                if row["Original Column"] == col_name:
                    if col_name in mappings:
                        # Collect all remaining mappings for this column
                        match_info = []
                        for ent, match in mappings[col_name].items():
                            match_info.append(f"{ent}: {match['snomed_id']} - {match['snomed_name']}")
                        row["SNOMED Match"] = "; ".join(match_info)
                    else:
                        # No mappings left for this column
                        row["SNOMED Match"] = "No matches found"
                    break
            
            # Return the updated data
            return json.dumps(mappings), json.dumps(renamed_cols), updated_column_mapping
            
        except Exception as e:
            print(f"Error deleting mapping: {e}")
            traceback.print_exc()
            raise PreventUpdate
