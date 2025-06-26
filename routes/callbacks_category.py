"""
Category selection callbacks for the Dash app.
"""

import json
import dash
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate


def register_category_callbacks(app):
    """Register all category-related callbacks."""
    
    @app.callback(
        [Output("category-selection-modal", "is_open"),
         Output("category-modal-column-name", "children"),
         Output("category-dropdown-modal", "value")],
        [Input("column-mapping-datatable", "active_cell")],
        [State("column-mapping-datatable", "data"),
         State("category-selection-modal", "is_open")]
    )
    def toggle_category_modal(active_cell, table_data, is_open):
        """Open category modal when Category column is clicked."""
        if active_cell and active_cell["column_id"] == "Category":
            # Get the row
            row_idx = active_cell["row"]
            row_data = table_data[row_idx]
            
            # Get column name and current category
            col_name = row_data["Original Column"]
            current_category = row_data.get("Category", "")
            
            return True, html.H5(f"Select category for column: {col_name}"), current_category
        
        return False, "", ""

    @app.callback(
        Output("current-category-column-index", "data"),
        [Input("category-selection-modal", "is_open"),
         Input("column-mapping-datatable", "active_cell")],
    )
    def store_current_column_index(is_open, active_cell):
        """Store the current column index when modal is opened."""
        if is_open and active_cell and active_cell["column_id"] == "Category":
            return active_cell["row"]
        return None

    @app.callback(
        Output("category-selection-modal", "is_open", allow_duplicate=True),
        [Input("cancel-category-button", "n_clicks")],
        prevent_initial_call=True
    )
    def close_category_modal(n_clicks):
        """Close category modal without saving."""
        if n_clicks:
            return False
        return dash.no_update

    @app.callback(
        [Output("category-selection-modal", "is_open", allow_duplicate=True),
         Output("column-mapping-datatable", "data", allow_duplicate=True),
         Output("snomed-mappings", "data", allow_duplicate=True)],
        [Input("save-category-button", "n_clicks")],
        [State("current-category-column-index", "data"),
         State("category-dropdown-modal", "value"),
         State("column-mapping-datatable", "data"),
         State("snomed-mappings", "data")],
        prevent_initial_call=True
    )
    def save_category(n_clicks, row_idx, category_value, table_data, mappings_json):
        """Save the selected category."""
        if not n_clicks or row_idx is None:
            return False, dash.no_update, dash.no_update
        
        # Update the category in the table data
        updated_data = table_data.copy()
        col_name = updated_data[row_idx]["Original Column"]
        updated_data[row_idx]["Category"] = category_value
        
        # Update the mappings if they exist
        if mappings_json:
            try:
                mappings = json.loads(mappings_json)
                
                # Update category for this column in all entity mappings
                if col_name in mappings:
                    for entity in mappings[col_name]:
                        mappings[col_name][entity]["category"] = category_value
                        
                updated_mappings = json.dumps(mappings)
            except Exception as e:
                print(f"Error updating mappings: {e}")
                updated_mappings = mappings_json
        else:
            updated_mappings = mappings_json
        
        return False, updated_data, updated_mappings
