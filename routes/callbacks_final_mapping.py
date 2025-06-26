"""
Final mapping display callbacks for the Dash app.
Handles the final mapping tab display and export functionality.
"""

import json
import traceback
from dash import Input, Output, State, dash_table, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


def register_final_mapping_callbacks(app):
    """Register final mapping display related callbacks."""
    
    @app.callback(
        Output("final-mapping-content", "children"),
        [Input("snomed-mappings", "data"),
         Input("renamed-columns", "data"),
         Input("stored-data", "data")],
        prevent_initial_call=True
    )
    def update_final_mapping_tab(mappings_json, renamed_cols_json, data_json):
        """Update the final mapping tab content."""
        if not mappings_json or not data_json:
            return html.P("Complete the entity mapping process in the previous tabs first.")
        
        try:
            mappings = json.loads(mappings_json)
            
            # Get renamed columns if available
            renamed_cols = {}
            if renamed_cols_json:
                try:
                    renamed_cols = json.loads(renamed_cols_json)
                except:
                    pass
            
            if not mappings:
                return html.Div([
                    html.P("No mappings created yet. Go to the Entity Search tab to create mappings."),
                    html.Hr(),
                    dbc.Button("Export Mappings", id="export-button", 
                              color="primary", disabled=True),
                    html.Div(id="download-mappings")
                ])
            
            # Create rows for the final mapping table - one row per entity (not per column)
            rows = []
            for col_name, entities_map in mappings.items():
                # Get the renamed column from our dynamically generated names
                renamed_column = renamed_cols.get(col_name, col_name)
                
                for entity, mapping in entities_map.items():
                    rows.append({
                        "Column": col_name,
                        "Renamed Column": renamed_column,
                        "Entity": entity,
                        "SNOMED ID": mapping.get("snomed_id", ""),
                        "SNOMED Name": mapping.get("snomed_name", ""),
                        "FHIR Resource": mapping.get("fhir_resource", ""),
                        "Category": mapping.get("category", ""),
                        "Action": "Delete"  # Add Delete action
                    })
            
            # Create the table - now including Delete action column
            mapping_table = dash_table.DataTable(
                id='final-mapping-datatable',
                columns=[
                    {"name": "Column", "id": "Column"},
                    {"name": "Renamed Column", "id": "Renamed Column"},
                    {"name": "Entity", "id": "Entity"},
                    {"name": "SNOMED ID", "id": "SNOMED ID"},
                    {"name": "SNOMED Name", "id": "SNOMED Name"},
                    {"name": "FHIR Resource", "id": "FHIR Resource"},
                    {"name": "Category", "id": "Category"},
                    {"name": "Action", "id": "Action"}  # Add Delete column
                ],
                data=rows,
                style_table={
                    'overflowX': 'auto',
                    'maxHeight': '500px',
                    'overflowY': 'auto'
                },
                style_cell={
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'maxWidth': 0,
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Renamed Column'},
                        'fontWeight': 'bold',
                        'backgroundColor': '#f0f8ff'  # Light blue background to highlight
                    },
                    {
                        'if': {'column_id': 'Action'},
                        'cursor': 'pointer',
                        'color': 'red',
                        'textDecoration': 'underline'
                    }
                ],
                filter_action="native",
                sort_action="native"
            )
            
            # Create a summary view showing the mappings by column
            column_summary = []
            for col_name, entities_map in mappings.items():
                # Get the renamed column from our dynamically generated names
                renamed_column = renamed_cols.get(col_name, col_name)
                
                # Get category from first entity mapping
                category = ""
                if entities_map:
                    first_entity = next(iter(entities_map.values()))
                    category = first_entity.get("category", "")
                
                # Create entity items for this column
                entity_items = []
                for entity, mapping in entities_map.items():
                    entity_items.append(html.Div([
                        html.Span(f"{entity}: ", className="font-weight-bold"),
                        html.Span(f"{mapping['snomed_id']} - {mapping['snomed_name']}"),
                        html.Span(f" ({mapping['fhir_resource']})", className="text-muted")
                    ], className="mb-1"))
                
                # Create card for this column
                column_summary.append(
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.Span(f"Original: {col_name}"),
                                html.Br(),
                                html.Span(f"Renamed: {renamed_column}", className="text-info")
                            ]),
                            html.Span(
                                dbc.Badge(category, color="info", className="ml-2 float-right"),
                                style={"marginTop": "5px"}
                            ) if category else None
                        ]),
                        dbc.CardBody(entity_items)
                    ], className="mb-3")
                )
            
            # Add the export functionality
            return html.Div([
                html.H5("SNOMED CT Mappings Summary"),
                html.Div(column_summary) if column_summary else html.P("No mappings created yet."),
                
                html.H5("All Mappings", className="mt-4"),
                html.P("Click 'Delete' to remove any unwanted mappings. Column renaming will be updated automatically."),
                mapping_table,
                
                html.Hr(),
                dbc.Button("Export Mappings", id="export-button", 
                          color="primary", className="mt-3"),
                html.Div(id="download-mappings")
            ])
            
        except Exception as e:
            print(f"Error updating final mapping tab: {e}")
            traceback.print_exc()
            return html.P(f"Error generating final mapping: {str(e)}")
