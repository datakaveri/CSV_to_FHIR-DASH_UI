"""
Entity-related callbacks for the Dash app.
Handles entity search tab display only (mapping functionality moved to callbacks_mapping_edit.py).
"""

import json
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State
from dash.exceptions import PreventUpdate


def register_entity_callbacks(app):
    """Register entity-related callbacks."""
    
    @app.callback(
        Output("entity-search-content", "children"),
        [Input("tabs", "active_tab"),
         Input("column-mapping-datatable", "data")],
        [State("adarv-entities", "data")]
    )
    def update_entity_search_tab(active_tab, table_data, adarv_entities_json):
        """Update the entity search tab content."""
        if active_tab != "tab-entity-search" or not table_data:
            raise PreventUpdate
        
        try:
            adarv_entities = json.loads(adarv_entities_json) if adarv_entities_json else {}
            
            search_content = []
            
            for idx, row in enumerate(table_data):
                col_name = row["Original Column"]
                category = row.get("Category", "")
                fhir_resource = row.get("FHIR Resource", "observation")
                entities_text = row.get("Extracted Entities", "None")
                
                # Parse extracted entities
                entities = []
                if entities_text and entities_text != "None":
                    entities = [e.strip() for e in entities_text.split(",") if e.strip()]
                
                # Get entities from ADARV data if available
                if col_name in adarv_entities and 'entities' in adarv_entities[col_name]:
                    adarv_entity_list = adarv_entities[col_name]['entities']
                    entities.extend([e for e in adarv_entity_list if e not in entities])
                
                # If no entities found, use column name as default
                if not entities:
                    entities = [col_name]
                
                # Create a card for this column
                card = dbc.Card([
                    dbc.CardHeader([
                        html.H6(f"Column: {col_name}", className="mb-0"),
                        html.Small(f"Category: {category} | FHIR Resource: {fhir_resource}", 
                                 className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.P(f"Extracted entities: {', '.join(entities)}", className="mb-2"),
                        html.P("Use the 'Edit Mappings' button in the Column Mapping tab to map these entities to SNOMED codes.", 
                               className="text-info")
                    ])
                ], className="mb-3")
                
                search_content.append(card)
            
            return html.Div([
                html.H5("Entity Search and SNOMED Mapping"),
                html.P("This tab shows the extracted entities for each column. Use the 'Edit Mappings' button in the Column Mapping tab to map entities to SNOMED codes."),
                html.Div(search_content)
            ])
            
        except Exception as e:
            print(f"Error updating entity search tab: {e}")
            return html.Div(f"Error: {str(e)}")
