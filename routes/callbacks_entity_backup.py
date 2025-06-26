"""
Entity search and SNOMED mapping callbacks for the Dash app.
"""

import json
import re
import requests
import traceback
import pandas as pd
import dash
from dash import Input, Output, State, html, ALL, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from snomed.manual_updater import process_manual_snomed_selection
from routes.helpers import sanitize_for_json, format_snomed_result
from config import API_ENDPOINT
from snomed.manual_updater import process_manual_snomed_selection


def register_entity_callbacks(app):
    """Register all entity-related callbacks."""
    
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
                
                # Get entities from ADARV data if available
                entities = []
                if col_name in adarv_entities and 'entities' in adarv_entities[col_name]:
                    entities = adarv_entities[col_name]['entities']
                else:
                    # Create a default entity from column name
                    entities = [col_name]
                
                # Create entity buttons for this column
                entity_buttons = []
                for entity in entities:
                    button = dbc.Button(
                        f"Search: {entity}",
                        id={
                            "type": "entity-button",
                            "index": idx,
                            "col": col_name,
                            "entity": entity,
                            "fhir": fhir_resource
                        },
                        color="outline-primary",
                        size="sm",
                        className="me-2 mb-2"
                    )
                    entity_buttons.append(button)
                
                # Create a card for this column
                card = dbc.Card([
                    dbc.CardHeader([
                        html.H6(f"Column: {col_name}", className="mb-0"),
                        html.Small(f"Category: {category} | FHIR Resource: {fhir_resource}", 
                                 className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.P(f"Entities to map: {', '.join(entities)}", className="mb-2"),
                        html.Div(entity_buttons)
                    ])
                ], className="mb-3")
                
                search_content.append(card)
            
            return html.Div([
                html.H5("Entity Search and SNOMED Mapping"),
                html.P("Click on entity buttons below to search for SNOMED codes:"),
                html.Div(search_content)
            ])
            
        except Exception as e:
            print(f"Error updating entity search tab: {e}")
            return html.Div(f"Error: {str(e)}")

    # Add callback for entity button clicks to open modal
    @app.callback(
        [Output("modal-entity-info", "children", allow_duplicate=True),
         Output("modal-search-results", "children", allow_duplicate=True),
         Output("entity-search-modal", "is_open", allow_duplicate=True),
         Output("manual-snomed-results", "children", allow_duplicate=True)],
        [Input({"type": "entity-button", "index": ALL, "col": ALL, "entity": ALL, "fhir": ALL}, "n_clicks")],
        [State("modal-entity-info", "children"),
         State("snomed-mappings", "data")],
        prevent_initial_call=True
    )
    def toggle_entity_button(n_clicks_list, current_info, mappings_json):
        """Handle entity button clicks to open modal and search."""
        # Check which button was clicked
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        try:
            # Parse the button ID that was clicked
            button_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
            
            entity = button_id["entity"]
            col_name = button_id["col"]
            fhir_resource = button_id["fhir"]
            category = ""  # Default category
            
            # Create entity info for the modal
            entity_info = html.Div([
                html.P(f"Searching for: {entity}"),
                html.P(f"Column: {col_name}"),
                html.P(f"FHIR Resource: {fhir_resource}"),
                html.Div(id="selected-entity-tracker", 
                        children=json.dumps({
                            "entity": entity,
                            "col": col_name,
                            "fhir": fhir_resource,
                            "category": category
                        }),
                        style={"display": "none"})
            ])
            
            # Make API request to search for SNOMED codes
            try:
                payload = {
                    "keywords": entity,
                    "threshold": 0.5,
                    "limit": 10
                }
                
                response = requests.post(API_ENDPOINT, json=payload)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    if results and isinstance(results, list):
                        search_results = []
                        for result in results:
                            # Create formatted result card
                            formatted_result = format_snomed_result(
                                result, entity, col_name, fhir_resource, category
                            )
                            search_results.append(formatted_result)
                        
                        return entity_info, search_results, True, None
                    else:
                        return entity_info, [html.P("No results found.")], True, None
                else:
                    return entity_info, [html.P(f"API Error: {response.status_code}")], True, None
                    
            except Exception as e:
                return entity_info, [html.P(f"Error: {str(e)}")], True, None
                
        except Exception as e:
            print(f"Error in toggle_entity_button: {e}")
            traceback.print_exc()
            raise PreventUpdate

    # Add callback for manual SNOMED code lookup
    @app.callback(
        Output("manual-snomed-results", "children"),
        [Input("lookup-snomed-code-button", "n_clicks")],
        [State("manual-snomed-code-input", "value"),
         State("modal-entity-info", "children"),
         State("snomed-mappings", "data")],
        prevent_initial_call=True
    )
    def lookup_snomed_code(n_clicks, code, entity_info, mappings_json):
        """Look up a manually entered SNOMED code."""
        if not n_clicks or not code:
            return None
        
        # Clean up the input - remove spaces and non-numeric characters
        code = re.sub(r'[^\d]', '', code.strip())
        
        if not code:
            return html.Div(
                dbc.Alert("Please enter a valid numeric SNOMED code", color="danger"),
                className="mt-2"
            )
        
        try:
            # Extract entity, column, and FHIR resource from the entity-info element
            col_name = ""
            active_entity = None
            fhir_resource = "observation"
            category = ""
            
            # Look for the hidden tracker div that stores the current selected entity
            if entity_info and len(entity_info) > 0:
                for child in entity_info:
                    if hasattr(child, 'id') and child.id == "selected-entity-tracker":
                        try:
                            tracker_data = json.loads(child.children)
                            active_entity = tracker_data.get("entity", "")
                            col_name = tracker_data.get("col", "")
                            fhir_resource = tracker_data.get("fhir", "observation")
                            category = tracker_data.get("category", "")
                            break
                        except:
                            continue
            
            if not active_entity or not col_name:
                return html.Div(
                    dbc.Alert("Could not determine entity or column information", color="danger"),
                    className="mt-2"
                )
            
            # Make API request to validate the SNOMED code
            try:
                payload = {
                    "keywords": code,
                    "threshold": 0.0,
                    "limit": 1
                }
                
                response = requests.post(API_ENDPOINT, json=payload)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    if results and isinstance(results, list) and len(results) > 0:
                        result = results[0]
                        concept_id = result.get("conceptid", "")
                        concept_name = result.get("conceptid_name", "")
                        
                        if concept_id == code:
                            # Valid SNOMED code found
                            return html.Div([
                                dbc.Alert("Valid SNOMED code found!", color="success"),
                                format_snomed_result(result, active_entity, col_name, fhir_resource, category)
                            ])
                        else:
                            return html.Div(
                                dbc.Alert(f"SNOMED code {code} not found in database", color="warning"),
                                className="mt-2"
                            )
                    else:
                        return html.Div(
                            dbc.Alert(f"SNOMED code {code} not found", color="warning"),
                            className="mt-2"
                        )
                else:
                    return html.Div(
                        dbc.Alert(f"API Error: {response.status_code}", color="danger"),
                        className="mt-2"
                    )
                    
            except Exception as e:
                return html.Div(
                    dbc.Alert(f"Error validating SNOMED code: {str(e)}", color="danger"),
                    className="mt-2"
                )
                
        except Exception as e:
            print(f"Error in lookup_snomed_code: {e}")
            traceback.print_exc()
            return html.Div(
                dbc.Alert(f"Unexpected error: {str(e)}", color="danger"),
                className="mt-2"
            )

    # Add callback for selecting SNOMED codes
    @app.callback(
        [Output("snomed-mappings", "data", allow_duplicate=True),
         Output("column-mapping-datatable", "data", allow_duplicate=True),
         Output("renamed-columns", "data", allow_duplicate=True),
         Output("entity-search-modal", "is_open", allow_duplicate=True)],
        [Input({"type": "manual-select-button", "snomed_id": ALL, "snomed_name": ALL, 
                "col": ALL, "fhir": ALL, "category": ALL, "entity": ALL}, "n_clicks")],
        [State("snomed-mappings", "data"),
         State("column-mapping-datatable", "data"),
         State("stored-data", "data"),
         State("renamed-columns", "data"),
         State("adarv-entities", "data")],
        prevent_initial_call=True
    )
    def select_manual_snomed_code(n_clicks_list, mappings_json, table_data, json_data, existing_renamed_cols, adarv_entities_json):
        """Handle selection of a SNOMED code from the modal."""
        # Check which button was clicked
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Check if any button was actually clicked (n_clicks > 0)
        button_clicked = False
        for n_clicks in n_clicks_list:
            if n_clicks and n_clicks > 0:
                button_clicked = True
                break
        
        if not button_clicked:
            raise PreventUpdate
        
        # Extract the button ID that was clicked
        button_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
        
        # Get mapping information from the button ID
        col_name = button_id["col"]
        snomed_id = button_id["snomed_id"]
        snomed_name = button_id["snomed_name"]
        fhir_resource = button_id["fhir"]
        category = button_id["category"]
        entity = button_id["entity"]
        
        # Process manual SNOMED selection for database update
        try:
            process_manual_snomed_selection(
                col_name=col_name,
                entity=entity,
                snomed_id=snomed_id,
                snomed_name=snomed_name,
                fhir_resource=fhir_resource,
                category=category
            )
        except Exception as e:
            print(f"Error processing manual SNOMED selection: {e}")
        
        # Use the snomed_name directly from the button
        best_term = snomed_name
        
        # Update mappings
        if not mappings_json:
            mappings = {}
        else:
            try:
                mappings = json.loads(mappings_json)
            except:
                mappings = {}
        
        # Initialize column mapping if it doesn't exist
        if col_name not in mappings:
            mappings[col_name] = {}
        
        # Add or update the mapping for the selected entity
        mappings[col_name][entity] = {
            "snomed_id": snomed_id,
            "snomed_name": best_term,
            "snomed_full_name": snomed_name,
            "fhir_resource": fhir_resource,
            "category": category
        }
          # Update the table data to show the new mapping
        updated_table_data = table_data.copy()
        for row in updated_table_data:
            if row["Original Column"] == col_name:
                # Update the mappings count and display
                column_mappings = mappings.get(col_name, {})
                num_mappings = len(column_mappings)
                
                if num_mappings > 0:
                    mapping_summary = []
                    for ent, mapping in column_mappings.items():
                        mapping_summary.append(f"{ent}: {mapping['snomed_name']}")
                    row["Current Mappings"] = f"{num_mappings} mapping(s): " + "; ".join(mapping_summary[:2])
                    if len(mapping_summary) > 2:
                        row["Current Mappings"] += f" (+{len(mapping_summary) - 2} more)"
                else:
                    row["Current Mappings"] = "No mappings"
                break
        
        # Generate updated column names
        try:
            from routes.helpers import generate_column_names, process_unified_api_response
            
            df = pd.read_json(json_data)
            columns_list = df.columns.tolist()
            
            # Get processed columns data
            adarv_entities = json.loads(adarv_entities_json) if adarv_entities_json else {}
            processed_columns, _ = process_unified_api_response(columns_list)
            
            # Generate new column names
            new_renamed_cols = generate_column_names(processed_columns, columns_list)
            updated_renamed_cols = json.dumps(new_renamed_cols)
            
        except Exception as e:
            print(f"Error updating column names: {e}")
            updated_renamed_cols = existing_renamed_cols
        
        # Store the abbreviated name in each entity's mapping for reference
        for col, entities_map in mappings.items():
            for entity_name, entity_mapping in entities_map.items():
                entity_mapping["abbreviated_name"] = new_renamed_cols.get(col, col)
        
        # Close the modal - user has selected a code
        return json.dumps(mappings), updated_table_data, updated_renamed_cols, False

    # Add callback to close the modal
    @app.callback(
        Output("entity-search-modal", "is_open", allow_duplicate=True),
        Input("modal-close", "n_clicks"),
        prevent_initial_call=True
    )
    def close_modal(n_clicks):
        """Close the entity search modal."""
        return False


def search_snomed_codes(entity):
    """Search for SNOMED codes for a given entity."""
    try:
        # API endpoint for SNOMED search
        API_ENDPOINT = "https://entitymapper.adarv.in/search"
        
        # Make API request
        payload = {
            "keywords": entity,
            "threshold": 0.5,
            "limit": 10
        }
        
        response = requests.post(API_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            results = json.loads(response.text)
            
            if results:
                search_results = []
                for idx, result in enumerate(results[:10]):  # Limit to top 10
                    score = result.get("score", 0)
                    concept_id = result.get("ConceptID", "")
                    concept_name = result.get("ConceptID_name", "")
                    
                    # Create a result card
                    card = dbc.Card([
                        dbc.CardBody([
                            html.H6(f"Score: {score:.3f}", className="card-title"),
                            html.P(f"SNOMED ID: {concept_id}", className="text-muted"),
                            html.P(concept_name, className="card-text"),
                            dbc.Button(
                                "Select",
                                id={
                                    "type": "manual-select-button",
                                    "snomed_id": concept_id,
                                    "snomed_name": concept_name,
                                    "col": "column_placeholder",
                                    "fhir": "fhir_placeholder",
                                    "category": "category_placeholder",
                                    "entity": entity
                                },
                                color="primary",
                                size="sm"
                            )
                        ])
                    ], className="mb-2")
                    
                    search_results.append(card)
                
                return html.Div(search_results)
            else:
                return html.Div("No results found.")
        else:
            return html.Div(f"Search failed with status {response.status_code}")
            
    except Exception as e:
        print(f"Error searching SNOMED codes: {e}")
        return html.Div(f"Error: {str(e)}")
