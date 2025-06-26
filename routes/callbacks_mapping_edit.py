"""
Edit mapping callbacks for the Dash app.
Handles the edit mapping modal and SNOMED search functionality.
"""

import json
import traceback
import requests
import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
from routes.text_utils import split_entities, sanitize_for_json
from snomed.manual_updater import process_manual_snomed_selection
from config import API_ENDPOINT, DB_CONFIG


def register_mapping_edit_callbacks(app):
    """Register edit mapping related callbacks."""
    
    @app.callback(
        [Output("entity-search-modal", "is_open", allow_duplicate=True),
         Output("modal-entity-info", "children", allow_duplicate=True),
         Output("manual-snomed-code-input", "value")],
        [Input("column-mapping-datatable", "active_cell")],
        [State("column-mapping-datatable", "data"),
         State("snomed-mappings", "data")],
        prevent_initial_call=True
    )
    def open_entity_edit_modal(active_cell, table_data, mappings_json):
        """Open the entity edit modal when 'Edit Mappings' action is clicked."""
        if active_cell and active_cell["column_id"] == "Actions":
            # Get the row
            row_idx = active_cell["row"]
            row_data = table_data[row_idx]
            
            # Get column information
            col_name = row_data["Original Column"]
            fhir_resource = row_data["FHIR Resource"]
            category = row_data.get("Category", "")
            
            # Get the entities list directly from the row data
            entities_list = []
            if "entities_list_json" in row_data:
                try:
                    entities_list = json.loads(row_data["entities_list_json"])
                except:
                    # Fallback if JSON parsing fails
                    pass
            
            if not entities_list and row_data.get("Extracted Entities", "None") != "None":
                # Split entities from text
                raw_entities = row_data["Extracted Entities"]
                entities_list = split_entities(raw_entities)
            
            if not entities_list:
                return True, html.Div([
                    html.H5(f"Column: {col_name}"),
                    html.H6(f"FHIR Resource: {fhir_resource}", className="text-muted"),
                    html.H6(f"Category: {category}", className="text-muted") if category else None,
                    html.P("No entities were extracted for this column. Please extract entities first.")
                ]), ""
            
            # Get already mapped entities from mappings
            mapped_entities = {}
            if mappings_json:
                try:
                    mappings = json.loads(mappings_json)
                    if col_name in mappings:
                        mapped_entities = mappings[col_name]
                except:
                    # If there's an error parsing mappings, continue with empty dict
                    pass
            
            # Create entity buttons for each individual entity
            entity_buttons = []
            for i, entity in enumerate(entities_list):
                # Sanitize values for JSON
                safe_entity = sanitize_for_json(entity)
                safe_col_name = sanitize_for_json(col_name)
                safe_fhir = sanitize_for_json(fhir_resource)
                
                # Check if this entity is already mapped
                is_mapped = entity in mapped_entities
                
                # Add mapping details to button text if it's mapped
                button_text = entity
                if is_mapped:
                    mapping = mapped_entities[entity]
                    button_text = f"{entity} → {mapping['snomed_id']} ({mapping['snomed_name']})"
                
                # Use outline for mapped entities but make the first one have success color for selection
                outline = is_mapped
                color = "primary"
                if i == 0:  # Make the first entity selected initially
                    color = "success"
                    outline = False
                elif is_mapped:
                    color = "success"
                    outline = True
                
                entity_buttons.append(
                    dbc.Button(
                        button_text,
                        id={"type": "entity-button", 
                            "index": i, 
                            "col": safe_col_name,
                            "entity": safe_entity, 
                            "fhir": safe_fhir},
                        color=color,
                        className="me-2 mb-2",
                        outline=outline
                    )
                )
            
            # Add instructions for the manual SNOMED code entry
            manual_lookup_instructions = html.Div([
                html.P([
                    "Select an entity from above, then either: ",
                    html.Br(),
                    "1. Use the search results below, or", 
                    html.Br(),
                    "2. Enter a SNOMED code directly and click 'Look Up'"
                ], className="mt-2 mb-2 text-info")
            ])
            
            entity_info = [
                html.H5(f"Edit SNOMED Mappings for Column: {col_name}"),
                html.H6(f"FHIR Resource: {fhir_resource}", className="text-muted"),
                html.H6(f"Category: {category}", className="text-muted") if category else None,
                html.P("Click on an entity to map it to a SNOMED code:"),
                html.Div(entity_buttons, style={"display": "flex", "flexWrap": "wrap", "gap": "5px"}),
                manual_lookup_instructions
            ]
            
            # Filter out None values from entity_info
            entity_info = [item for item in entity_info if item is not None]
            
            return True, entity_info, ""
        
        return False, None, ""

    @app.callback(
        Output("entity-search-modal", "is_open", allow_duplicate=True),
        Input("modal-close", "n_clicks"),
        prevent_initial_call=True
    )
    def close_modal(n_clicks):
        """Close the entity edit modal."""
        return False

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
        """Handle entity button clicks and perform SNOMED search."""
        # Check which button was clicked
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        try:
            # Extract the button ID that was clicked with robust parsing
            button_id_str = ctx.triggered[0]["prop_id"].split(".")[0]
            print(f"Raw button ID string: {repr(button_id_str)}")
            
            button_id = None
            
            try:
                # Clean the string first before JSON parsing
                import re
                
                # Replace any non-printable characters and normalize spaces
                cleaned_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', button_id_str)
                cleaned_str = re.sub(r'[\u00a0\u2000-\u200f\u2028-\u202f\u205f-\u206f]', ' ', cleaned_str)
                cleaned_str = re.sub(r'\s+', ' ', cleaned_str)
                
                button_id = json.loads(cleaned_str)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                
                # Manual regex extraction as fallback
                type_match = re.search(r'"type"\s*:\s*"([^"]*)"', button_id_str)
                index_match = re.search(r'"index"\s*:\s*(\d+)', button_id_str)
                col_match = re.search(r'"col"\s*:\s*"([^"]*)"', button_id_str)
                entity_match = re.search(r'"entity"\s*:\s*"([^"]*)"', button_id_str)
                fhir_match = re.search(r'"fhir"\s*:\s*"([^"]*)"', button_id_str)
                
                button_id = {
                    "type": type_match.group(1) if type_match else "entity-button",
                    "index": int(index_match.group(1)) if index_match else 0,
                    "col": col_match.group(1) if col_match else "",
                    "entity": entity_match.group(1) if entity_match else "",
                    "fhir": fhir_match.group(1) if fhir_match else "observation"
                }
            
            if not button_id:
                raise PreventUpdate
            
            # Sanitize all extracted values
            selected_entity = sanitize_for_json(button_id.get("entity", ""))
            col_name = sanitize_for_json(button_id.get("col", ""))
            fhir_resource = sanitize_for_json(button_id.get("fhir", ""))
            selected_index = button_id.get("index", 0)
            
            # Get already mapped entities from mappings
            mapped_entities = {}
            if mappings_json:
                try:
                    mappings = json.loads(mappings_json)
                    if col_name in mappings:
                        mapped_entities = mappings[col_name]
                except Exception as e:
                    print(f"Error parsing mappings: {e}")
                    pass
            
            # Get the existing entity info structure
            if not current_info:
                return dash.no_update, dash.no_update, True, None
            
            # Extract the existing elements and update them
            updated_info = []
            entity_buttons_container = None
            
            for item in current_info if isinstance(current_info, list) else [current_info]:
                if isinstance(item, dict):
                    if item.get('props', {}).get('style', {}).get('display') == 'flex':
                        entity_buttons_container = item
                        break
            
            if entity_buttons_container:
                # Extract all buttons and update their properties
                buttons = entity_buttons_container.get('props', {}).get('children', [])
                updated_buttons = []
                
                for button in buttons if isinstance(buttons, list) else [buttons]:
                    if not isinstance(button, dict) or 'props' not in button:
                        continue
                        
                    button_props = button.get('props', {})
                    button_id_props = button_props.get('id', {})
                    entity = sanitize_for_json(button_id_props.get('entity', ''))
                    
                    # Check if this is the selected button
                    is_selected = (entity == selected_entity)
                    # Check if this entity is mapped
                    is_mapped = entity in mapped_entities
                    
                    # Update button properties
                    new_button_props = button_props.copy()
                    
                    # Update button text if needed
                    button_text = entity
                    if is_mapped:
                        mapping = mapped_entities[entity]
                        snomed_id = sanitize_for_json(str(mapping.get('snomed_id', '')))
                        snomed_name = sanitize_for_json(mapping.get('snomed_name', ''))
                        source = mapping.get('source', '')
                        source_label = f" [{source}]" if source else ""
                        button_text = f"{entity} → {snomed_id} ({snomed_name}){source_label}"
                    
                    new_button_props['children'] = button_text
                    
                    # Update button color and outline
                    if is_selected:
                        new_button_props['color'] = 'success'
                        new_button_props['outline'] = False
                    else:
                        new_button_props['color'] = 'success' if is_mapped else 'primary'
                        new_button_props['outline'] = is_mapped
                    
                    # Sanitize the button ID properties before creating the updated button
                    sanitized_id = {
                        "type": "entity-button",
                        "index": button_id_props.get('index', 0),
                        "col": sanitize_for_json(button_id_props.get('col', '')),
                        "entity": sanitize_for_json(button_id_props.get('entity', '')),
                        "fhir": sanitize_for_json(button_id_props.get('fhir', ''))
                    }
                    new_button_props['id'] = sanitized_id
                    
                    # Create updated button
                    updated_button = button.copy()
                    updated_button['props'] = new_button_props
                    updated_buttons.append(updated_button)
                
                # Update the buttons in the container
                updated_container = entity_buttons_container.copy()
                updated_container['props'] = {
                    **entity_buttons_container.get('props', {}),
                    'children': updated_buttons
                }
                
                # Rebuild the info with updated container
                for item in current_info:
                    if isinstance(item, dict) and item.get('props', {}).get('style', {}).get('display') == 'flex':
                        updated_info.append(updated_container)
                    else:
                        updated_info.append(item)
                
                # Add a hidden field to track the current selected entity
                current_entity_tracker = html.Div(
                    id="current-selected-entity",
                    children=selected_entity,
                    style={"display": "none"}
                )
                updated_info.append(current_entity_tracker)
            else:
                updated_info = current_info
            
            # Get category from mappings if available
            category = ""
            if mappings_json:
                try:
                    mappings = json.loads(mappings_json)
                    if col_name in mappings and selected_entity in mappings[col_name]:
                        category = sanitize_for_json(mappings[col_name][selected_entity].get("category", ""))
                except Exception as e:
                    print(f"Error getting category: {e}")
                    pass
            
            # Check if the selected entity is already mapped by ADARV
            is_adarv_mapped = False
            existing_mapping = None
            if mappings_json:
                try:
                    mappings = json.loads(mappings_json)
                    if col_name in mappings and selected_entity in mappings[col_name]:
                        existing_mapping = mappings[col_name][selected_entity]
                        is_adarv_mapped = existing_mapping.get("source") == "ADARV"
                except Exception as e:
                    print(f"Error checking ADARV mapping: {e}")
                    pass
            
            # Only perform SNOMED search if the entity is NOT already mapped by ADARV
            if is_adarv_mapped:
                print(f"Entity {selected_entity} is already mapped by ADARV - skipping automatic SNOMED search")
                
                # Show a message instead of search results
                search_results = html.Div([
                    dbc.Alert([
                        html.H5("ADARV Mapping Active", className="alert-heading"),
                        html.P(f"This entity '{selected_entity}' is already mapped using ADARV data:"),
                        html.Hr(),
                        html.P([
                            html.Strong("SNOMED ID: "), existing_mapping.get('snomed_id', ''), html.Br(),
                            html.Strong("SNOMED Name: "), existing_mapping.get('snomed_name', ''), html.Br(),
                            html.Strong("FHIR Resource: "), existing_mapping.get('fhir_resource', ''), html.Br(),
                            html.Strong("Category: "), existing_mapping.get('category', '')
                        ]),
                        html.P([
                            "If you want to change this mapping, use the manual SNOMED code lookup above, ",
                            "or go to the Final Mapping tab to delete this mapping first."
                        ], className="mb-0")
                    ], color="info")
                ])
            else:
                # Perform a new SNOMED search for the selected entity
                try:
                    # Make API request to search for SNOMED codes
                    payload = {
                        "keywords": selected_entity,
                        "threshold": 0.5,
                        "limit": 20
                    }
                    
                    # Make request to the API
                    response = requests.post(API_ENDPOINT, json=payload)
                    
                    # Process the search results
                    if response.status_code == 200:
                        results = json.loads(response.text)
                        
                        if not results:
                            search_results = html.P(f"No SNOMED CT codes found for '{selected_entity}'")
                        else:
                            # Create result cards for the new entity
                            cards = []
                            
                            # Add a header for the results
                            cards.append(html.H5(f"Search results for: {selected_entity}"))
                            
                            # Create a card for each result
                            for i, result in enumerate(results):
                                # Validate the result has all required fields
                                if not all(k in result for k in ["conceptid", "conceptid_name", "toplevelhierarchy_name"]):
                                    continue
                                
                                # Sanitize all result data
                                conceptid = sanitize_for_json(str(result.get("conceptid", "")))
                                conceptid_name = sanitize_for_json(result.get("conceptid_name", ""))
                                toplevelhierarchy_name = sanitize_for_json(result.get("toplevelhierarchy_name", "Unknown"))
                                rrf_score = result.get("rrf_score", 0)
                                
                                # Check if this is the current mapping
                                is_selected_result = False
                                if existing_mapping and existing_mapping.get("snomed_id") == result.get("conceptid"):
                                    is_selected_result = True
                                
                                # Create the selection button with sanitized data
                                select_button = dbc.Button(
                                    "Selected" if is_selected_result else "Select",
                                    id={
                                        "type": "select-snomed-button",
                                        "index": i,
                                        "col": col_name,
                                        "entity": selected_entity,
                                        "fhir": fhir_resource,
                                        "snomed_id": conceptid,
                                        "snomed_name": conceptid_name,
                                        "category": category
                                    },
                                    color="success" if is_selected_result else "primary",
                                    size="sm",
                                    className="mt-2"
                                )
                                
                                # Create a card for this result
                                card = dbc.Card([
                                    dbc.CardBody([
                                        html.H5(f"{conceptid}: {conceptid_name}", className="card-title"),
                                        html.P(f"Hierarchy: {toplevelhierarchy_name}", className="card-text"),
                                        html.P(f"Match Score: {rrf_score:.2f}", className="card-text text-muted"),
                                        select_button
                                    ])
                                ], className="mb-3", color="light" if is_selected_result else None, outline=True)
                                
                                cards.append(card)
                            
                            search_results = html.Div(cards)
                    else:
                        search_results = html.P(f"API error ({response.status_code}) while searching for '{selected_entity}'")
                        
                except Exception as e:
                    print(f"Error searching for SNOMED codes: {e}")
                    import traceback
                    traceback.print_exc()
                    search_results = html.P(f"Error searching for SNOMED codes for '{selected_entity}': {str(e)}")
                    
            # Return the updated UI components and open the modal
            # Also clear the manual SNOMED results when selecting a new entity
            return updated_info, search_results, True, None
            
        except Exception as e:
            print(f"Error toggling entity button: {e}")
            import traceback
            traceback.print_exc()
            return dash.no_update, dash.no_update, True, None    @app.callback(
        [Output("snomed-mappings", "data", allow_duplicate=True),
         Output("column-mapping-datatable", "data", allow_duplicate=True),
         Output("renamed-columns", "data", allow_duplicate=True),
         Output("entity-search-modal", "is_open", allow_duplicate=True)],
        [Input({"type": "select-snomed-button", "index": ALL, "col": ALL, "entity": ALL, "fhir": ALL,
                "snomed_id": ALL, "snomed_name": ALL, "category": ALL}, "n_clicks")],
        [State("snomed-mappings", "data"),
         State("column-mapping-datatable", "data"),
         State("stored-data", "data"),
         State("renamed-columns", "data"),
         State("adarv-entities", "data")],
        prevent_initial_call=True
    )
    def select_snomed_from_search(n_clicks_list, mappings_json, table_data, json_data, existing_renamed_cols, adarv_entities_json):
        """Handle SNOMED code selection from search results."""
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
            print(f"\n SNOMED SELECTION FROM SEARCH")
            print(f"Entity: {entity}")
            print(f"SNOMED ID: {snomed_id}")
            print(f"SNOMED Name: {snomed_name}")
            
            # Execute real database update
            db_update_result = process_manual_snomed_selection(
                snomed_id, 
                snomed_name, 
                entity, 
                execute_update=True
            )
            
            # Log the result
            if db_update_result["success"]:
                print(f"SNOMED search selection processing completed successfully")
            else:
                print(f"SNOMED search selection processing failed: {db_update_result}")
                
        except Exception as e:
            print(f"Error in SNOMED search selection processing: {e}")
            traceback.print_exc()
        
        # Load current mappings
        mappings = {}
        if mappings_json:
            try:
                mappings = json.loads(mappings_json)
            except:
                pass
        
        # Load existing renamed columns
        try:
            renamed_cols = json.loads(existing_renamed_cols) if existing_renamed_cols else {}
        except:
            renamed_cols = {}
        
        # Add the new mapping
        if col_name not in mappings:
            mappings[col_name] = {}
        
        mappings[col_name][entity] = {
            "snomed_id": snomed_id,
            "snomed_name": snomed_name,
            "fhir_resource": fhir_resource,
            "category": category,
            "source": "Manual"
        }
        
        # Update the column mapping table
        updated_table_data = table_data.copy()
        for row in updated_table_data:
            if row["Original Column"] == col_name:
                # Collect all mappings for this column
                match_info = []
                for ent, match in mappings[col_name].items():
                    match_info.append(f"{ent}: {match['snomed_id']} - {match['snomed_name']}")
                row["SNOMED Match"] = "; ".join(match_info)
                break
        
        # Handle column renaming based on SNOMED mappings
        from routes.column_naming import generate_column_name
        new_name = generate_column_name(col_name, mappings.get(col_name, {}))
        if new_name != col_name:
            renamed_cols[col_name] = new_name
        
        # Close the modal after successful selection
        return json.dumps(mappings), updated_table_data, json.dumps(renamed_cols), False

    @app.callback(
        Output("manual-snomed-results", "children"),
        [Input("lookup-snomed-code-button", "n_clicks")],
        [State("manual-snomed-code-input", "value"),
         State("modal-entity-info", "children"),
         State("snomed-mappings", "data")],
        prevent_initial_call=True
    )
    def lookup_snomed_code(n_clicks, code, entity_info, mappings_json):
        """Handle manual SNOMED code lookup."""
        if not n_clicks or not code:
            return None
        
        # Clean up the input - remove spaces and non-numeric characters
        import re
        code = re.sub(r'[^\d]', '', code.strip())
        
        if not code:
            return html.Div(
                dbc.Alert("Please enter a valid numeric SNOMED code", color="danger"),
                className="mt-2"
            )
        
        try:
            from utils.embedding import initialize_emb_model
            
            # Extract entity, column, and FHIR resource from the entity-info element
            col_name = ""
            active_entity = None
            fhir_resource = "observation"
            category = ""
            
            # First look for the hidden tracker div that stores the current selected entity
            if entity_info and isinstance(entity_info, list):
                for item in entity_info:
                    if isinstance(item, dict) and item.get('type') == 'Div' and item.get('props', {}).get('id') == "current-selected-entity":
                        active_entity = item.get('props', {}).get('children')
                        print(f"Found active entity in tracker: {active_entity}")
                        break
            
            # If entity not found in tracker, extract other information anyway
            if entity_info:
                # Extract column name and FHIR resource
                for item in entity_info if isinstance(entity_info, list) else [entity_info]:
                    if isinstance(item, dict) and item.get('type') == 'H5':
                        title_text = item.get('props', {}).get('children', '')
                        if isinstance(title_text, str) and "Column:" in title_text:
                            col_name = title_text.split("Column:")[1].strip()
                            
                    # Extract FHIR resource
                    if isinstance(item, dict) and item.get('type') == 'H6':
                        resource_text = item.get('props', {}).get('children', '')
                        if isinstance(resource_text, str) and "FHIR Resource:" in resource_text:
                            fhir_resource = resource_text.split("FHIR Resource:")[1].strip()
                            
                    # Extract category
                    if isinstance(item, dict) and item.get('type') == 'H6':
                        category_text = item.get('props', {}).get('children', '')
                        if isinstance(category_text, str) and "Category:" in category_text:
                            category = category_text.split("Category:")[1].strip()
            
            # If we still don't have the active entity, try to find the selected button
            if not active_entity and entity_info and isinstance(entity_info, list):
                for item in entity_info:
                    if isinstance(item, dict) and item.get('props', {}).get('style', {}).get('display') == 'flex':
                        buttons = item.get('props', {}).get('children', [])
                        for button in buttons if isinstance(buttons, list) else [buttons]:
                            if isinstance(button, dict) and button.get('props', {}).get('color') == 'success' and not button.get('props', {}).get('outline', True):
                                button_id = button.get('props', {}).get('id', {})
                                if isinstance(button_id, dict) and 'entity' in button_id:
                                    active_entity = button_id.get('entity', '')
                                    print(f"Found active entity in button: {active_entity}")
                                    break
            
            # If we still don't have an active entity, show an error
            if not active_entity:
                return html.Div(
                    dbc.Alert("Please select an entity to map before looking up SNOMED codes", color="warning"),
                    className="mt-2"
                )
            
            # Initialize database connection to query SNOMED codes
            cur, encoder = initialize_emb_model()
            
            # Direct query by conceptid
            query = f"""
            SELECT 
                conceptid, 
                conceptid_name, 
                toplevelhierarchy_name, 
                fhir_resource
            FROM 
                {DB_CONFIG['snomed_ct_table']}
            WHERE 
                conceptid = %s
            """
            
            cur.execute(query, (code,))
            result = cur.fetchone()
            
            if result:
                # Create result card with the selected entity name prominently displayed
                card = dbc.Card([
                    dbc.CardHeader([
                        f"SNOMED Code for ",
                        html.Span(active_entity, className="fw-bold"),
                        f": {code}"
                    ]),
                    dbc.CardBody([
                        html.H5(f"{result['conceptid_name']}", className="card-title"),
                        html.P(f"Hierarchy: {result['toplevelhierarchy_name']}", className="card-text"),
                        html.P(f"FHIR Resource: {result['fhir_resource']}", className="card-text text-muted"),
                        
                        # Create a selection button with the active entity specified
                        dbc.Button(
                            f"Use This Code for '{active_entity}'",
                            id={
                                "type": "manual-select-button",
                                "snomed_id": code,
                                "snomed_name": sanitize_for_json(result['conceptid_name']),
                                "col": sanitize_for_json(col_name),
                                "fhir": sanitize_for_json(result['fhir_resource'] or fhir_resource),
                                "category": sanitize_for_json(category),
                                "entity": sanitize_for_json(active_entity)
                            },
                            color="success",
                            className="mt-2"
                        )
                    ])
                ], className="mb-3")
                
                return card
            else:
                # No results found
                return html.Div(dbc.Alert(f"No SNOMED code found with ID: {code}", color="warning"))
                    
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in SNOMED code lookup: {e}")
            print(error_details)
            
            return html.Div(dbc.Alert(f"Database error: {str(e)}", color="danger"))

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
        """Handle SNOMED code selection from manual lookup."""
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
            print(f"\n MANUAL SNOMED SELECTION DETECTED")
            print(f"Entity: {entity}")
            print(f"SNOMED ID: {snomed_id}")
            print(f"SNOMED Name: {snomed_name}")
            
            # Execute real database update
            db_update_result = process_manual_snomed_selection(
                snomed_id, 
                snomed_name, 
                entity, 
                execute_update=True
            )
            
            # Log the result
            if db_update_result["success"]:
                print(f"Manual SNOMED processing completed successfully")
            else:
                print(f"Manual SNOMED processing failed: {db_update_result}")
                
        except Exception as e:
            print(f"Error in manual SNOMED processing: {e}")
            traceback.print_exc()
        
        # Load current mappings
        mappings = {}
        if mappings_json:
            try:
                mappings = json.loads(mappings_json)
            except:
                pass
        
        # Load existing renamed columns
        try:
            renamed_cols = json.loads(existing_renamed_cols) if existing_renamed_cols else {}
        except:
            renamed_cols = {}
        
        # Add the new mapping
        if col_name not in mappings:
            mappings[col_name] = {}
        
        mappings[col_name][entity] = {
            "snomed_id": snomed_id,
            "snomed_name": snomed_name,
            "fhir_resource": fhir_resource,
            "category": category,
            "source": "Manual"
        }
        
        # Update the column mapping table
        updated_table_data = table_data.copy()
        for row in updated_table_data:
            if row["Original Column"] == col_name:
                # Collect all mappings for this column
                match_info = []
                for ent, match in mappings[col_name].items():
                    match_info.append(f"{ent}: {match['snomed_id']} - {match['snomed_name']}")
                row["SNOMED Match"] = "; ".join(match_info)
                break
        
        # Handle column renaming based on SNOMED mappings
        from routes.column_naming import generate_column_name
        new_name = generate_column_name(col_name, mappings.get(col_name, {}))
        if new_name != col_name:
            renamed_cols[col_name] = new_name
        
        # Close the modal after successful selection
        return json.dumps(mappings), updated_table_data, json.dumps(renamed_cols), False
