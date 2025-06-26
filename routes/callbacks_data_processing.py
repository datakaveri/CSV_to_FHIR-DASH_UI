"""
Data processing callbacks for the Dash app.
Handles initial data processing, column extraction, and entity mapping.
"""

import json
import pandas as pd
import traceback
import dash
from dash import Input, Output, State, dash_table, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from routes.helpers import (
    process_unified_api_response, 
    generate_column_names
)


def register_data_processing_callbacks(app):
    """Register data processing related callbacks."""
    
    @app.callback(
        [Output("extracted-entities", "data"),
        Output("adarv-entities", "data"),
        Output("renamed-columns", "data"),
        Output("column-stats", "children"),
        Output("processing-status", "children", allow_duplicate=True),
        Output("results-section", "style"),
        Output("column-mapping-table", "children"),
        Output("snomed-mappings", "data", allow_duplicate=True)],
        Input("stored-data", "data"),
        prevent_initial_call=True
    )
    def process_data(json_data):
        """Process uploaded data through the unified API and generate mappings."""
        if not json_data:
            return None, None, None, "", "No data to process", {"display": "none"}, None, None
        
        try:
            # Load the dataframe
            df = pd.read_json(json_data, orient='split')
            columns_list = df.columns.tolist()
            
            print(f"Processing {len(columns_list)} columns: {columns_list}")
            
            # Use the parallel unified API processing
            try:
                # Determine optimal number of workers based on number of columns
                max_workers = min(10, max(1, len(columns_list) // 2))
                processed_columns, exception_cols = process_unified_api_response(
                    columns_list, 
                    max_workers=max_workers
                )
                print(f"Parallel API processed: {len(processed_columns)} columns mapped, {len(exception_cols)} exceptions")
            except Exception as e:
                print(f"Error in parallel API processing: {e}")
                traceback.print_exc()
                processed_columns = {}
                exception_cols = columns_list
            
            # Generate column names using ADARV suggested names and entity-based logic
            try:
                column_renamed_mapping = generate_column_names(processed_columns, columns_list)
                print(f"Generated column names: {column_renamed_mapping}")
            except Exception as e:
                print(f"Error generating column names: {e}")
                traceback.print_exc()
                # Fallback column naming
                column_renamed_mapping = {}
                used_names = set()
                for i, col in enumerate(columns_list):
                    base_name = f"col_{i}"
                    counter = 1
                    abbr_col = base_name
                    while abbr_col in used_names:
                        abbr_col = f"{base_name}_{counter}"
                        counter += 1
                    column_renamed_mapping[col] = abbr_col
                    used_names.add(abbr_col)
            
            # Build category map and FHIR resource map from API results
            category_map = {}
            fhir_resource_map = {}
            
            for col_name, col_data in processed_columns.items():
                category_map[col_name] = col_data.get('category', '')
                fhir_resource_map[col_name] = col_data.get('primary_fhir_resource', 'observation')
            
            # For columns without API results, assign defaults
            for col_name in exception_cols:
                category_map[col_name] = ""
                fhir_resource_map[col_name] = "observation"
            
            # Format entities for display
            formatted_entities = {}
            for col_name, col_data in processed_columns.items():
                entities = [entity['entity_name'] for entity in col_data['entities']]
                formatted_entities[col_name] = {
                    'entities': entities,
                    'fhir_resource': col_data['primary_fhir_resource'],
                    'source': 'unified_api'
                }
            
            # Initialize mappings with all entities from API response
            mappings = {}
            for col_name, col_data in processed_columns.items():
                mappings[col_name] = {}
                
                for entity_data in col_data['entities']:
                    entity_name = entity_data['entity_name']
                    mappings[col_name][entity_name] = {
                        "snomed_id": entity_data['snomed_id'],
                        "snomed_name": entity_data['snomed_name'],
                        "snomed_full_name": entity_data['snomed_full_name'],
                        "fhir_resource": entity_data['fhir_resource'],
                        "category": entity_data.get('category', ''),
                        "source": "unified_api",
                        "renamed_column": column_renamed_mapping.get(col_name, col_name)
                    }
            
            # Create mapping table
            table_data = []
            for i, col_name in enumerate(columns_list):
                try:
                    # Get FHIR resource
                    fhir_resource = fhir_resource_map.get(col_name, 'observation')
                    
                    # Get category
                    category = category_map.get(col_name, "")
                    
                    # Get entities - prioritize API results
                    entities_source = "fallback"
                    entities = [col_name]  # Default fallback
                    
                    if col_name in processed_columns:
                        entities = [entity['entity_name'] for entity in processed_columns[col_name]['entities']]
                        entities_source = "unified_api (parallel)"
                    elif col_name in formatted_entities:
                        entities = formatted_entities[col_name]['entities']
                        entities_source = formatted_entities[col_name].get('source', 'unknown')
                    
                    # Get SNOMED matches
                    snomed_matches = []
                    if col_name in mappings:
                        for entity, match in mappings[col_name].items():
                            source_label = f" ({match.get('source', 'unknown')})"
                            snomed_matches.append(f"{entity}: {match['snomed_id']} - {match['snomed_name']}{source_label}")
                    
                    snomed_match_text = "; ".join(snomed_matches) if snomed_matches else "No matches found"
                    
                    # Create entities list for JSON storage
                    entities_list_json = json.dumps(entities)
                    
                    table_data.append({
                        "Original Column": col_name,
                        "FHIR Resource": fhir_resource,
                        "Category": category,
                        "Extracted Entities": f"{', '.join(entities)} ({entities_source})",
                        "SNOMED Match": snomed_match_text,
                        "Actions": "Edit Mappings",
                        "entities_list_json": entities_list_json
                    })
                except Exception as e:
                    print(f"Error creating table row for {col_name}: {e}")
                    table_data.append({
                        "Original Column": col_name,
                        "FHIR Resource": "observation",
                        "Category": "",
                        "Extracted Entities": "Error extracting entities",
                        "SNOMED Match": "Error",
                        "Actions": "Edit Mappings",
                        "entities_list_json": "[]"
                    })
            
            # Create data table for display
            mapping_table = dash_table.DataTable(
                id='column-mapping-datatable',
                columns=[
                    {"name": "Original Column", "id": "Original Column"},
                    {"name": "FHIR Resource", "id": "FHIR Resource"},
                    {"name": "Category", "id": "Category"},
                    {"name": "Extracted Entities", "id": "Extracted Entities"},
                    {"name": "SNOMED Match", "id": "SNOMED Match"},
                    {"name": "Actions", "id": "Actions"}
                ],
                data=table_data,
                style_table={
                    'overflowX': 'auto', 
                    'overflowY': 'auto', 
                    'maxHeight': '500px'
                },
                style_cell={
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'maxWidth': 0,
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Actions'},
                        'cursor': 'pointer',
                        'color': 'blue',
                        'textDecoration': 'underline'
                    },
                    {
                        'if': {'column_id': 'Category'},
                        'cursor': 'pointer',
                        'color': 'green',
                        'textDecoration': 'underline'
                    },
                    {
                        'if': {'column_id': 'SNOMED Match', 'filter_query': '{SNOMED Match} contains "No matches"'},
                        'color': 'orange'
                    },
                    {
                        'if': {'column_id': 'SNOMED Match', 'filter_query': '{SNOMED Match} contains "Error"'},
                        'color': 'red'
                    },
                    {
                        'if': {'column_id': 'SNOMED Match', 'filter_query': '{SNOMED Match} contains "(unified_api)"'},
                        'backgroundColor': '#e6f3ff',
                        'fontWeight': 'bold'
                    }
                ],
                tooltip_delay=0,
                tooltip_duration=None,
                cell_selectable=True,
                filter_action="native",
                sort_action="native",
            )
            
            # Stats display
            api_mapped_count = len(processed_columns)
            total_entities = sum(len(col_data['entities']) for col_data in processed_columns.values())
            processing_time_estimate = f"~{len(columns_list)/max_workers:.1f}s (parallel)"
            
            stats = html.Div([
                f"Total columns: {len(columns_list)}",
                html.Br(),
                f"API processed columns: {api_mapped_count}",
                html.Br(),
                f"Exception columns: {len(exception_cols)}",
                html.Br(),
                f"Total mapped entities: {total_entities}",
                html.Br(),
                f"Processing time: {processing_time_estimate}"
            ])
            
            return (
                json.dumps(formatted_entities),
                json.dumps(processed_columns),  # Store full API response data
                json.dumps(column_renamed_mapping),
                stats,
                f"Parallel processing complete - {api_mapped_count} columns processed via unified API using {max_workers} workers, {len(exception_cols)} exceptions",
                {"display": "block"},
                mapping_table,
                json.dumps(mappings)
            )
        
        except Exception as e:
            print(f"Error processing data: {e}")
            traceback.print_exc()
            return (
                None,
                None,
                None, 
                f"Error: {str(e)}", 
                f"Error: {str(e)}", 
                {"display": "none"},
                None,
                None
            )
