import os
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import json
import requests
import re
import traceback
from tqdm import tqdm
from urllib.parse import parse_qs  # Add this import
from concurrent.futures import ThreadPoolExecutor, as_completed  # Add this import
from demo_utils_3 import (
    assign_categories,
    initialize_emb_model, 
    load_templates,
    process_snmd_cds,
    mark_condition_resources,
    remove_punctuation,
    create_valueset,
    load_fhir_data
)
from manual_snomed_updater import process_manual_snomed_selection
from dash.exceptions import PreventUpdate
import time
import copy

# Add this utility function to your file (at the top with other utility functions)
# def sanitize_for_json(text):
#     """Sanitize text to be safely included in JSON strings."""
#     if not isinstance(text, str):
#         return str(text) if text is not None else ""
    
#     # Replace all JSON-problematic characters
#     return (text.replace('\\', '\\\\')
#                 .replace('"', '\\"')
#                 .replace('\n', '\\n')
#                 .replace('\r', '\\r')
#                 .replace('\t', '\\t')
#                 .replace('\b', '\\b')
#                 .replace('\f', '\\f'))

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
    import re
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


# Add a new helper function to automatically search and select the top match
def search_and_select_top_match(entity, col_name, fhir_resource, category=""):
    """
    Search for SNOMED codes for an entity and select the top result automatically
    Returns a dict with the mapping information or None if no match found
    """
    try:
        # Make API request to search for SNOMED codes
        payload = {
            "keywords": entity,
            "threshold": 0.5,
            "limit": 10
        }
        
        # Make request to the API
        response = requests.post(API_ENDPOINT, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            results = json.loads(response.text)
            
            # If we have results, select the top one
            if results and isinstance(results, list) and len(results) > 0:
                top_result = results[0]
                
                # Validate the result has all required fields
                if not all(k in top_result for k in ["conceptid", "conceptid_name", "toplevelhierarchy_name"]):
                    print(f"Invalid result format for entity {entity}")
                    return None
                
                # Extract best term from comma-separated concept_name
                concept_name = top_result["conceptid_name"]
                best_term = concept_name
                
                if ',' in concept_name:
                    try:
                        # Import sentence transformer for cosine similarity
                        from sentence_transformers import SentenceTransformer
                        import numpy as np
                        
                        # Split the concept name by commas
                        terms = [term.strip() for term in concept_name.split(',')]
                        
                        try:
                            # First try to load the model on CPU explicitly to avoid CUDA OOM
                            model = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
                            
                            # Encode the entity and all terms in batches
                            entity_embedding = model.encode([entity], show_progress_bar=False, batch_size=1)[0]
                            
                            # Process terms in batches to avoid memory issues
                            batch_size = 5  # Small batch size to avoid memory issues
                            similarities = []
                            
                            for i in range(0, len(terms), batch_size):
                                batch_terms = terms[i:i+batch_size]
                                batch_embeddings = model.encode(batch_terms, show_progress_bar=False, batch_size=1)
                                
                                # Calculate similarity for each term in the batch
                                for term_embedding in batch_embeddings:
                                    # Cosine similarity calculation
                                    similarity = np.dot(entity_embedding, term_embedding) / (
                                        np.linalg.norm(entity_embedding) * np.linalg.norm(term_embedding)
                                    )
                                    similarities.append(similarity)
                            
                            # Find term with highest similarity
                            best_idx = np.argmax(similarities)
                            best_term = terms[best_idx]
                            
                        except Exception as e:
                            print(f"Error in batched cosine similarity calculation: {e}")
                            # Fallback to simple text-based selection
                            # Sort terms by length (prefer shorter, more concise terms)
                            terms.sort(key=len)
                            best_term = terms[0]
                            
                    except Exception as e:
                        print(f"Error in cosine similarity calculation: {e}")
                        # Safely get the first term from the comma-separated list
                        try:
                            terms = [term.strip() for term in concept_name.split(',')]
                            best_term = terms[0] if terms else concept_name
                        except:
                            best_term = concept_name
                
                # Create and return the mapping
                return {
                    "snomed_id": top_result["conceptid"],
                    "snomed_name": best_term,
                    "snomed_full_name": concept_name,
                    "fhir_resource": fhir_resource,
                    "category": category,
                    # "score": top_result.get("rrf_score", 0)
                }
            
            print(f"No results found for entity {entity}")
            return None
        else:
            print(f"API error for entity {entity}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error searching SNOMED for entity {entity}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_file_from_path(file_path, filename):
    """Process a file from a full file path"""
    try:
        # Read the file based on extension
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return None, f"Unsupported file type: {filename}"
        
        # Extract base filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        return df, base_filename
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error processing file: {str(e)}"

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "CSV to FHIR Converter"

# Define API endpoint
API_ENDPOINT = "http://localhost:8000/search"
# API_ENDPOINT = "https://entitymapper.adarv.in/search"


# Try to load FHIR templates
try:
    templates = load_templates()
    print("FHIR templates loaded successfully")
except Exception as e:
    print(f"Error loading templates: {e}")
    templates = None

# Layout
app.layout = dbc.Container([
    # Add this Location component at the top of your layout
    dcc.Location(id='url', refresh=False),
    html.Div(id='api-data-container', style={'display': 'none'}),
    
    html.H2("CSV to FHIR Converter", className="mt-4 mb-4"),
    
    # Replace the current dataset information card with a processing indicator
    dbc.Card([
        dbc.CardHeader("Dataset Information"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    # Processing indicator
                    dbc.Spinner(
                        html.Div(id="dataset-info-display", children="Initializing..."),
                        color="primary",
                        type="grow",
                        spinner_style={"width": "3rem", "height": "3rem"}
                    ),
                    # Make input fields visible for direct upload
                    html.Div([
                        html.H5("Direct File Upload", className="mt-3"),
                        # Username field
                        dbc.Label("Username:"),
                        dbc.Input(id="username", placeholder="Enter your username", type="text", className="mb-3"),
                        
                        # Dataset type dropdown
                        dbc.Label("Dataset Type:"),
                        dcc.Dropdown(
                            id="dataset-type",
                            options=[
                                {"label": "OBI", "value": "OBI"},
                                {"label": "Surveillance", "value": "Surveillance"},
                                {"label": "Research", "value": "Research"}
                            ],
                            placeholder="Select dataset type",
                            className="mb-3"
                        ),
                        
                        # File upload component
                        dbc.Label("Upload Dataset:"),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                    ]),
                    html.Div(id='upload-status')
                ], width=12)
            ]),
        ])
    ], className="mb-4"),
    
    # Status indicators
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Processing Status", className="card-title"),
                    html.Div(id="processing-status", children="No file uploaded")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Column Stats", className="card-title"),
                    html.Div(id="column-stats")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Results section
    html.Div([
        dbc.Tabs([
            dbc.Tab(label="Column Mapping", tab_id="tab-mapping", children=[
                html.Div(id="column-mapping-table", className="mt-3")
            ]),
            dbc.Tab(label="Entity Search", tab_id="tab-entity-search", children=[
                html.Div(id="entity-search-content", className="mt-3")
            ]),
            dbc.Tab(label="Final Mapping", tab_id="tab-final-mapping", children=[
                html.Div(id="final-mapping-content", className="mt-3")
            ]),
            dbc.Tab(label="FHIR Processing", tab_id="tab-fhir-processing", children=[
                html.Div(id="fhir-processing-content", className="mt-3")
            ]),
        ], id="tabs", active_tab="tab-mapping")
    ], id="results-section", style={"display": "none"}),
    
    # Store components for data
    dcc.Store(id="stored-data"),
    dcc.Store(id="extracted-entities"),
    dcc.Store(id="adarv-entities"),
    dcc.Store(id="renamed-columns"),
    dcc.Store(id="snomed-mappings"),
    dcc.Store(id="original-filename"),  # Add this store for the original filename
    dcc.Store(id="current-category-column-index"),  # Add this store for the current category column index
    dcc.Store(id="resource-ids"),  # Store resource and group IDs
    
    # Modal for entity search
    dbc.Modal([
        dbc.ModalHeader("Search SNOMED CT Codes"),
        dbc.ModalBody([
            html.Div(id="modal-entity-info"),
            
            # Add a new section for manual SNOMED code entry
            html.Div([
                html.H6("Enter SNOMED Code Directly:", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(
                                id="manual-snomed-code-input", 
                                placeholder="Enter SNOMED code (e.g., 73211009)",
                                type="text"
                            ),
                            dbc.InputGroupText(""),
                            dbc.Button(
                                "Look Up", 
                                id="lookup-snomed-code-button", 
                                color="secondary",
                                className="ms-2"
                            )
                        ]),
                    ], width=12),
                ]),
                html.Div(id="manual-snomed-results", className="mt-3"),
            ], className="mb-4 p-3 border rounded"),
            
            html.H6("Search Results:", className="mt-3"),
            html.Div(id="modal-search-results", style={"maxHeight": "400px", "overflow": "auto"}),
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="modal-close", className="ml-auto"),
        ]),
    ], id="entity-search-modal", size="lg"),

    # Add this to your app.layout, after the entity-search-modal
    dbc.Modal([
        dbc.ModalHeader("Select Category"),
        dbc.ModalBody([
            html.Div(id="category-modal-column-name"),
            dcc.Dropdown(
                id="category-dropdown-modal",
                options=[
                    {'label': 'Date/Time', 'value': 'Date/Time'},
                    {'label': 'Exposure', 'value': 'Exposure'},
                    {'label': 'ID', 'value': 'ID'},
                    {'label': 'Outcome', 'value': 'Outcome'},
                    {'label': 'Social Demographic', 'value': 'Social Demographic'},
                    {'label': 'Status', 'value': 'Status'},
                    {'label': 'Symptom', 'value': 'Symptom'},
                    {'label': '', 'value': ''}  # Empty option
                ],
                clearable=False,
                style={"width": "100%"}
            ),
        ]),
        dbc.ModalFooter([
            dbc.Button("Save", id="save-category-button", className="ml-auto", color="primary"),
            dbc.Button("Cancel", id="cancel-category-button", className="ml-2")
        ]),
    ], id="category-selection-modal", size="md"),

    # Add this near the end of your app.layout, before the closing brackets
    dcc.Loading(
        id="loading-fhir",
        type="circle",
        children=html.Div(id="fhir-processing-output")
    ),
], fluid=True)

# Update the URL parameter processing callback to properly handle encoded JSON
@app.callback(
    [Output('api-data-container', 'children'),
     Output('username', 'value'),
     Output('dataset-type', 'value'),
     Output('processing-status', 'children', allow_duplicate=True),
     Output('dataset-info-display', 'children')],  # Add this output
    [Input('url', 'search')],
    prevent_initial_call=True
)
def process_url_parameters(search):
    if not search:
        # When no URL parameters, show empty dataset info with upload instructions
        return None, None, None, "Waiting for dataset information or direct upload...", html.Div([
            html.H5("No dataset loaded via URL"),
            html.P("Use the direct file upload option below or access with proper URL parameters.")
        ])
    try:
        print(f"Raw URL search string: {search}")
        
        # Parse query parameters
        parsed = parse_qs(search.lstrip('?'))
        
        # Convert to proper dictionary format
        params = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
        
        # Parse JSON if data was passed as a single parameter
        if 'data' in params:
            try:
                # The data might be URL-encoded JSON
                import urllib.parse
                
                # First try to decode if it's URL encoded
                decoded_data = urllib.parse.unquote(params['data'])
                print(f"Decoded data parameter: {decoded_data[:100]}...")
                
                # Then parse as JSON
                params = json.loads(decoded_data)
                print(f"Parsed JSON data: {params}")
            except Exception as e:
                print(f"Error decoding JSON data: {e}")
                # Try direct JSON parsing as fallback
                try:
                    params = json.loads(params['data'])
                except Exception as e2:
                    print(f"Error parsing data as JSON: {e2}")
        
        # Extract relevant fields
        username = params.get('userName', '')
        dataset_type = params.get('typeOfDataset', '')
        path = params.get('path', '')
        filename = params.get('fileName', '')
        
        # Additional data for later access if needed
        disease_name = params.get('diseaseName', '')
        res_grp_id = params.get('resGrpId', '')
        rid = params.get('rid', '')
        
        print(f"Extracted parameters: username={username}, dataset_type={dataset_type}, filename={filename}, path={path}")
        
        # Create detailed dataset info display
        dataset_info = html.Div([
            html.H5(f"Processing: {filename}", className="mb-2"),
            html.Div([
                html.Strong("Disease: "), 
                html.Span(disease_name)
            ], className="mb-1"),
            html.Div([
                html.Strong("Type: "), 
                html.Span(dataset_type)
            ], className="mb-1"),
            html.Div([
                html.Strong("Resource ID: "), 
                html.Span(rid)
            ], className="mb-1 text-muted"),
            html.Div([
                html.Strong("Resource Group ID: "), 
                html.Span(res_grp_id)
            ], className="mb-1 text-muted")
        ])
        
        # Store the complete data for use in other callbacks
        return json.dumps(params), username, dataset_type, f"Processing dataset: {filename}", dataset_info
    
    except Exception as e:
        print(f"Error processing URL parameters: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, f"Error processing parameters: {str(e)}", "Error loading dataset information"
    
# Update the file loading callback to properly handle Windows file paths
@app.callback(
    [Output("stored-data", "data", allow_duplicate=True),
     Output("upload-status", "children", allow_duplicate=True),
     Output("processing-status", "children", allow_duplicate=True),
     Output("original-filename", "data", allow_duplicate=True),
     Output("resource-ids", "data")],  # Add this output
    [Input("api-data-container", "children")],
    prevent_initial_call=True  # Already has this, good!
)
def load_file_from_parameters(api_data_json):
    if not api_data_json:
        raise PreventUpdate
    
    try:
        # Parse the API data
        api_data = json.loads(api_data_json)
        
        # Extract file information
        path = api_data.get('path', '')
        filename = api_data.get('fileName', '')
        
        # Clean up the filename by removing any leading backslash
        if filename.startswith('\\'):
            filename = filename[1:]
        
        # Extract resource IDs
        res_grp_id = api_data.get('resGrpId', '')
        rid = api_data.get('rid', '')
        
        # Store resource IDs in a structure for later use
        resource_ids = {
            "resGrpId": res_grp_id,
            "rid": rid,
            "userName": api_data.get('userName', ''),
            "typeOfDataset": api_data.get('typeOfDataset', ''),
            "diseaseName": api_data.get('diseaseName', '')
        }
        
        if not path or not filename:
            return None, "Missing file information", "Error: Missing file information", None, json.dumps(resource_ids)
        
        # Fix Windows file path formatting
        # Replace any double backslashes with single ones
        path = path.replace('\\\\', '\\')
        
        # Check if the path ends with a backslash or forward slash
        if not path.endswith('\\') and not path.endswith('/'):
            path += '\\'  # Add a trailing backslash for Windows paths
        
        # Construct the full file path
        full_path = os.path.join(path, filename)
        
        print(f"Attempting to read file: {full_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(full_path):
                print(f"File not found: {full_path}")
                # Try alternative path format (with forward slashes)
                alt_path = path.replace('\\', '/') + filename
                print(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    full_path = alt_path
                else:
                    return None, f"File not found: {full_path}", f"Error: File not found", None, json.dumps(resource_ids)
            
            # Process the file using the new utility function
            df, base_filename = process_file_from_path(full_path, filename)
            if df is None:
                return None, base_filename, "Error: Unsupported file type", None, json.dumps(resource_ids)
            
            print(f"Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")
            
            return df.to_json(orient='split'), f"File loaded: {filename}", "Processing data...", base_filename, json.dumps(resource_ids)
            
        except Exception as e:
            print(f"Error reading file: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error reading file: {str(e)}", f"Error: {str(e)}", None, json.dumps(resource_ids)
        
    except Exception as e:
        print(f"Error processing API data: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error processing API data: {str(e)}", f"Error: {str(e)}", None, "{}"
    
# Update your existing process_upload callback with allow_duplicate attribute
@app.callback(
    [Output("stored-data", "data", allow_duplicate=True),
     Output("upload-status", "children", allow_duplicate=True),
     Output("processing-status", "children", allow_duplicate=True),
     Output("original-filename", "data", allow_duplicate=True)],  
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True  # Already has this, good!
)
def process_upload(contents, filename):
    if contents is None:
        return None, "", "No file uploaded", None
    
    try:
        # Decode the uploaded content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Process the file based on its type
        try:
            if 'csv' in filename.lower():
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename.lower():
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return None, html.Div(['Unsupported file type']), "Error: Unsupported file type", None
        except Exception as e:
            return None, html.Div(['Error processing this file: ' + str(e)]), f"Error: {str(e)}", None
        
        # Extract the base filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        return df.to_json(orient='split'), html.Div(['File uploaded successfully: ', filename]), "Processing data...", base_filename
    
    except Exception as e:
        return None, html.Div(['Error: ', str(e)]), f"Error: {str(e)}", None

# Utility function to split entities from comma-separated text
def split_entities(entity_text):
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

# Callback to open the category modal when a cell in the Category column is clicked
@app.callback(
    [Output("category-selection-modal", "is_open"),
     Output("category-modal-column-name", "children"),
     Output("category-dropdown-modal", "value")],
    [Input("column-mapping-datatable", "active_cell")],
    [State("column-mapping-datatable", "data"),
     State("category-selection-modal", "is_open")]
)
def toggle_category_modal(active_cell, table_data, is_open):
    if active_cell and active_cell["column_id"] == "Category":
        # Get the row
        row_idx = active_cell["row"]
        row_data = table_data[row_idx]
        
        # Get column name and current category
        col_name = row_data["Original Column"]
        current_category = row_data.get("Category", "")
        
        return True, html.H5(f"Select category for column: {col_name}"), current_category
    
    return False, "", ""

# Callback to update the current column index when the modal is opened
@app.callback(
    Output("current-category-column-index", "data"),
    [Input("category-selection-modal", "is_open"),
     Input("column-mapping-datatable", "active_cell")],
)
def store_current_column_index(is_open, active_cell):
    if is_open and active_cell and active_cell["column_id"] == "Category":
        return active_cell["row"]
    return None

# Callback to close the modal without saving
@app.callback(
    Output("category-selection-modal", "is_open", allow_duplicate=True),
    [Input("cancel-category-button", "n_clicks")],
    prevent_initial_call=True  # Already has this, good!
)
def close_category_modal(n_clicks):
    if n_clicks:
        return False
    return dash.no_update

# Callback to save the selected category
@app.callback(
    [Output("category-selection-modal", "is_open", allow_duplicate=True),
     Output("column-mapping-datatable", "data", allow_duplicate=True),
     Output("snomed-mappings", "data", allow_duplicate=True)],
    [Input("save-category-button", "n_clicks")],
    [State("current-category-column-index", "data"),
     State("category-dropdown-modal", "value"),
     State("column-mapping-datatable", "data"),
     State("snomed-mappings", "data")],
    prevent_initial_call=True  # Already has this, good!
)
def save_category(n_clicks, row_idx, category_value, table_data, mappings_json):
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


# Update the lookup_snomed_code callback to correctly extract the selected entity
@app.callback(
    Output("manual-snomed-results", "children"),
    [Input("lookup-snomed-code-button", "n_clicks")],
    [State("manual-snomed-code-input", "value"),
     State("modal-entity-info", "children"),
     State("snomed-mappings", "data")],
    prevent_initial_call=True
)
def lookup_snomed_code(n_clicks, code, entity_info, mappings_json):
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
        query = """
        SELECT 
            conceptid, 
            conceptid_name, 
            toplevelhierarchy_name, 
            fhir_resource
        FROM 
            snomed_ct_codes_tst
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
                    html.Span(active_entity, className="fw-bold"),  # Use active_entity here
                    f": {code}"
                ]),
                dbc.CardBody([
                    html.H5(f"{result['conceptid_name']}", className="card-title"),
                    html.P(f"Hierarchy: {result['toplevelhierarchy_name']}", className="card-text"),
                    html.P(f"FHIR Resource: {result['fhir_resource']}", className="card-text text-muted"),
                    
                    # Create a selection button with the active entity specified
                    dbc.Button(
                        f"Use This Code for '{active_entity}'",  # Use active_entity here
                        id={
                            "type": "manual-select-button",
                            "snomed_id": code,
                            "snomed_name": sanitize_for_json(result['conceptid_name']),
                            "col": sanitize_for_json(col_name),
                            "fhir": sanitize_for_json(result['fhir_resource'] or fhir_resource),
                            "category": sanitize_for_json(category),
                            "entity": sanitize_for_json(active_entity)  # Pass the active entity in the button ID
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


# 2. MODIFY toggle_entity_button to include modal opening functionality
# ...existing code...

# 2. MODIFY toggle_entity_button to include modal opening functionality
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
    # Check which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    try:
        # Extract the button ID that was clicked with ultra-robust parsing
        button_id_str = ctx.triggered[0]["prop_id"].split(".")[0]
        print(f"Raw button ID string: {repr(button_id_str)}")  # Debug print
        
        button_id = None
        
        # Strategy 1: Try to clean the string first before JSON parsing
        try:
            # Remove any potential problematic characters before JSON parsing
            import re
            
            # Replace any non-printable characters and normalize spaces
            cleaned_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', button_id_str)
            cleaned_str = re.sub(r'[\u00a0\u2000-\u200f\u2028-\u202f\u205f-\u206f]', ' ', cleaned_str)
            cleaned_str = re.sub(r'\s+', ' ', cleaned_str)
            
            print(f"Cleaned button ID string: {repr(cleaned_str)}")
            
            button_id = json.loads(cleaned_str)
            print(f"Successfully parsed JSON: {button_id}")
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed even after cleaning: {e}")
            print(f"Failed at position: {e.pos}")
            
            # Show the problematic part of the string
            if hasattr(e, 'pos') and e.pos < len(button_id_str):
                problem_area = button_id_str[max(0, e.pos-10):e.pos+10]
                print(f"Problem area: {repr(problem_area)}")
            
            # Strategy 2: Manual regex extraction as ultra-robust fallback
            print("Using regex fallback extraction")
            
            # Extract components using regex with more flexible patterns
            type_match = re.search(r'"type"\s*:\s*"([^"]*)"', button_id_str)
            index_match = re.search(r'"index"\s*:\s*(\d+)', button_id_str)
            col_match = re.search(r'"col"\s*:\s*"([^"]*)"', button_id_str)
            entity_match = re.search(r'"entity"\s*:\s*"([^"]*)"', button_id_str)
            fhir_match = re.search(r'"fhir"\s*:\s*"([^"]*)"', button_id_str)
            
            # For entity, try multiple patterns to handle edge cases
            if not entity_match:
                # Try with single quotes
                entity_match = re.search(r'"entity"\s*:\s*\'([^\']*)\'', button_id_str)
            
            if not entity_match:
                # Try extracting everything between entity and the next field
                entity_match = re.search(r'"entity"\s*:\s*"([^"]*)', button_id_str)
            
            button_id = {
                "type": type_match.group(1) if type_match else "entity-button",
                "index": int(index_match.group(1)) if index_match else 0,
                "col": col_match.group(1) if col_match else "",
                "entity": entity_match.group(1) if entity_match else "",
                "fhir": fhir_match.group(1) if fhir_match else "observation"
            }
            
            print(f"Regex extracted: {button_id}")
        
        if not button_id:
            print("All parsing strategies failed")
            raise PreventUpdate
        
        # IMMEDIATELY sanitize all extracted values
        selected_entity = sanitize_for_json(button_id.get("entity", ""))
        col_name = sanitize_for_json(button_id.get("col", ""))
        fhir_resource = sanitize_for_json(button_id.get("fhir", ""))
        selected_index = button_id.get("index", 0)
        
        print(f"Sanitized values - entity: '{selected_entity}', col: '{col_name}', fhir: '{fhir_resource}'")
        
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
        
        # Extract the existing h5, h6 elements and entity buttons
        h5_element = None
        h6_elements = []
        entity_buttons_container = None
        manual_lookup_instructions = None
        entity_select_instruction = None
        
        for item in current_info if isinstance(current_info, list) else [current_info]:
            if isinstance(item, dict):
                item_type = item.get('type')
                if item_type == 'H5':
                    h5_element = item
                elif item_type == 'H6':
                    h6_elements.append(item)
                elif item_type == 'P' and 'Select an entity' in str(item.get('props', {}).get('children', '')):
                    entity_select_instruction = item
                elif item_type == 'P' and 'Use the search results' in str(item.get('props', {}).get('children', '')):
                    manual_lookup_instructions = item
                elif item.get('props', {}).get('style', {}).get('display') == 'flex':
                    entity_buttons_container = item
                elif item_type == 'Div' and isinstance(item.get('props', {}).get('children', []), list):
                    children = item.get('props', {}).get('children', [])
                    for child in children:
                        if isinstance(child, dict) and child.get('type') == 'P' and 'Use the search results' in str(child.get('props', {}).get('children', '')):
                            manual_lookup_instructions = item
        
        # If we couldn't find the buttons container, still open the modal but don't update other components
        if not entity_buttons_container:
            return dash.no_update, dash.no_update, True, None
        
        # Extract all buttons and update their properties
        buttons = entity_buttons_container.get('props', {}).get('children', [])
        updated_buttons = []
        
        # For each button, update its properties based on selection and mapping status
        for button in buttons if isinstance(buttons, list) else [buttons]:
            if not isinstance(button, dict) or 'props' not in button or 'id' not in button.get('props', {}):
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
            
            # Update button text if needed - SANITIZE HERE TOO
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
            
            # SANITIZE the button ID properties before creating the updated button
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
        
        # Add a hidden field to track the current selected entity
        current_entity_tracker = html.Div(
            id="current-selected-entity",
            children=selected_entity,
            style={"display": "none"}
        )
        
        # Compile the updated entity_info
        updated_info = []
        if h5_element:
            updated_info.append(h5_element)
        updated_info.extend(h6_elements)
        if entity_select_instruction:
            updated_info.append(entity_select_instruction)
        updated_info.append(updated_container)
        updated_info.append(current_entity_tracker)
        if manual_lookup_instructions:
            updated_info.append(manual_lookup_instructions)
        
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
        
        # NEW CODE: Check if the selected entity is already mapped by ADARV
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
            # Perform a new SNOMED search for the selected entity (existing code)
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
                            
                            # SANITIZE all result data
                            conceptid = sanitize_for_json(str(result.get("conceptid", "")))
                            conceptid_name = sanitize_for_json(result.get("conceptid_name", ""))
                            toplevelhierarchy_name = sanitize_for_json(result.get("toplevelhierarchy_name", "Unknown"))
                            rrf_score = result.get("rrf_score", 0)
                            
                            # Check if this is the current mapping
                            is_selected_result = False
                            if existing_mapping and existing_mapping.get("snomed_id") == result.get("conceptid"):
                                is_selected_result = True
                            
                            # Create the selection button with SANITIZED data
                            select_button = dbc.Button(
                                "Selected" if is_selected_result else "Select",
                                id={
                                    "type": "select-snomed-button",
                                    "index": i,
                                    "col": col_name,  # Already sanitized
                                    "entity": selected_entity,  # Already sanitized
                                    "fhir": fhir_resource,  # Already sanitized
                                    "snomed_id": conceptid,  # Sanitized
                                    "snomed_name": conceptid_name,  # Sanitized
                                    "category": category  # Already sanitized
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
        return dash.no_update, dash.no_update, True, None

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
     State("renamed-columns", "data"),  # Add this state to preserve existing names
     State("adarv-entities", "data")],  # Add this state to check ADARV mappings
    prevent_initial_call=True
)
def select_manual_snomed_code(n_clicks_list, mappings_json, table_data, json_data, existing_renamed_cols, adarv_entities_json):
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
    entity = button_id["entity"]  # Get the entity directly from the button

    # Process manual SNOMED selection for database update
    try:
        print(f"\n MANUAL SNOMED SELECTION DETECTED")
        print(f"Entity: {entity}")
        print(f"SNOMED ID: {snomed_id}")
        print(f"SNOMED Name: {snomed_name}")
        
        # EXECUTE REAL DATABASE UPDATE
        db_update_result = process_manual_snomed_selection(
            snomed_id, 
            snomed_name, 
            entity, 
            execute_update=True  # THIS ACTUALLY UPDATES THE DATABASE
        )
        
        # Log the result
        if db_update_result["success"]:
            print(f"Manual SNOMED processing completed successfully")
            
            if db_update_result["database_updated"]:
                print(f"Database successfully updated for SNOMED ID: {snomed_id}")
                db_result = db_update_result["update_info"]["database_result"]
                print(f"Database message: {db_result['message']}")
                print(f"Rows affected: {db_result.get('rows_affected', 'N/A')}")
                
                if db_result.get('updated_record'):
                    updated_record = db_result['updated_record']
                    print(f"New concept name: {updated_record['conceptid_name']}")
                    
            elif db_update_result["update_info"]["action"] == "no_update_needed":
                print(f"ℹNo database update needed - manual tag already exists")
            else:
                print(f"Database update was processed but not executed")
                
        else:
            error_msg = db_update_result["update_info"].get('error', 'Unknown error')
            print(f"Error processing manual SNOMED selection: {error_msg}")
            
    except Exception as e:
        print(f"Error in manual SNOMED processing branch: {e}")
        import traceback
        traceback.print_exc()
  
    # Use the snomed_name directly from the button - no need for select_best_term
    best_term = snomed_name  # Simply use the snomed_name as-is
    
    # Update mappings
    if not mappings_json:
        mappings = {}
    else:
        try:
            mappings = json.loads(mappings_json)
        except Exception as e:
            print(f"Error parsing mappings JSON: {e}")
            mappings = {}
    
    # Initialize column mapping if it doesn't exist
    if col_name not in mappings:
        mappings[col_name] = {}
    
    # Add or update the mapping for the selected entity
    mappings[col_name][entity] = {
        "snomed_id": snomed_id,
        "snomed_name": best_term,  # Use the snomed_name directly
        "snomed_full_name": snomed_name,  # Keep the full name
        "fhir_resource": fhir_resource,
        "category": category
    }
    
    # Update the table data to show the new mapping
    updated_table_data = table_data.copy()
    for row in updated_table_data:
        if row["Original Column"] == col_name:
            # Collect all mappings for this column
            match_info = []
            for ent, match in mappings[col_name].items():
                source_label = " (unified_api)" if match.get("source") == "unified_api" else " (manual)"
                match_info.append(f"{ent}: {match['snomed_id']} - {match['snomed_name']}{source_label}")
            row["SNOMED Match"] = "; ".join(match_info)
            break
    
    # Generate updated column names using the same logic as the other callbacks
    try:
        df = pd.read_json(json_data, orient='split')
        columns_list = df.columns.tolist()
        
        # Create a structure similar to what generate_column_names expects
        processed_columns = {}
        for col, entities_map in mappings.items():
            entities = []
            for entity_name, mapping in entities_map.items():
                entities.append({
                    'entity_name': entity_name,
                    'snomed_name': mapping['snomed_name'],
                    'fhir_resource': mapping['fhir_resource']
                })
            
            if entities:
                processed_columns[col] = {
                    'entities': entities
                }
        
        # Generate updated column names
        column_renamed_mapping = generate_column_names(processed_columns, columns_list)
        
        # Update the mappings with the new renamed column info
        for col, entities_map in mappings.items():
            renamed_col = column_renamed_mapping.get(col, col)
            for entity_name in entities_map:
                mappings[col][entity_name]["renamed_column"] = renamed_col
        
    except Exception as e:
        print(f"Error regenerating column names: {e}")
        # Fallback to existing names
        if existing_renamed_cols:
            try:
                column_renamed_mapping = json.loads(existing_renamed_cols)
            except:
                column_renamed_mapping = {col: col for col in mappings.keys()}
        else:
            column_renamed_mapping = {col: col for col in mappings.keys()}

    # Store the abbreviated name in each entity's mapping for reference
    for col, entities_map in mappings.items():
        abbr_col = column_renamed_mapping.get(col, col)
        for entity_name in entities_map:
            mappings[col][entity_name]["renamed_column"] = abbr_col
    
    # Close the modal - user has selected a code
    return json.dumps(mappings), updated_table_data, json.dumps(column_renamed_mapping), False

# Add this callback to handle deletion of mappings
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
        
        # Remove this entity from the mappings
        del mappings[col_name][entity]
        
        # If no more entities for this column, remove the column entirely
        if not mappings[col_name]:
            del mappings[col_name]
        
        # Now we need to regenerate the renamed columns based on the updated mappings
        renamed_cols = {}
        used_names = set()
        
        # First pass: generate names for columns with mappings
        for col, entities_map in mappings.items():
            if not entities_map:
                continue
                
            # Extract FHIR resource from the first entity
            first_entity = next(iter(entities_map.values()))
            fhir_resource = first_entity.get("fhir_resource", "observation")
            
            # Extract prefix from FHIR resource
            if fhir_resource.lower().startswith('condition'):
                prefix = 'con'
            elif fhir_resource.lower().startswith('patient.gender'):
                prefix = 'pat'
            elif fhir_resource.lower().startswith('patient.age'):
                prefix = 'pat'
            elif fhir_resource.lower().startswith('patient'):
                prefix = 'pat'
            else:
                prefix = fhir_resource[:3].lower()
            
            # Collect SNOMED names for each entity (this is the part that changes when deleting)
            snomed_prefixes = []
            for ent, mapping in entities_map.items():
                snomed_name = mapping.get("snomed_name", "").strip()
                if snomed_name:
                    snomed_prefixes.append(snomed_name[:3].lower())
            
            # Generate the abbreviated column name with remaining entities
            if snomed_prefixes:
                abbr_col = f"{prefix}_{'_'.join(snomed_prefixes)}"
            else:
                # Fallback if no SNOMED names available
                abbr_col = f"{prefix}_{col[:3].lower()}"
            
            # Handle duplicates
            base_abbr = abbr_col
            counter = 1
            while abbr_col in used_names:
                abbr_col = f"{base_abbr}{counter}"
                counter += 1
            
            renamed_cols[col] = abbr_col
            used_names.add(abbr_col)
        
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
        import traceback
        traceback.print_exc()
        raise PreventUpdate

# Update the process_data callback to use the unified API
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
    
# Update the select_snomed_code callback to remove select_best_term usage
@app.callback(
    [Output("snomed-mappings", "data", allow_duplicate=True),
     Output("column-mapping-datatable", "data", allow_duplicate=True),
     Output("renamed-columns", "data", allow_duplicate=True)],  
    [Input({"type": "select-snomed-button", "index": ALL, "col": ALL, "entity": ALL, 
           "fhir": ALL, "snomed_id": ALL, "snomed_name": ALL, "category": ALL}, "n_clicks")],
    [State("snomed-mappings", "data"),
     State("column-mapping-datatable", "data"),
     State("stored-data", "data"),
     State("renamed-columns", "data")],
    prevent_initial_call=True
)
def select_snomed_code(n_clicks_list, mappings_json, table_data, json_data, existing_renamed_cols):
    # Check which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Extract the button ID that was clicked
    button_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
    
    # Get mapping information
    col_name = button_id["col"]
    entity = button_id["entity"]
    fhir_resource = button_id["fhir"]
    snomed_id = button_id["snomed_id"]
    snomed_name = button_id["snomed_name"]
    category = button_id["category"]
    
    # Use the snomed_name directly from API - no need for select_best_term
    
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
    
    # Add or update the mapping for this entity
    mappings[col_name][entity] = {
        "snomed_id": snomed_id,
        "snomed_name": snomed_name,  # Use directly from API
        "snomed_full_name": snomed_name,  # Same as snomed_name
        "fhir_resource": fhir_resource,
        "category": category,
        "source": "unified_api"
    }
    
    # Update the table data to show the new mapping
    updated_table_data = table_data.copy()
    for row in updated_table_data:
        if row["Original Column"] == col_name:
            # Collect all mappings for this column
            match_info = []
            for ent, match in mappings[col_name].items():
                source_label = f" ({match.get('source', 'unified_api')})"
                match_info.append(f"{ent}: {match['snomed_id']} - {match['snomed_name']}{source_label}")
            row["SNOMED Match"] = "; ".join(match_info)
            break
    
    # Regenerate column names when mappings change
    try:
        df = pd.read_json(json_data, orient='split')
        columns_list = df.columns.tolist()
        
        # Create a structure similar to what generate_column_names expects
        processed_columns = {}
        for col, entities_map in mappings.items():
            entities = []
            for entity_name, mapping in entities_map.items():
                entities.append({
                    'entity_name': entity_name,
                    'snomed_name': mapping['snomed_name'],
                    'fhir_resource': mapping['fhir_resource']
                })
            
            if entities:
                processed_columns[col] = {
                    'entities': entities
                }
        
        # Generate updated column names
        column_renamed_mapping = generate_column_names(processed_columns, columns_list)
        
        # Update the mappings with the new renamed column info
        for col, entities_map in mappings.items():
            renamed_col = column_renamed_mapping.get(col, col)
            for entity_name in entities_map:
                mappings[col][entity_name]["renamed_column"] = renamed_col
        
    except Exception as e:
        print(f"Error regenerating column names: {e}")
        # Fallback to existing names
        if existing_renamed_cols:
            try:
                column_renamed_mapping = json.loads(existing_renamed_cols)
            except:
                column_renamed_mapping = {col: col for col in mappings.keys()}
        else:
            column_renamed_mapping = {col: col for col in mappings.keys()}
    
    return json.dumps(mappings), updated_table_data, json.dumps(column_renamed_mapping)
    
# def process_unified_api_response(columns_list):
#     """
#     Process columns through the unified API that handles both entity extraction and SNOMED mapping.
#     Returns structured data for display and mapping.
#     """
#     import requests
#     import json
    
#     processed_columns = {}
#     exception_columns = []
    
#     for col in tqdm(columns_list, desc="Processing columns via unified API"):
#         try:
#             # Make API request for this column
#             payload = {
#                 "keywords": col,
#                 "threshold": 0.5,
#                 "limit": 1  # Get more results to have options
#             }
            
#             # Make request to the unified API
#             response = requests.post(API_ENDPOINT, json=payload)
            
#             if response.status_code == 200:
#                 results = response.json()
                
#                 if results and isinstance(results, list) and len(results) > 0:
#                     # Process all entities found for this column
#                     column_entities = []
                    
#                     for result in results:
#                         # Validate the result has all required fields
#                         if not all(k in result for k in ["conceptid", "conceptid_name", "toplevelhierarchy_name"]):
#                             continue
                        
#                         # Use the concept name directly from API
#                         concept_name = result["conceptid_name"]
                        
#                         # Determine FHIR resource type
#                         fhir_resource = result.get("fhir_resource", "observation")
                        
#                         # Handle special cases for FHIR resource mapping
#                         if 'age' in concept_name.lower() and 'age' in col.lower():
#                             fhir_resource = 'Patient.Age'
#                         elif 'gender' in concept_name.lower():
#                             fhir_resource = 'Patient.Gender'
                        
#                         # Create entity mapping
#                         entity_mapping = {
#                             'entity_name': concept_name,
#                             'original_query': col,
#                             'snomed_id': result["conceptid"],
#                             'snomed_name': concept_name,
#                             'snomed_full_name': concept_name,
#                             'toplevelhierarchy_name': result.get("toplevelhierarchy_name", ""),
#                             'fhir_resource': fhir_resource,
#                             'category': result.get("category", ""),
#                             'score': result.get("match_score", 0),
#                             'source': 'unified_api'
#                         }
                        
#                         column_entities.append(entity_mapping)
                    
#                     if column_entities:
#                         processed_columns[col] = {
#                             'entities': column_entities,
#                             'primary_fhir_resource': column_entities[0]['fhir_resource'],
#                             'category': column_entities[0].get('category', ''),
#                         }
#                     else:
#                         exception_columns.append(col)
#                 else:
#                     exception_columns.append(col)
#             else:
#                 print(f"API error for column {col}: {response.status_code}")
#                 exception_columns.append(col)
                
#         except Exception as e:
#             print(f"Error processing column {col}: {e}")
#             exception_columns.append(col)
    
#     return processed_columns, exception_columns


def process_column_via_api(col_name, max_retries=3):
    """
    Process a single column through the unified API with retry logic.
    Returns a tuple of (col_name, result_data, success_flag)
    """
    print(f"🔍 Starting API processing for column: {col_name}")
    
    for attempt in range(max_retries):
        try:
            # Make API request for this column
            payload = {
                "keywords": col_name,
                "threshold": 0.5,
                "limit": 1
            }
            
            print(f"📤 Attempt {attempt + 1} for '{col_name}' - Payload: {payload}")
            print(f"📡 API Endpoint: {API_ENDPOINT}")
            
            # Make request to the unified API with increased timeout
            response = requests.post(API_ENDPOINT, json=payload, timeout=600)
            
            print(f"📥 Response for '{col_name}': Status {response.status_code}")
            
            if response.status_code == 200:
                try:
                    results = response.json()
                    print(f"✅ Parsed JSON for '{col_name}': {len(results) if isinstance(results, list) else 'Not a list'} results")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error for '{col_name}': {e}")
                    print(f"Raw response text: {response.text[:200]}...")
                    if attempt == max_retries - 1:
                        return (col_name, None, False)
                    continue
                
                if results and isinstance(results, list) and len(results) > 0:
                    print(f"🎯 Processing {len(results)} results for '{col_name}'")
                    
                    # Process all entities found for this column
                    column_entities = []
                    
                    for i, result in enumerate(results):
                        print(f"  Result {i+1}: {result.get('conceptid', 'NO_ID')} - {result.get('conceptid_name', 'NO_NAME')}")
                        
                        # Updated required fields - only conceptid and conceptid_name are required
                        required_fields = ["conceptid", "conceptid_name"]
                        missing_fields = [field for field in required_fields if field not in result]
                        
                        if missing_fields:
                            print(f"  ⚠️ Missing fields in result {i+1}: {missing_fields}")
                            continue
                        
                        # Use the concept name directly from API
                        concept_name = result["conceptid_name"]
                        
                        # Determine FHIR resource type from API response or use default
                        fhir_resource = result.get("fhir_resource", "observation")
                        
                        # Handle special cases for FHIR resource mapping
                        if 'age' in concept_name.lower() and 'age' in col_name.lower():
                            fhir_resource = 'Patient.Age'
                        elif 'gender' in concept_name.lower():
                            fhir_resource = 'Patient.Gender'
                        
                        # Create entity mapping with all available fields
                        entity_mapping = {
                            'entity_name': concept_name,
                            'original_query': col_name,
                            'snomed_id': result["conceptid"],
                            'snomed_name': concept_name,
                            'snomed_full_name': concept_name,
                            'toplevelhierarchy_name': result.get("toplevelhierarchy_name", "Unknown"),  # Optional with default
                            'fhir_resource': fhir_resource,
                            'category': result.get("category", ""),
                            'score': result.get("match_score", 0),
                            'suggested_name': result.get("suggested_name", ""),  # Add suggested_name from API
                            'source': 'unified_api'
                        }
                        
                        column_entities.append(entity_mapping)
                    
                    if column_entities:
                        success_data = {
                            'entities': column_entities,
                            'primary_fhir_resource': column_entities[0]['fhir_resource'],
                            'category': column_entities[0].get('category', ''),
                        }
                        print(f"✅ Successfully processed '{col_name}' with {len(column_entities)} entities")
                        return (col_name, success_data, True)
                    else:
                        print(f"❌ No valid entities found for '{col_name}'")
                        return (col_name, None, False)
                else:
                    print(f"❌ No results or invalid results for '{col_name}': {results}")
                    return (col_name, None, False)
            else:
                print(f"❌ API error for column '{col_name}' (attempt {attempt + 1}): {response.status_code}")
                print(f"Response text: {response.text[:200]}...")
                if attempt == max_retries - 1:
                    return (col_name, None, False)
                time.sleep(1)  # Wait before retry
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout for column '{col_name}' (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return (col_name, None, False)
            time.sleep(1)
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 Connection error for column '{col_name}' (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return (col_name, None, False)
            time.sleep(1)
        except Exception as e:
            print(f"💥 Unexpected error processing column '{col_name}' (attempt {attempt + 1}): {e}")
            import traceback
            traceback.print_exc()
            if attempt == max_retries - 1:
                return (col_name, None, False)
            time.sleep(1)
    
    print(f"❌ All retries failed for column '{col_name}'")
    return (col_name, None, False)


def process_unified_api_response(columns_list, max_workers=10):  # Reduced workers
    """
    Process columns through the unified API using ThreadPoolExecutor for parallel processing.
    Returns structured data for display and mapping.
    """
    processed_columns = {}
    exception_columns = []
    
    print(f"🚀 Processing {len(columns_list)} columns using {max_workers} parallel workers...")
    print(f"📋 Columns to process: {columns_list}")
    print(f"🌐 API Endpoint: {API_ENDPOINT}")
    
    # Test API connectivity first
    try:
        test_response = requests.get(API_ENDPOINT.replace('/search', '/'), timeout=10)
        print(f"🔍 API connectivity test: Status {test_response.status_code}")
    except Exception as e:
        print(f"⚠️ API connectivity test failed: {e}")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"📤 Submitting {len(columns_list)} tasks to executor...")
        
        # Submit all tasks
        future_to_column = {
            executor.submit(process_column_via_api, col): col 
            for col in columns_list
        }
        
        print(f"✅ Submitted {len(future_to_column)} futures")
        
        # Process results as they complete
        completed_count = 0
        for future in tqdm(as_completed(future_to_column), 
                          total=len(columns_list), 
                          desc="Processing API responses"):
            
            try:
                col_name, result_data, success = future.result(timeout=600)  # Add timeout
                completed_count += 1
                
                if success and result_data:
                    processed_columns[col_name] = result_data
                    print(f"✅ Processed {col_name} ({completed_count}/{len(columns_list)})")
                else:
                    exception_columns.append(col_name)
                    print(f"❌ Failed to process {col_name} ({completed_count}/{len(columns_list)})")
            except Exception as e:
                col_name = future_to_column[future]
                exception_columns.append(col_name)
                completed_count += 1
                print(f"💥 Exception processing {col_name}: {e}")
    
    print(f"\n📊 API Processing Summary:")
    print(f"✅ Successfully processed: {len(processed_columns)} columns")
    print(f"❌ Failed to process: {len(exception_columns)} columns")
    
    if exception_columns:
        print(f"📝 Failed columns: {exception_columns}")
    
    return processed_columns, exception_columns


def get_suggested_names_from_adarv_dict(columns_list):
    """
    Get suggested names from adarv_data_dict table using vector similarity.
    Returns a dictionary mapping column names to suggested names.
    """
    import psycopg2
    import psycopg2.extras
    from config import DB_CONFIG
    
    suggested_names = {}
    
    # Create connection config without table names
    connection_config = {
        'dbname': DB_CONFIG['dbname'],
        'user': DB_CONFIG['user'],
        'password': DB_CONFIG['password'],
        'host': DB_CONFIG['host'],
        'port': DB_CONFIG['port']
    }
    
    # Get table name from config
    adarv_dict_table = DB_CONFIG.get('adarv_dict_table', 'adarv_data_dict')
    
    # Establish fresh database connection
    try:
        conn = psycopg2.connect(**connection_config)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Initialize embedding model
        encoder = initialize_emb_model()[1]
        
        for col in tqdm(columns_list, desc="Getting suggested names from adarv_data_dict"):
            try:
                # Start a new transaction for each column to avoid transaction errors
                conn.rollback()  # Reset any previous transaction state
                
                # Generate embedding for column name
                vector = encoder.encode(col)
                vector_str = '[' + ', '.join(map(str, vector)) + ']'
                
                # Use the adarv_data_dict table with suggested_name column
                query = f"""
                    SELECT *, (1 - (embeddings <=> %s::vector)) as score
                    FROM {adarv_dict_table}
                    ORDER BY score DESC
                    LIMIT 1;
                """
                
                cur.execute(query, (vector_str,))
                result = cur.fetchone()
                
                if result and result['score'] > 0.75:  # Use threshold of 0.75
                    suggested_name = result.get('suggested_name', '')
                    if suggested_name:
                        suggested_names[col] = suggested_name
                        print(f"Found suggested name for {col}: {suggested_name}")
                
            except Exception as e:
                print(f"Error getting suggested name for {col}: {e}")
                # Reset connection state on error
                try:
                    conn.rollback()
                except:
                    pass
                continue
        
        # Close connection
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Database connection error in get_suggested_names_from_adarv_dict: {e}")
        return {}
    
    return suggested_names

# def generate_column_names(processed_columns, columns_list):
#     """
#     Generate column names using ADARV suggested names and fallback logic for multiple entities.
#     """
#     # First, get suggested names from ADARV dict table
#     adarv_suggested_names = get_suggested_names_from_adarv_dict(columns_list)
    
#     column_renamed_mapping = {}
#     used_names = set()
    
#     # Process each column
#     for col_name in columns_list:
#         # Check if we have an ADARV suggested name
#         if col_name in adarv_suggested_names:
#             suggested_name = adarv_suggested_names[col_name]
            
#             # Handle duplicates
#             base_name = suggested_name
#             counter = 1
#             while suggested_name in used_names:
#                 suggested_name = f"{base_name}_{counter}"
#                 counter += 1
            
#             column_renamed_mapping[col_name] = suggested_name
#             used_names.add(suggested_name)
#             print(f"Using ADARV suggested name for {col_name}: {suggested_name}")
            
#         # If no ADARV suggested name but we have API entities, generate based on entities
#         elif col_name in processed_columns:
#             col_data = processed_columns[col_name]
#             entities = col_data['entities']
            
#             if entities:
#                 # Get FHIR resource from first entity
#                 fhir_resource = entities[0]['fhir_resource']
                
#                 # Extract prefix from FHIR resource
#                 if fhir_resource.lower().startswith('condition'):
#                     prefix = 'con'
#                 elif fhir_resource.lower().startswith('patient.gender'):
#                     prefix = 'gen'
#                 elif fhir_resource.lower().startswith('patient.age'):
#                     prefix = 'age'
#                 elif fhir_resource.lower().startswith('patient'):
#                     prefix = 'pat'
#                 else:
#                     prefix = fhir_resource[:3].lower()
                
#                 # Generate name based on entities
#                 if len(entities) == 1:
#                     # Single entity - use first 3 chars of concept name
#                     concept_name = entities[0].get('snomed_name', '').strip()
#                     if concept_name:
#                         # Get first word and take first 3 chars
#                         first_word = concept_name.split(',')[0].strip()
#                         clean_name = first_word.lower().replace(' ', '').replace('(', '').replace(')', '')
#                         abbr_col = f"{prefix}_{clean_name[:3]}"
#                     else:
#                         abbr_col = f"{prefix}_{col_name[:3].lower()}"
                        
#                 else:
#                     # Multiple entities - use first 3 chars of each concept name
#                     entity_parts = []
#                     for entity in entities[:3]:  # Limit to first 3 entities
#                         concept_name = entity.get('snomed_name', '').strip()
#                         if concept_name:
#                             first_word = concept_name.split(',')[0].strip()
#                             clean_name = first_word.lower().replace(' ', '').replace('(', '').replace(')', '')
#                             entity_parts.append(clean_name[:3])
                    
#                     if entity_parts:
#                         abbr_col = f"{prefix}_{'_'.join(entity_parts)}"
#                     else:
#                         abbr_col = f"{prefix}_{col_name[:3].lower()}"
                
#                 # Handle duplicates
#                 base_abbr = abbr_col
#                 counter = 1
#                 while abbr_col in used_names:
#                     abbr_col = f"{base_abbr}_{counter}"
#                     counter += 1
                
#                 column_renamed_mapping[col_name] = abbr_col
#                 used_names.add(abbr_col)
#                 print(f"Generated name for {col_name}: {abbr_col}")
#             else:
#                 # Fallback if no entities
#                 abbr_col = f"col_{col_name[:3].lower()}"
                
#                 # Handle duplicates
#                 base_abbr = abbr_col
#                 counter = 1
#                 while abbr_col in used_names:
#                     abbr_col = f"{base_abbr}_{counter}"
#                     counter += 1
                
#                 column_renamed_mapping[col_name] = abbr_col
#                 used_names.add(abbr_col)
#         else:
#             # No API data and no ADARV suggested name - create simple fallback
#             vowels = "aeiouAEIOU0123456789"
#             no_vowels = "".join([char for char in col_name if char.lower() not in vowels])
#             no_vowels = remove_punctuation(no_vowels.replace(" ", ""))
            
#             abbr_col = no_vowels[:6].lower() if no_vowels else col_name[:6].lower()
            
#             # Handle duplicates
#             base_name = abbr_col
#             counter = 1
#             while abbr_col in used_names:
#                 abbr_col = f"{base_name}_{counter}"
#                 counter += 1
            
#             column_renamed_mapping[col_name] = abbr_col
#             used_names.add(abbr_col)
    
#     return column_renamed_mapping

##############CORRECT LOGIC##############################

# def generate_column_names(processed_columns, columns_list):
#     """
#     Generate column names using ADARV suggested names and fallback logic for multiple entities.
#     This follows the same logic as fhirconv.py - prioritize ADARV suggested names first.
#     """
#     # First, get suggested names from ADARV dict table
#     adarv_suggested_names = get_suggested_names_from_adarv_dict(columns_list)
    
#     column_renamed_mapping = {}
#     used_names = set()
    
#     # FIRST PASS: Use ADARV suggested names where available (highest priority)
#     for col_name in columns_list:
#         if col_name in adarv_suggested_names:
#             suggested_name = adarv_suggested_names[col_name]
            
#             # Handle duplicates in suggested names
#             base_name = suggested_name
#             counter = 1
#             while suggested_name in used_names:
#                 suggested_name = f"{base_name}_{counter}"
#                 counter += 1
            
#             column_renamed_mapping[col_name] = suggested_name
#             used_names.add(suggested_name)
#             print(f"Using ADARV suggested name for {col_name}: {suggested_name}")
    
#     # SECOND PASS: Generate names for columns with unified API mappings (not ADARV mapped)
#     for col_name in columns_list:
#         # Skip if already named by ADARV
#         if col_name in column_renamed_mapping:
#             continue
            
#         # If we have API entities for this column, generate based on entities
#         if col_name in processed_columns:
#             col_data = processed_columns[col_name]
#             entities = col_data['entities']
            
#             if entities:
#                 # Get FHIR resource from first entity
#                 fhir_resource = entities[0]['fhir_resource']
                
#                 # Extract prefix from FHIR resource (same logic as fhirconv.py)
#                 if fhir_resource.lower().startswith('condition'):
#                     prefix = 'con'
#                 elif fhir_resource.lower().startswith('patient.gender'):
#                     prefix = 'gen'
#                 elif fhir_resource.lower().startswith('patient.age'):
#                     prefix = 'age'
#                 elif fhir_resource.lower().startswith('patient'):
#                     prefix = 'pat'
#                 else:
#                     prefix = fhir_resource[:3].lower()
                
#                 # Generate name based on entities (same logic as fhirconv.py)
#                 if len(entities) == 1:
#                     # Single entity - use first 3 chars of concept name
#                     concept_name = entities[0].get('snomed_name', '').strip()
#                     if concept_name:
#                         # Get first word and take first 3 chars
#                         first_word = concept_name.split(',')[0].strip()
#                         clean_name = first_word.lower().replace(' ', '').replace('(', '').replace(')', '')
#                         abbr_col = f"{prefix}_{clean_name[:3]}"
#                     else:
#                         abbr_col = f"{prefix}_{col_name[:3].lower()}"
                        
#                 else:
#                     # Multiple entities - use first 3 chars of each concept name
#                     entity_parts = []
#                     for entity in entities[:3]:  # Limit to first 3 entities
#                         concept_name = entity.get('snomed_name', '').strip()
#                         if concept_name:
#                             first_word = concept_name.split(',')[0].strip()
#                             clean_name = first_word.lower().replace(' ', '').replace('(', '').replace(')', '')
#                             entity_parts.append(clean_name[:3])
                    
#                     if entity_parts:
#                         abbr_col = f"{prefix}_{'_'.join(entity_parts)}"
#                     else:
#                         abbr_col = f"{prefix}_{col_name[:3].lower()}"
                
#                 # Handle duplicates
#                 base_abbr = abbr_col
#                 counter = 1
#                 while abbr_col in used_names:
#                     abbr_col = f"{base_abbr}_{counter}"
#                     counter += 1
                
#                 column_renamed_mapping[col_name] = abbr_col
#                 used_names.add(abbr_col)
#                 print(f"Generated name for {col_name}: {abbr_col}")
    
#     # THIRD PASS: Generate fallback names for columns that don't have mappings at all
#     # (same logic as fhirconv.py)
#     for col_name in columns_list:
#         if col_name not in column_renamed_mapping:
#             # Create a simple fallback abbreviation
#             vowels = "aeiouAEIOU0123456789"
#             no_vowels = "".join([char for char in col_name if char.lower() not in vowels])
#             no_vowels = remove_punctuation(no_vowels.replace(" ", ""))
            
#             # Create a simple abbreviation
#             abbr_col = no_vowels[:6].lower() if no_vowels else col_name[:6].lower()
            
#             # Handle duplicates
#             base_name = abbr_col
#             counter = 1
#             while abbr_col in used_names:
#                 abbr_col = f"{base_name}_{counter}"
#                 counter += 1
            
#             column_renamed_mapping[col_name] = abbr_col
#             used_names.add(abbr_col)
    
#     return column_renamed_mapping


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
    
    # FIRST PASS: Use API suggested_name where available (highest priority)
    for col_name in columns_list:
        if col_name in processed_columns:
            col_data = processed_columns[col_name]
            entities = col_data['entities']
            
            # Check if any entity has a suggested_name from API
            api_suggested_name = None
            for entity in entities:
                suggested_name = entity.get('suggested_name', '').strip()
                if suggested_name:
                    api_suggested_name = suggested_name
                    break  # Use the first suggested name found
            
            if api_suggested_name:
                # Handle duplicates in API suggested names
                base_name = api_suggested_name
                counter = 1
                while api_suggested_name in used_names:
                    api_suggested_name = f"{base_name}_{counter}"
                    counter += 1
                
                column_renamed_mapping[col_name] = api_suggested_name
                used_names.add(api_suggested_name)
                print(f"Using API suggested name for {col_name}: {api_suggested_name}")
    
    # SECOND PASS: Use ADARV suggested names for unmapped columns
    for col_name in columns_list:
        # Skip if already named by API suggested_name
        if col_name in column_renamed_mapping:
            continue
            
        if col_name in adarv_suggested_names:
            suggested_name = adarv_suggested_names[col_name]
            
            # Handle duplicates in suggested names
            base_name = suggested_name
            counter = 1
            while suggested_name in used_names:
                suggested_name = f"{base_name}_{counter}"
                counter += 1
            
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
                base_abbr = abbr_col
                counter = 1
                while abbr_col in used_names:
                    abbr_col = f"{base_abbr}_{counter}"
                    counter += 1
                
                column_renamed_mapping[col_name] = abbr_col
                used_names.add(abbr_col)
                print(f"Generated entity-based name for {col_name}: {abbr_col}")
    
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
            base_name = abbr_col
            counter = 1
            while abbr_col in used_names:
                abbr_col = f"{base_name}_{counter}"
                counter += 1
            
            column_renamed_mapping[col_name] = abbr_col
            used_names.add(abbr_col)
            print(f"Generated fallback name for {col_name}: {abbr_col}")
    
    return column_renamed_mapping

# Update the search_and_select_top_match function to work with the unified API
def search_and_select_top_match(entity, col_name, fhir_resource, category=""):
    """
    Search for SNOMED codes for an entity using the unified API
    Returns a dict with the mapping information or None if no match found
    """
    try:
        # Make API request to search for SNOMED codes
        payload = {
            "keywords": entity,
            "threshold": 0.5,
            "limit": 10
        }
        
        # Make request to the unified API
        response = requests.post(API_ENDPOINT, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            results = response.json()
            
            # If we have results, select the top one
            if results and isinstance(results, list) and len(results) > 0:
                top_result = results[0]
                
                # Validate the result has all required fields
                if not all(k in top_result for k in ["conceptid", "conceptid_name", "toplevelhierarchy_name"]):
                    print(f"Invalid result format for entity {entity}")
                    return None
                
                # Use concept name directly from API
                concept_name = top_result["conceptid_name"]
                
                # Create and return the mapping
                return {
                    "snomed_id": top_result["conceptid"],
                    "snomed_name": concept_name,
                    "snomed_full_name": concept_name,
                    "fhir_resource": fhir_resource,
                    "category": category,
                    "source": "unified_api"
                }
            
            print(f"No results found for entity {entity}")
            return None
        else:
            print(f"API error for entity {entity}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error searching SNOMED for entity {entity}: {e}")
        return None
    

# Modify the open_entity_edit_modal callback to better highlight the selected entity
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

# Callback to close the modal
@app.callback(
    Output("entity-search-modal", "is_open", allow_duplicate=True),
    Input("modal-close", "n_clicks"),
    prevent_initial_call=True  # Already has this, good!
)
def close_modal(n_clicks):
    return False

# Update the export mappings callback to use resource IDs in filename
# Update the export mappings function to include renamed columns
@app.callback(
    Output("download-mappings", "data"),
    Input("export-button", "n_clicks"),
    [State("snomed-mappings", "data"),
     State("renamed-columns", "data"),  # Add renamed-columns state
     State("resource-ids", "data"),
     State("original-filename", "data")],
    prevent_initial_call=True
)
def export_mappings(n_clicks, mappings_json, renamed_cols_json, resource_ids_json, original_filename):
    if not mappings_json:
        return None
    
    try:
        mappings = json.loads(mappings_json)
        
        # Get renamed columns if available
        renamed_cols = {}
        if renamed_cols_json:
            try:
                renamed_cols = json.loads(renamed_cols_json)
            except:
                pass
        
        # Get resource IDs if available
        resource_ids = {}
        if resource_ids_json:
            try:
                resource_ids = json.loads(resource_ids_json)
            except:
                pass
        
        # Convert to DataFrame for export
        rows = []
        for col, entities_map in mappings.items():
            # Get the renamed column from our dynamically generated names
            renamed_column = renamed_cols.get(col, col)
            
            for entity, mapping in entities_map.items():
                rows.append({
                    "Column": col,
                    "Renamed_Column": renamed_column,  # Use our dynamic column name
                    "Entity": entity,
                    "SNOMED_ID": mapping["snomed_id"],
                    "SNOMED_Name": mapping["snomed_name"],
                    "SNOMED_Full_Name": mapping.get("snomed_full_name", mapping["snomed_name"]),
                    "FHIR_Resource": mapping["fhir_resource"],
                    "Category": mapping.get("category", "")
                })
        
        # Extract info from resource_ids
        res_grp_id = resource_ids.get('resGrpId', 'unknown_group')
        rid = resource_ids.get('rid', 'unknown_resource')
        username = resource_ids.get('userName', 'unknown_user')
        dataset_type = resource_ids.get('typeOfDataset', 'unknown_type')
        
        # Clean the filename components
        res_grp_id = re.sub(r'[^a-zA-Z0-9]', '_', res_grp_id)
        username = re.sub(r'[^a-zA-Z0-9@.]', '_', username)
        dataset_type = re.sub(r'[^a-zA-Z0-9]', '_', dataset_type)
        original_filename = original_filename or "file"
        original_filename = re.sub(r'[^a-zA-Z0-9]', '_', original_filename)
        
        # Generate the custom filename with resource group ID
        custom_filename = f"{res_grp_id}_{username}_{dataset_type}_{original_filename}_snomed_mappings.csv"
        
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, custom_filename, index=False)
    except Exception as e:
        print(f"Error exporting mappings: {e}")
        return None
    
# Initialize values for store components to prevent errors
@app.callback(
    Output("snomed-mappings", "data"),
    Input("stored-data", "data")
)
def initialize_mappings(data):
    return json.dumps({})

# Init final mapping content
@app.callback(
    Output("final-mapping-content", "children", allow_duplicate=True),
    Input("stored-data", "data"),
    prevent_initial_call=True  # Add this line
)
def initialize_final_mapping(data):
    return html.P("Select matches from the Entity Search tab to build your final mapping.")

# Update the final mapping tab callback to include delete buttons
@app.callback(
    Output("final-mapping-content", "children"),
    [Input("snomed-mappings", "data"),
     Input("renamed-columns", "data"),
     Input("stored-data", "data")],
    prevent_initial_call=True
)
def update_final_mapping_tab(mappings_json, renamed_cols_json, data_json):
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
            dcc.Download(id="download-mappings")
        ])
        
    except Exception as e:
        print(f"Error updating final mapping tab: {e}")
        import traceback
        traceback.print_exc()
        return html.P(f"Error generating final mapping: {str(e)}")

# Content for entity search tab
@app.callback(
    Output("entity-search-content", "children"),
    Input("extracted-entities", "data")
)
def update_entity_search_tab(entities_json):
    if not entities_json:
        return html.P("No entities extracted yet.")
    
    return html.Div([
        html.P("Click on a column in the Column Mapping tab to search for entities."),
        html.Hr(),
        html.P([
            html.Strong("Instructions:"), 
            html.Br(),
            "1. In the Column Mapping tab, click on the 'Actions' column for any row",
            html.Br(),
            "2. Select an entity from the popup to search for SNOMED codes",
            html.Br(),
            "3. Choose the best match from the search results",
           
            html.Br(),
            "4. View and export your final mappings from the Final Mapping tab"
        ])
    ])

# Update in the FHIR processing tab callback
@app.callback(
    Output("fhir-processing-content", "children"),
    [Input("stored-data", "data"),
     Input("snomed-mappings", "data"),
     Input("resource-ids", "data")]  # Add resource-ids as input
)
def update_fhir_processing_tab(json_data, mappings_json, resource_ids_json):
    if not json_data or not mappings_json:
        return html.Div([
            html.P("Upload a dataset and create SNOMED mappings first."),
            html.P("Once you've mapped your entities to SNOMED codes in the previous tabs, you can generate FHIR resources here.")
        ])
    
    try:
        mappings = json.loads(mappings_json)
        if not mappings:
            return html.Div([
                html.P("No mappings available. Please create mappings in the Entity Search tab first."),
            ])
        
        # Load resource IDs if available
        dataset_name_default = ""
        resource_id_display = ""
        
        if resource_ids_json:
            try:
                resource_ids = json.loads(resource_ids_json)
                rid = resource_ids.get('rid', '')
                res_grp_id = resource_ids.get('resGrpId', '')
                
                # Use resource ID directly for default dataset name (without dataset_ prefix)
                if rid:
                    dataset_name_default = rid
                    
                # Display resource IDs
                if rid or res_grp_id:
                    resource_id_display = html.Div([
                        html.H6("Resource Information:"),
                        html.Div([
                            html.Strong("Resource ID: "), 
                            html.Span(rid)
                        ]) if rid else None,
                        html.Div([
                            html.Strong("Resource Group ID: "), 
                            html.Span(res_grp_id)
                        ]) if res_grp_id else None
                    ], className="mb-3 p-2 border rounded bg-light")
            except:
                pass
                        
        return html.Div([
            html.H5("Generate FHIR Resources"),
            html.P("Create FHIR resources from your dataset using the SNOMED mappings you've defined."),
            
            # Show resource IDs if available
            resource_id_display,
            
            html.Hr(),
            
            dbc.Form([
                html.Div([
                    dbc.Label("Dataset Name (for Group resource):", html_for="dataset-name-input"),
                    dbc.Input(id="dataset-name-input", type="text", 
                             value=dataset_name_default,
                             placeholder="Enter a name for this dataset"),
                ], className="mb-3"),
                
                html.Div([
                    dbc.Label("FHIR Server URL:", html_for="fhir-server-url"),
                    dbc.Input(id="fhir-server-url", type="text", 
                             value="http://65.0.127.208:30007/fhir", 
                             placeholder="Enter FHIR server URL"),
                ], className="mb-3"),
                
                html.Div([
                    dbc.Checkbox(id="enable-fhir-upload", className="me-2"),
                    dbc.Label("Enable FHIR server upload (otherwise just generate bundle)", 
                             html_for="enable-fhir-upload"),
                ], className="mb-3"),
                
                dbc.Button("Generate FHIR Resources", id="generate-fhir-button", 
                          color="success", className="mt-3"),
            ]),
            
            html.Div(id="fhir-process-status", className="mt-3"),
            html.Div(id="fhir-download-section", className="mt-3"),
        ])
    except Exception as e:
        return html.Div([
            html.P(f"Error preparing FHIR processing tab: {str(e)}"),
        ])

@app.callback(
    [Output("fhir-process-status", "children"),
     Output("fhir-download-section", "children"),
     Output("fhir-processing-output", "children")],
    [Input("generate-fhir-button", "n_clicks")],
    [State("stored-data", "data"),
     State("snomed-mappings", "data"),
     State("resource-ids", "data"),
     State("dataset-name-input", "value"),
     State("enable-fhir-upload", "checked"),
     State("fhir-server-url", "value")]
)
def generate_fhir_resources(n_clicks, json_data, mappings_json, resource_ids_json, dataset_name, enable_upload, server_url):
    # Prevent the callback from running on page load
    if not n_clicks:
        raise PreventUpdate
    
    if not json_data or not mappings_json:
        return html.P("No data or mappings available."), None, None
    
    try:
        # Load resource IDs if available
        if resource_ids_json:
            resource_ids = json.loads(resource_ids_json)
            rid = resource_ids.get('rid', '')
            res_grp_id = resource_ids.get('resGrpId', '')
            disease_name = resource_ids.get('diseaseName', '')
        else:
            rid = ''
            res_grp_id = ''
            disease_name = ''
        
        # Use resource ID directly without dataset_ prefix
        if not dataset_name:
            if rid:
                dataset_name = rid
            else:
                dataset_name = f"{int(time.time())}"
        
        # For FHIR, include the resource ID in metadata if available
        metadata = {
            "resourceId": rid,
            "resourceGroupId": res_grp_id,
            "diseaseName": disease_name
        }
                
        # Load the original dataframe
        df = pd.read_json(json_data, orient='split')
        mappings = json.loads(mappings_json)
        
        # Initialize attributes for all columns
        for col_name in df.columns:
            # Create attrs dictionary if it doesn't exist
            if not hasattr(df[col_name], 'attrs'):
                df[col_name].attrs = {}
            
            # Initialize with default values to prevent KeyErrors
            df[col_name].attrs.setdefault('FHIR_Resource', 'observation')
            df[col_name].attrs.setdefault('Entities', [])
            df[col_name].attrs.setdefault('valueSet', 'valueString')
            df[col_name].attrs.setdefault('valueModifier', None)
        
        # Apply mappings to the dataframe by adding attributes
        for col_name, entities_map in mappings.items():
            if col_name not in df.columns:
                continue
                
            # Get the first entity mapping to determine FHIR resource type
            try:
                first_entity_key = next(iter(entities_map))
                first_entity = entities_map[first_entity_key]
                fhir_resource = first_entity.get("fhir_resource", "observation")
                
                # Set FHIR resource type as attribute
                df[col_name].attrs["FHIR_Resource"] = fhir_resource
                
                # Collect entity information for the column
                entities_list = []
                for entity_name, entity_data in entities_map.items():
                    # Create properly structured entity dictionary
                    entity_dict = {
                        'Entity': entity_name,
                        'SCT_Name': entity_data.get('snomed_name', ''),
                        'SCT_ID': entity_data.get('snomed_id', '')
                    }
                    entities_list.append(entity_dict)
                
                # Set entities attribute
                df[col_name].attrs["Entities"] = entities_list
            except Exception as e:
                print(f"Error processing mappings for column {col_name}: {e}")
                # If there's an error, keep the default attributes already set
        
        # Create valueSet attributes based on data types
        df = create_valueset(df)        

        # For debugging: verify that all columns have the necessary attributes
        for col_name in df.columns:
            print(f"Column {col_name} attributes:")
            for attr_name, attr_value in df[col_name].attrs.items():
                print(f"  {attr_name}: {attr_value}")
        
        # Generate FHIR bundle without uploading to server
        config = {
            "upload_to_server": enable_upload,
            "server_url": server_url if enable_upload else None
        }
        
        # Use the load_fhir_data function to generate FHIR resources
        # from mesop_demo_utils_v2_mod import load_fhir_data
        fullbundle, sample_bundle = load_fhir_data(df, dataset_name)
        
        # Count resources by type
        resource_counts = {}
        for bundle in fullbundle:
            for entry in bundle["entry"]:
                resource_type = entry["resource"]["resourceType"]
                if resource_type in resource_counts:
                    resource_counts[resource_type] += 1
                else:
                    resource_counts[resource_type] = 1
        
        # Create a status message
        status_message = [
            html.H5("FHIR Resources Generated Successfully"),
            html.P(f"Total bundles: {len(fullbundle)}"),
            html.Ul([
                html.Li(f"{resource_type}: {count} resources") 
                for resource_type, count in resource_counts.items()
            ])
        ]
        
        if enable_upload:
            status_message.append(html.P(f"Resources were uploaded to {server_url}", className="text-success"))
        else:
            status_message.append(html.P("Resources were generated but not uploaded to the FHIR server", className="text-warning"))
        
        # Create download buttons
        download_section = html.Div([
            dbc.Button("Download Full Bundle", id="download-full-bundle", color="primary", className="me-2"),
            dbc.Button("Download Sample Bundle", id="download-sample-bundle", color="info"),
            dcc.Download(id="download-bundle-data")
        ])
        
        # Store bundles in a hidden div for download
        hidden_storage = html.Div([
            html.Div(id="full-bundle-storage", children=json.dumps(fullbundle), style={"display": "none"}),
            html.Div(id="sample-bundle-storage", children=json.dumps(sample_bundle), style={"display": "none"})
        ])
        
        return status_message, download_section, hidden_storage
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        return [
            html.H5("Error Generating FHIR Resources", className="text-danger"),
            html.P(str(e)),
            html.Details([
                html.Summary("Error Details"),
                html.Pre(error_details)
            ], className="text-danger")
        ], None, None

# Update the bundle download callback to include resource IDs in filenames
@app.callback(
    Output("download-bundle-data", "data"),
    [Input("download-full-bundle", "n_clicks"),
     Input("download-sample-bundle", "n_clicks")],
    [State("full-bundle-storage", "children"),
     State("sample-bundle-storage", "children"),
     State("dataset-name-input", "value"),
     State("resource-ids", "data")]  # Add resource-ids as state
)
def download_bundle(full_clicks, sample_clicks, full_bundle_json, sample_bundle_json, dataset_name, resource_ids_json):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0] 
    
    # Get resource ID for filename
    rid = ""
    if resource_ids_json:
        try:
            resource_ids = json.loads(resource_ids_json)
            rid = resource_ids.get('rid', '')
        except:
            pass
    
    # Create filename with resource ID if available
    if not dataset_name:
        if rid:
            dataset_name = f"dataset_{rid}"
        else:
            dataset_name = f"dataset_{int(time.time())}"
    elif rid and rid not in dataset_name:
        dataset_name = f"{dataset_name}_{rid}"
    
    if button_id == "download-full-bundle":
        return dict(
            content=full_bundle_json,
            filename=f"{dataset_name}_full_bundle.json",
            type="application/json"
        )
    elif button_id == "download-sample-bundle":
        return dict(
            content=sample_bundle_json,
            filename=f"{dataset_name}_sample_bundle.json",
            type="application/json"
        )
    
    raise PreventUpdate

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)