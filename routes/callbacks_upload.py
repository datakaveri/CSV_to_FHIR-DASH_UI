"""
Upload and file processing callbacks for the Dash app.
"""

import json
import os
import pandas as pd
from urllib.parse import parse_qs
from dash import Input, Output, html
from dash.exceptions import PreventUpdate
from utils.io_utils import process_file_from_path, process_upload_content


def register_upload_callbacks(app):
    """Register all upload-related callbacks."""
    
    @app.callback(
        [Output('api-data-container', 'children'),
         Output('username', 'value'),
         Output('dataset-type', 'value'),
         Output('processing-status', 'children', allow_duplicate=True),
         Output('dataset-info-display', 'children')],
        [Input('url', 'search')],
        prevent_initial_call=True
    )
    def process_url_parameters(search):
        """Process URL parameters for API data."""
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

    @app.callback(
        [Output("stored-data", "data", allow_duplicate=True),
         Output("upload-status", "children", allow_duplicate=True),
         Output("processing-status", "children", allow_duplicate=True),
         Output("original-filename", "data", allow_duplicate=True),
         Output("resource-ids", "data")],
        [Input("api-data-container", "children")],
        prevent_initial_call=True
    )
    def load_file_from_parameters(api_data_json):
        """Load file from API parameters."""
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

    @app.callback(
        [Output("stored-data", "data", allow_duplicate=True),
         Output("upload-status", "children", allow_duplicate=True),
         Output("processing-status", "children", allow_duplicate=True),
         Output("original-filename", "data", allow_duplicate=True)],
        Input("upload-data", "contents"),
        Input("upload-data", "filename"),
        prevent_initial_call=True
    )
    def process_upload(contents, filename):
        """Process uploaded file."""
        if contents is None:
            raise PreventUpdate
        
        df, error = process_upload_content(contents, filename)
        
        if error:
            return None, error, error, None
        
        # Extract base filename without extension
        base_filename = os.path.splitext(filename)[0]
        
        return (
            df.to_json(orient='split'),
            f"File uploaded: {filename}",
            "Processing data...",
            base_filename
        )
