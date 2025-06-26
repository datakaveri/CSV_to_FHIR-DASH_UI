"""
FHIR processing callbacks for the Dash app.
Handles FHIR resource generation, upload, and download functionality.
"""

import json
import pandas as pd
import traceback
import time
from dash import Input, Output, State, html, dcc, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from utils.value_set import create_valueset
from utils.fhir_creation import load_fhir_data


def register_fhir_processing_callbacks(app):
    """Register FHIR processing related callbacks."""
    
    @app.callback(
        Output("fhir-processing-content", "children"),
        [Input("stored-data", "data"),
         Input("snomed-mappings", "data"),
         Input("resource-ids", "data")]
    )
    def update_fhir_processing_tab(json_data, mappings_json, resource_ids_json):
        """Update the FHIR processing tab content."""
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
        """Generate FHIR resources from the mapped data."""
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
            error_details = traceback.format_exc()
            
            return [
                html.H5("Error Generating FHIR Resources", className="text-danger"),
                html.P(str(e)),
                html.Details([
                    html.Summary("Error Details"),
                    html.Pre(error_details)
                ], className="text-danger")
            ], None, None

    @app.callback(
        Output("download-bundle-data", "data"),
        [Input("download-full-bundle", "n_clicks"),
         Input("download-sample-bundle", "n_clicks")],
        [State("full-bundle-storage", "children"),
         State("sample-bundle-storage", "children"),
         State("dataset-name-input", "value"),
         State("resource-ids", "data")]
    )
    def download_bundle(full_clicks, sample_clicks, full_bundle_json, sample_bundle_json, dataset_name, resource_ids_json):
        """Handle bundle download requests."""
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
