"""
Dash app layout components.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table


def get_main_layout():
    """Return the main layout for the Dash app."""
    return dbc.Container([
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
                    html.Div([
                        # Add the DataTable directly here
                        dash_table.DataTable(
                            id="column-mapping-datatable",
                            data=[],
                            columns=[
                                {"name": "Original Column", "id": "Original Column", "editable": False},
                                {"name": "Abbreviated Name", "id": "Abbreviated Name", "editable": True},
                                {"name": "Category", "id": "Category", "editable": False},
                                {"name": "FHIR Resource", "id": "FHIR Resource", "editable": False},
                                {"name": "Entities", "id": "Entities", "editable": False},
                                {"name": "Sample Data", "id": "Sample Data", "editable": False}
                            ],
                            style_cell={
                                'textAlign': 'left',
                                'minWidth': '100px',
                                'maxWidth': '200px',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                            },
                            style_data={
                                'whiteSpace': 'normal',
                                'height': 'auto',
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'Category'},
                                    'backgroundColor': '#f0f0f0',
                                    'cursor': 'pointer'
                                }
                            ],
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            },
                            editable=True,
                            row_selectable="single",
                            page_action="native",
                            page_current=0,
                            page_size=20,
                            style_table={'overflowX': 'auto'}
                        )
                    ], id="column-mapping-table", className="mt-3")
                ]),
                # dbc.Tab(label="Entity Search", tab_id="tab-entity-search", children=[
                #     html.Div(id="entity-search-content", className="mt-3")
                # ]),
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
        dcc.Store(id="original-filename"),
        dcc.Store(id="current-category-column-index"),
        dcc.Store(id="resource-ids"),
        
        # Modal for entity search
        _get_entity_search_modal(),
        
        # Category selection modal
        _get_category_modal(),
        
        # Loading component
        dcc.Loading(
            id="loading-fhir",
            type="circle",
            children=html.Div(id="fhir-processing-output")
        ),
    ], fluid=True)


def _get_entity_search_modal():
    """Return the entity search modal component."""
    return dbc.Modal([
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
    ], id="entity-search-modal", size="lg")


def _get_category_modal():
    """Return the category selection modal component."""
    return dbc.Modal([
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
                    {'label': '', 'value': ''}
                ],
                clearable=False,
                style={"width": "100%"}
            ),
        ]),
        dbc.ModalFooter([
            dbc.Button("Save", id="save-category-button", className="ml-auto", color="primary"),
            dbc.Button("Cancel", id="cancel-category-button", className="ml-2")
        ]),
    ], id="category-selection-modal", size="md")
