"""
UI component helper functions for the Dash app.
"""

from dash import html
import dash_bootstrap_components as dbc


def create_column_stats(df):
    """Create column statistics display."""
    if df is None:
        return html.Div("No data available")
    
    total_columns = len(df.columns)
    total_rows = len(df)
    
    # Calculate data type distribution
    numeric_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
    text_cols = len(df.select_dtypes(include=['object']).columns)
    datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
    
    return html.Div([
        html.P(f"Total Columns: {total_columns}"),
        html.P(f"Total Rows: {total_rows}"),
        html.P(f"Numeric Columns: {numeric_cols}"),
        html.P(f"Text Columns: {text_cols}"),
        html.P(f"DateTime Columns: {datetime_cols}")
    ])


def create_processing_alert(message, alert_type="info"):
    """Create a processing status alert."""
    color_map = {
        "success": "success",
        "error": "danger",
        "warning": "warning",
        "info": "info"
    }
    
    return dbc.Alert(
        message,
        color=color_map.get(alert_type, "info"),
        className="mb-3"
    )


def format_snomed_result(result, entity, col_name, fhir_resource, category):
    """Format a SNOMED search result for display."""
    score = result.get("match_score", 0)
    concept_id = result.get("conceptid", "")
    concept_name = result.get("conceptid_name", "")
    
    return dbc.Card([
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
                    "col": col_name,
                    "fhir": fhir_resource,
                    "category": category,
                    "entity": entity
                },
                color="primary",
                size="sm"
            )
        ])
    ], className="mb-2")


def create_entity_button(entity, col_name, fhir_resource, idx):
    """Create an entity search button."""
    return dbc.Button(
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


def create_error_display(error_message):
    """Create a standardized error display."""
    return dbc.Alert([
        html.H5("Error", className="alert-heading"),
        html.P(str(error_message)),
        html.Hr(),
        html.P("Please check your input and try again.", className="mb-0")
    ], color="danger")
