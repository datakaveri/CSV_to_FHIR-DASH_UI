"""
Main entry point for the CSV to FHIR Converter Dash application.
"""

import dash
import dash_bootstrap_components as dbc

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "CSV to FHIR Converter"

# Import and set layout after app is created to avoid circular imports
from routes.layout import get_main_layout
app.layout = get_main_layout()

# Register all callbacks after layout is set
from routes.callbacks_upload import register_upload_callbacks
from routes.callbacks_mapping import register_mapping_callbacks
from routes.callbacks_entity import register_entity_callbacks
from routes.callbacks_category import register_category_callbacks

register_upload_callbacks(app)
register_mapping_callbacks(app)
register_entity_callbacks(app)
register_category_callbacks(app)

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
