"""
Main mapping callbacks registry.
Imports and registers all mapping-related callbacks for the Dash app.
"""

from routes.callbacks_data_processing import register_data_processing_callbacks
from routes.callbacks_mapping_management import register_mapping_management_callbacks
from routes.callbacks_mapping_edit import register_mapping_edit_callbacks
from routes.callbacks_final_mapping import register_final_mapping_callbacks
from routes.callbacks_fhir_processing import register_fhir_processing_callbacks


def register_mapping_callbacks(app):
    """Register all mapping-related callbacks."""
    
    # Register data processing callbacks
    register_data_processing_callbacks(app)
    
    # Register mapping management callbacks
    register_mapping_management_callbacks(app)
    
    # Register mapping edit callbacks (new)
    register_mapping_edit_callbacks(app)
    
    # Register final mapping display callbacks
    register_final_mapping_callbacks(app)
    
    # Register FHIR processing callbacks
    register_fhir_processing_callbacks(app)
