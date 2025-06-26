"""
Consolidated helper functions for the Dash app.
This module imports and re-exports functions from specialized modules.
"""

# Import all functions from specialized modules
from .ui_components import (
    create_column_stats,
    create_processing_alert,
    format_snomed_result,
    create_entity_button,
    create_error_display
)

from .text_utils import (
    safe_json_loads,
    truncate_text,
    get_sample_data_string,
    validate_snomed_code,
    sanitize_for_json,
    split_entities
)

from .data_processing import (
    process_file_from_path
)

from .api_client import (
    search_and_select_top_match,
    process_column_via_api,
    process_unified_api_response
)

from .db_operations import (
    get_suggested_names_from_adarv_dict
)

from .column_naming import (
    generate_column_names
)

# Re-export all functions for backward compatibility
__all__ = [
    # UI Components
    'create_column_stats',
    'create_processing_alert',
    'format_snomed_result',
    'create_entity_button',
    'create_error_display',
    
    # Text Utils
    'safe_json_loads',
    'truncate_text',
    'get_sample_data_string',
    'validate_snomed_code',
    'sanitize_for_json',
    'split_entities',
    
    # Data Processing
    'process_file_from_path',
    
    # API Client
    'search_and_select_top_match',
    'process_column_via_api',
    'process_unified_api_response',
    
    # Database Operations
    'get_suggested_names_from_adarv_dict',
    
    # Column Naming
    'generate_column_names'
]