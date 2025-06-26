"""
Main mapping callbacks module (DEPRECATED).
This file has been refactored into smaller, focused modules.
Use callbacks_mapping_registry.py instead.
"""

# Import the new registry for backward compatibility
from routes.callbacks_mapping_registry import register_mapping_callbacks

# Re-export the main function for backward compatibility
__all__ = ['register_mapping_callbacks']