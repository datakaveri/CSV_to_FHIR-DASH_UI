# Callbacks Refactoring

The large `callbacks_mapping.py` file has been refactored into smaller, more manageable modules for better maintainability and organization.

## New File Structure

### Original File
- `callbacks_mapping.py` - Large monolithic file with all mapping callbacks (DEPRECATED but kept for backward compatibility)

### New Modular Structure
- `callbacks_mapping_registry.py` - Main registry that imports and registers all callbacks
- `callbacks_data_processing.py` - Data processing and initial API processing callbacks
- `callbacks_mapping_management.py` - CRUD operations for SNOMED mappings
- `callbacks_final_mapping.py` - Final mapping display and export functionality
- `callbacks_fhir_processing.py` - FHIR resource generation and processing

## Module Responsibilities

### `callbacks_data_processing.py`
- `process_data()` - Main data processing callback that handles:
  - Loading and processing uploaded datasets
  - Parallel unified API processing
  - Column name generation
  - Entity extraction and mapping
  - Creating the initial mapping table display

### `callbacks_mapping_management.py`
- `delete_mapping_row()` - Handles deletion of individual SNOMED mappings:
  - Removes entity mappings
  - Updates column naming automatically
  - Refreshes the mapping display tables

### `callbacks_final_mapping.py`
- `update_final_mapping_tab()` - Updates the final mapping tab:
  - Creates summary views of all mappings
  - Displays mapping tables with delete functionality
  - Provides export capabilities

### `callbacks_fhir_processing.py`
- `update_fhir_processing_tab()` - Sets up the FHIR processing interface
- `generate_fhir_resources()` - Generates FHIR resources from mapped data:
  - Applies mappings to dataframe attributes
  - Creates FHIR bundles
  - Handles server upload (optional)
- `download_bundle()` - Handles FHIR bundle downloads

### `callbacks_mapping_registry.py`
- `register_mapping_callbacks()` - Main registration function that imports and registers all callbacks from the modular files

## Backward Compatibility

The original `callbacks_mapping.py` file has been converted to a compatibility layer that imports from the new registry. This ensures that existing code that imports from `callbacks_mapping` will continue to work without modification.

## Benefits of Refactoring

1. **Improved Maintainability**: Each module has a specific responsibility, making it easier to locate and modify specific functionality
2. **Better Organization**: Related callbacks are grouped together logically
3. **Easier Testing**: Individual modules can be tested in isolation
4. **Reduced File Size**: No more scrolling through 800+ lines to find specific functionality
5. **Better Collaboration**: Multiple developers can work on different modules simultaneously without conflicts
6. **Clearer Dependencies**: Import statements clearly show what each module depends on

## Usage

The refactoring is transparent to existing code. The main app continues to import:

```python
from routes.callbacks_mapping import register_mapping_callbacks
```

This will automatically use the new modular structure through the compatibility layer.

## Future Enhancements

Consider further breaking down modules if they grow too large:
- Extract table creation logic into a separate UI utilities module
- Create separate modules for different types of data processing
- Add dedicated error handling modules
