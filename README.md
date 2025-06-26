# CSV to FHIR Converter Dash UI

## How to Run ?

Make sure that the Templates, Routes, Utils and Snomed directories are on the same level as the app.py file. Then ``` python app.py ```

## Backup

The ```fhirconv_4.py``` , ```demo_utils_3.py``` and ```manual_snomed_updater.py``` are the monolithic base version of the refactored code. If all else fails; load these 3 along with the Templates directory and run ```python fhirconv_4.py```


## Project Structure

### Core Application

```
app.py
├── routes/layout.py
├── routes/callbacks_upload.py
├── routes/callbacks_mapping.py (compatibility layer)
│   └── routes/callbacks_mapping_registry.py
│       ├── routes/callbacks_data_processing.py
│       ├── routes/callbacks_mapping_management.py
│       ├── routes/callbacks_mapping_edit.py
│       ├── routes/callbacks_final_mapping.py
│       └── routes/callbacks_fhir_processing.py
├── routes/callbacks_entity.py
└── routes/callbacks_category.py
```
### Helper Modules
```
routes/helpers.py (central aggregator)
├── routes/ui_components.py
├── routes/text_utils.py
├── routes/data_processing.py
├── routes/api_client.py
├── routes/db_operations.py
└── routes/column_naming.py
```

## File Function Documentation

### Layout & Main Callbacks

#### layout.py
Defines the main Dash application layout and UI structure. Creates the complete web interface with tabs, forms, tables, and interactive components for the CSV to FHIR converter.

#### callbacks_upload.py
Handles file upload functionality and initial data processing. Manages CSV/Excel file uploads, validates file formats, processes file content, and populates the initial data containers.

#### callbacks_entity.py
Manages entity search tab display and entity-related UI updates. Controls the entity search interface visibility and updates entity-related components based on mapping data changes.

#### callbacks_category.py
Handles category selection modal and category-related interactions. Manages the category dropdown modal that appears when users click on mapping table cells for categorization.

### Mapping System (Registry Pattern)

#### callbacks_mapping.py
Provides backward compatibility for the refactored mapping system. Acts as a thin wrapper that redirects imports to the new modular mapping registry system.

#### callbacks_mapping_registry.py
Central registry that coordinates all mapping-related callback modules. Imports and registers all mapping callbacks from specialized modules, providing a single entry point for mapping functionality.

#### callbacks_data_processing.py
Handles data processing and initial API interactions for mapping. Processes uploaded datasets, performs parallel API calls for entity recognition, and generates initial mapping tables.

#### callbacks_mapping_management.py
Manages CRUD operations on SNOMED mappings and mapping lifecycle. Handles deletion of mapping rows, updates column naming automatically, and refreshes mapping display tables.

#### callbacks_mapping_edit.py
Provides advanced mapping editing capabilities and entity processing. Handles complex entity searches, manual SNOMED selections, and detailed mapping modifications with validation.

#### callbacks_final_mapping.py
Displays final mapping summaries and provides export functionality. Creates comprehensive mapping overview tables, manages mapping export options, and provides final validation displays.

#### callbacks_fhir_processing.py
Handles FHIR resource generation and bundle creation from mapped data. Converts mapped data into FHIR resources, creates FHIR bundles, and manages FHIR server uploads with download capabilities.

### Utility & Helper Modules

#### helpers.py
Central hub that imports and re-exports functions from all specialized utility modules. Provides a single import point for all helper functions, simplifying imports across the application.

#### ui_components.py
Creates reusable UI components and widgets for the Dash interface. Generates column statistics displays, processing alerts, error messages, and standardized UI elements.

#### text_utils.py
Provides text processing and string manipulation utilities. Handles JSON parsing, text truncation, data sanitization, and entity string processing with validation.

#### data_processing.py
Handles file processing and data manipulation operations. Processes CSV/Excel files from file paths, performs data validation, and handles various file format conversions.

#### api_client.py
Manages external API interactions and SNOMED code searches. Handles API requests for entity recognition, processes unified API responses, and manages parallel API processing.

#### db_operations.py
Manages database connections and data dictionary operations. Connects to PostgreSQL database, performs vector similarity searches, and retrieves suggested column names from ADARV dictionary.

#### column_naming.py
Generates intelligent column names using multiple strategies. Creates column names using API suggestions, database lookups, entity-based logic, and fallback naming conventions.

## Utils Directory
```
Utils Directory Dependencies:
├── routes/callbacks_upload.py
│   └── utils/io_utils.py
├── routes/callbacks_fhir_processing.py
│   ├── utils/value_set.py
│   └── utils/fhir_creation.py
├── routes/column_naming.py
│   └── utils/embedding.py
├── routes/db_operations.py
│   └── utils/embedding.py
├── routes/callbacks_mapping_edit.py
│   └── utils/embedding.py
├── utils/fhir_creation.py
│   ├── utils/templates.py
│   ├── utils/snomed_processing.py
│   └── utils/fhir_resources.py
├── utils/column_utils.py
│   └── utils/embedding.py
└── External Files (fhirconv_4.py, demo_utils_3.py)
    └── utils/templates.py
```

```
SNOMED Directory Dependencies:
├── routes/callbacks_mapping_edit.py
│   └── snomed/manual_updater.py
├── routes/callbacks_entity_backup.py (unused)
│   └── snomed/manual_updater.py
├── snomed/manual_updater.py
│   └── utils/embedding.py
```

#### io_utils.py
Handles input/output operations and file processing for various data formats. Processes file uploads, sanitizes text for JSON, validates file formats, and manages base64 encoding/decoding operations.

#### embedding.py
Manages embedding model initialization and database connections for similarity searches. Initializes SentenceTransformer models, establishes PostgreSQL connections, and provides text preprocessing utilities for vector operations.

#### templates.py
Loads FHIR resource JSON templates from the templates directory. Reads and provides access to standardized FHIR resource templates (Patient, Observation, Condition, Location, etc.) for resource generation.

#### fhir_creation.py
Core FHIR resource creation engine with comprehensive resource type support. Creates FHIR Condition, Observation, Patient, Age, and other resources from mapped data with proper validation and structure.

#### fhir_resources.py
Extended FHIR resource creation utilities for specialized resource types. Handles Location, Group, NutritionIntake, Encounter, and ServiceRequest resource creation with specific business logic.

#### value_set.py
Creates value sets and data type mappings for FHIR resource attributes. Analyzes DataFrame columns to determine appropriate FHIR value types (boolean, string, numeric) and creates corresponding value sets.

#### snomed_processing.py
Processes SNOMED codes and performs sentiment analysis on medical data. Marks condition resources, performs sentiment analysis using transformers, and handles SNOMED code validation and processing.

## SNOMED Directory

#### manual_updater.py
Provides manual SNOMED code selection and database update functionality for entity mapping. Handles user-driven SNOMED code selection, performs semantic similarity searches, updates entity mappings, and manages manual overrides for automated mappings.
