"""
API client functions for external service interactions.
"""

import json
import requests
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import API_ENDPOINT


def search_and_select_top_match(entity, col_name, fhir_resource, category=""):
    """
    Search for SNOMED codes for an entity using the unified API
    Returns a dict with the mapping information or None if no match found
    """
    try:
        # Make API request to search for SNOMED codes
        payload = {
            "keywords": entity,
            "threshold": 0.5,
            "limit": 1
        }
        
        # Make request to the unified API
        response = requests.post(API_ENDPOINT, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            results = json.loads(response.text)
            
            # If we have results, select the top one
            if results and isinstance(results, list) and len(results) > 0:
                top_result = results[0]
                
                return {
                    "snomed_id": top_result.get("conceptid", ""),
                    "snomed_name": top_result.get("conceptid_name", ""),
                    "fhir_resource": fhir_resource,
                    "category": category,
                    "score": top_result.get("match_score", 0),
                    "source": "unified_api"
                }
            
            print(f"No results found for entity {entity}")
            return None
        else:
            print(f"API error for entity {entity}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error searching SNOMED for entity {entity}: {e}")
        traceback.print_exc()
        return None


def process_column_via_api(col_name, max_retries=3):
    """
    Process a single column through the unified API with retry logic.
    Returns a tuple of (col_name, result_data, success_flag)
    """
    print(f"🔍 Starting API processing for column: {col_name}")
    
    for attempt in range(max_retries):
        try:
            payload = {
                "keywords": col_name,
                "threshold": 0.5,
                "limit": 1
            }
            
            print(f"📤 API Request attempt {attempt + 1} for '{col_name}': {payload}")
            
            response = requests.post(API_ENDPOINT, json=payload, timeout=600)
            print(f"📥 API Response for '{col_name}': Status {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                print(f"✅ Successful response for '{col_name}': {len(results) if results else 0} results")
                
                if results and isinstance(results, list) and len(results) > 0:
                    # Process all entities found for this column
                    column_entities = []
                    
                    for result in results:
                        # Validate the result has all required fields
                        if not all(k in result for k in ["conceptid", "conceptid_name"]):
                            continue
                        
                        # Use the concept name directly from API
                        concept_name = result["conceptid_name"]
                        
                        # Determine FHIR resource type
                        fhir_resource = result.get("fhir_resource", "observation")
                        
                        # Handle special cases for FHIR resource mapping
                        if 'age' in concept_name.lower() and 'age' in col_name.lower():
                            fhir_resource = 'Patient.Age'
                        elif 'gender' in concept_name.lower():
                            fhir_resource = 'Patient.Gender'
                        
                        # Create entity mapping
                        entity_mapping = {
                            'entity_name': concept_name,
                            'original_query': col_name,
                            'snomed_id': result["conceptid"],
                            'snomed_name': concept_name,
                            'snomed_full_name': concept_name,
                            'toplevelhierarchy_name': result.get("toplevelhierarchy_name", ""),
                            'fhir_resource': fhir_resource,
                            'category': result.get("category", ""),
                            'score': result.get("match_score", 0),
                            'source': 'unified_api',
                            'suggested_name': result.get("suggested_name", "")  # Add suggested_name
                        }
                        
                        column_entities.append(entity_mapping)
                    
                    if column_entities:
                        processed_data = {
                            'entities': column_entities,
                            'primary_fhir_resource': column_entities[0]['fhir_resource'],
                            'category': column_entities[0].get('category', ''),
                        }
                        return (col_name, processed_data, True)
                    else:
                        print(f"⚠️ No valid entities found for column '{col_name}'")
                        return (col_name, None, False)
                else:
                    print(f"⚠️ Empty response for column '{col_name}'")
                    return (col_name, None, False)
            else:
                print(f"❌ API error for column '{col_name}': Status {response.status_code}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying column '{col_name}' (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    return (col_name, None, False)
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout for column '{col_name}' on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                continue
            else:
                return (col_name, None, False)
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 Connection error for column '{col_name}' on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return (col_name, None, False)
        except Exception as e:
            print(f"💥 Unexpected error for column '{col_name}' on attempt {attempt + 1}: {e}")
            traceback.print_exc()
            if attempt < max_retries - 1:
                continue
            else:
                return (col_name, None, False)
    
    print(f"❌ All retries failed for column '{col_name}'")
    return (col_name, None, False)


def process_unified_api_response(columns_list, max_workers=10):
    """
    Process columns through the unified API using ThreadPoolExecutor for parallel processing.
    Returns structured data for display and mapping.
    """
    processed_columns = {}
    exception_columns = []
    
    print(f"🚀 Processing {len(columns_list)} columns using {max_workers} parallel workers...")
    print(f"📋 Columns to process: {columns_list}")
    print(f"🌐 API Endpoint: {API_ENDPOINT}")
    
    # Test API connectivity first
    try:
        test_response = requests.get(API_ENDPOINT.replace('/search', '/'), timeout=600)
        print(f"🔍 API connectivity test: Status {test_response.status_code}")
    except Exception as e:
        print(f"⚠️ API connectivity test failed: {e}")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"📤 Submitting {len(columns_list)} tasks to executor...")
        
        # Submit all tasks
        future_to_column = {
            executor.submit(process_column_via_api, col): col 
            for col in columns_list
        }
        
        print(f"✅ Submitted {len(future_to_column)} futures")
        
        # Process results as they complete
        completed_count = 0
        for future in tqdm(as_completed(future_to_column), 
                          total=len(columns_list), 
                          desc="Processing API responses"):
            
            try:
                col_name, result_data, success = future.result()
                completed_count += 1
                
                print(f"📋 Completed {completed_count}/{len(columns_list)}: {col_name} -> {'Success' if success else 'Failed'}")
                
                if success and result_data:
                    processed_columns[col_name] = result_data
                    print(f"✅ Successfully processed column '{col_name}' with {len(result_data['entities'])} entities")
                else:
                    exception_columns.append(col_name)
                    print(f"❌ Failed to process column '{col_name}'")
                    
            except Exception as e:
                col_name = future_to_column[future]
                print(f"💥 Exception processing future for '{col_name}': {e}")
                exception_columns.append(col_name)
    
    print(f"\n📊 API Processing Summary:")
    print(f"✅ Successfully processed: {len(processed_columns)} columns")
    print(f"❌ Failed to process: {len(exception_columns)} columns")
    
    if exception_columns:
        print(f"📝 Failed columns: {exception_columns}")
    
    return processed_columns, exception_columns
