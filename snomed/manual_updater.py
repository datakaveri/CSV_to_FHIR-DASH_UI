"""
Manual SNOMED updater utilities.
"""

import json
import numpy as np
import pandas as pd
import re
import traceback
import psycopg2
from sentence_transformers import SentenceTransformer
from utils.embedding import initialize_emb_model
from typing import Dict, Any, Optional


def remove_bracketed_content(text: str) -> str:
    """
    Remove all content within brackets from the text.
    
    Args:
        text: Input text that may contain bracketed content
        
    Returns:
        Text with all bracketed content removed
    """
    # Remove content within parentheses, square brackets, and curly brackets
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Clean up extra spaces and commas
    text = re.sub(r'\s*,\s*,\s*', ', ', text)  # Remove double commas
    text = re.sub(r'^\s*,\s*', '', text)       # Remove leading comma
    text = re.sub(r'\s*,\s*$', '', text)       # Remove trailing comma
    text = re.sub(r'\s+', ' ', text)           # Normalize spaces
    
    return text.strip()


def find_most_similar_term(entity: str, concept_name: str) -> str:
    """
    Find the most similar term to the entity from the concept name using cosine similarity.
    
    Args:
        entity: The original entity that was searched for
        concept_name: The full concept name from SNOMED
        
    Returns:
        The most similar term from the concept name
    """
    try:
        # Remove bracketed content from the concept name
        cleaned_concept_name = remove_bracketed_content(concept_name)
        print(f"Original concept name: {concept_name}")
        print(f"After removing brackets: {cleaned_concept_name}")
        
        # Split the cleaned concept name by commas
        terms = [term.strip() for term in cleaned_concept_name.split(',') if term.strip()]
        
        if not terms:
            print("No terms found after cleaning, using original concept name")
            return concept_name
        
        if len(terms) == 1:
            print(f"Only one term found: {terms[0]}")
            return terms[0]
        
        print(f"Found {len(terms)} terms: {terms}")
        
        # Use sentence transformer for cosine similarity
        try:
            # Load model on CPU to avoid CUDA issues
            model = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
            
            # Encode the entity
            entity_embedding = model.encode([entity], show_progress_bar=False, batch_size=1)[0]
            
            # Encode all terms in batches
            batch_size = 5
            similarities = []
            
            for i in range(0, len(terms), batch_size):
                batch_terms = terms[i:i+batch_size]
                batch_embeddings = model.encode(batch_terms, show_progress_bar=False, batch_size=1)
                
                # Calculate similarity for each term in the batch
                for j, term_embedding in enumerate(batch_embeddings):
                    # Cosine similarity calculation
                    similarity = np.dot(entity_embedding, term_embedding) / (
                        np.linalg.norm(entity_embedding) * np.linalg.norm(term_embedding)
                    )
                    similarities.append((terms[i+j], similarity))
            
            # Sort by similarity and get the best match
            similarities.sort(key=lambda x: x[1], reverse=True)
            best_term, best_score = similarities[0]
            
            print(f"Similarity scores:")
            for term, score in similarities:
                print(f"  '{term}': {score:.3f}")
            
            print(f"Best match: '{best_term}' with score {best_score:.3f}")
            return best_term
            
        except Exception as e:
            print(f"Error in cosine similarity calculation: {e}")
            # Fallback: return the first term
            print(f"Falling back to first term: {terms[0]}")
            return terms[0]
            
    except Exception as e:
        print(f"Error in find_most_similar_term: {e}")
        # Ultimate fallback: return original concept name
        return concept_name


def update_snomed_database(snomed_id: str, updated_concept_name: str, new_embeddings: list) -> Dict[str, Any]:
    """
    Actually update the SNOMED database with the new concept name and embeddings.
    
    Args:
        snomed_id: The SNOMED concept ID to update
        updated_concept_name: The new concept name with (manual) tag
        new_embeddings: The new embedding vector as a list
        
    Returns:
        Dictionary containing the update result
    """
    try:
        # Initialize database connection
        cur, encoder = initialize_emb_model()
        
        # Prepare the update query
        update_query = """
        UPDATE snomed_ct_codes_tst 
        SET 
            ConceptID_name = %s,
            embeddings = %s
        WHERE ConceptID = %s
        """
        
        print(f"🔄 Executing database update...")
        print(f"SNOMED ID: {snomed_id}")
        print(f"New concept name: {updated_concept_name}")
        print(f"Embedding dimension: {len(new_embeddings)}")
        
        # Execute the update
        cur.execute(update_query, (updated_concept_name, new_embeddings, snomed_id))
        
        # Check how many rows were affected
        rows_affected = cur.rowcount
        
        if rows_affected == 0:
            return {
                "success": False,
                "message": f"No rows were updated. SNOMED ID {snomed_id} may not exist in the database.",
                "rows_affected": 0
            }
        elif rows_affected == 1:
            # Commit the transaction
            cur.connection.commit()
            
            # Verify the update by querying the record
            verification_query = """
            SELECT ConceptID, ConceptID_name, TopLevelHierarchy_name, fhir_resource
            FROM snomed_ct_codes_tst
            WHERE ConceptID = %s
            """
            
            cur.execute(verification_query, (snomed_id,))
            updated_record = cur.fetchone()
            
            if updated_record:
                return {
                    "success": True,
                    "message": f"Successfully updated SNOMED ID {snomed_id}",
                    "rows_affected": rows_affected,
                    "updated_record": {
                        "conceptid": updated_record['conceptid'],
                        "conceptid_name": updated_record['conceptid_name'],
                        "toplevelhierarchy_name": updated_record['toplevelhierarchy_name'],
                        "fhir_resource": updated_record['fhir_resource']
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Update committed but could not verify the record for SNOMED ID {snomed_id}",
                    "rows_affected": rows_affected
                }
        else:
            # Multiple rows affected - this shouldn't happen if ConceptID is unique
            cur.connection.rollback()
            return {
                "success": False,
                "message": f"Unexpected: {rows_affected} rows would be affected. Rolling back for safety.",
                "rows_affected": rows_affected
            }
            
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        # Rollback the transaction on error
        if 'cur' in locals() and cur.connection:
            cur.connection.rollback()
        
        return {
            "success": False,
            "message": f"Database error: {str(e)}",
            "error_type": "database_error"
        }
        
    except Exception as e:
        print(f"Unexpected error in database update: {e}")
        # Rollback the transaction on error
        if 'cur' in locals() and cur.connection:
            cur.connection.rollback()
        
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "error_type": "unexpected_error"
        }
        
    finally:
        # Close the database connection
        if 'cur' in locals() and cur:
            cur.close()


def update_snomed_for_manual_lookup(snomed_id: str, entity: str, current_concept_name: str, 
                                   simulate_only: bool = False) -> Dict[str, Any]:
    """
    Process manual SNOMED lookup to update the database with (manual) tag and recalculate embeddings.
    
    Args:
        snomed_id: The SNOMED concept ID that was manually looked up
        entity: The original entity that was searched for
        current_concept_name: The current concept name from the database
        simulate_only: If True, only simulate the update without actually executing it
    
    Returns:
        Dictionary containing the update information and results
    """
    
    try:
        # Step 1: Check if (manual) is already in the concept name
        if "(manual)" in current_concept_name.lower():
            print(f"SNOMED ID {snomed_id} already has manual tag")
            return {
                "action": "no_update_needed",
                "snomed_id": snomed_id,
                "entity": entity,
                "current_concept_name": current_concept_name,
                "reason": "Manual tag already exists",
                "database_result": None
            }
        
        # Step 2: Find the most similar term using cosine similarity
        print(f"Finding most similar term to entity '{entity}' from concept name...")
        most_similar_term = find_most_similar_term(entity, current_concept_name)
        
        # Step 3: Create the updated concept name with the similar term + (manual) tag
        updated_concept_name = f"{current_concept_name}, {most_similar_term} (manual)"
        
        print(f"Most similar term: '{most_similar_term}'")
        print(f"Updated concept name will be: {updated_concept_name}")
        
        # Step 4: Initialize the embedding model
        print("Initializing embedding model for recalculation...")
        try:
            # Use CPU to avoid CUDA OOM issues
            model = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return {
                "action": "error",
                "snomed_id": snomed_id,
                "entity": entity,
                "error": f"Failed to load embedding model: {str(e)}",
                "database_result": None
            }
        
        # Step 5: Calculate new embeddings for the updated concept name
        print(f"Calculating embeddings for updated concept: {updated_concept_name}")
        try:
            # Generate embedding with error handling
            new_embedding = model.encode([updated_concept_name], show_progress_bar=False, batch_size=1)[0]
            
            # Convert to list for JSON serialization and database storage
            embedding_list = new_embedding.tolist()
            
            print(f"Successfully calculated embeddings. Vector dimension: {len(embedding_list)}")
            
        except Exception as e:
            print(f"Error calculating embeddings: {e}")
            return {
                "action": "error",
                "snomed_id": snomed_id,
                "entity": entity,
                "error": f"Failed to calculate embeddings: {str(e)}",
                "database_result": None
            }
        
        # Step 6: Update the database (if not simulating)
        database_result = None
        if not simulate_only:
            print(f"🚀 EXECUTING REAL DATABASE UPDATE...")
            database_result = update_snomed_database(snomed_id, updated_concept_name, embedding_list)
            
            if database_result["success"]:
                print(f"✅ Database update successful!")
                print(f"Updated record: {database_result.get('updated_record', {})}")
            else:
                print(f"❌ Database update failed: {database_result['message']}")
                return {
                    "action": "error",
                    "snomed_id": snomed_id,
                    "entity": entity,
                    "error": f"Database update failed: {database_result['message']}",
                    "database_result": database_result
                }
        else:
            print(f"🔍 SIMULATION MODE - Database update would be executed here")
        
        # Step 7: Prepare the comprehensive result
        update_info = {
            "action": "update_executed" if not simulate_only else "update_simulated",
            "snomed_id": snomed_id,
            "entity_searched": entity,
            "original_concept_name": current_concept_name,
            "most_similar_term": most_similar_term,
            "updated_concept_name": updated_concept_name,
            "new_embeddings": embedding_list,
            "embedding_dimension": len(embedding_list),
            "database_result": database_result,
            "simulation_mode": simulate_only,
            "sql_query_executed": f"""
UPDATE snomed_ct_codes_tst 
SET 
    ConceptID_name = '{updated_concept_name}',
    embeddings = [vector of {len(embedding_list)} dimensions]
WHERE ConceptID = '{snomed_id}';
            """ if not simulate_only else "Not executed (simulation mode)"
        }
        
        print(f"Update {'executed' if not simulate_only else 'simulated'} for SNOMED ID: {snomed_id}")
        print(f"Original name: {current_concept_name}")
        print(f"Most similar term: {most_similar_term}")
        print(f"Updated name: {updated_concept_name}")
        print(f"Embedding vector length: {len(embedding_list)}")
        
        return update_info
        
    except Exception as e:
        print(f"Unexpected error in update_snomed_for_manual_lookup: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "action": "error",
            "snomed_id": snomed_id,
            "entity": entity,
            "error": f"Unexpected error: {str(e)}",
            "database_result": None
        }


def simulate_database_update(update_info: Dict[str, Any]) -> None:
    """
    Display the database update information in a readable format.
    
    Args:
        update_info: The update information dictionary from update_snomed_for_manual_lookup
    """
    
    if update_info["action"] == "no_update_needed":
        print("\n" + "="*80)
        print("DATABASE UPDATE - NO ACTION NEEDED")
        print("="*80)
        print(f"SNOMED ID: {update_info['snomed_id']}")
        print(f"Reason: {update_info['reason']}")
        print(f"Current concept name: {update_info['current_concept_name']}")
        print("="*80)
        return
    
    if update_info["action"] == "error":
        print("\n" + "="*80)
        print("DATABASE UPDATE - ERROR OCCURRED")
        print("="*80)
        print(f"SNOMED ID: {update_info['snomed_id']}")
        print(f"Entity searched: {update_info['entity']}")
        print(f"Error: {update_info['error']}")
        if update_info.get('database_result'):
            print(f"Database result: {update_info['database_result']}")
        print("="*80)
        return
    
    if update_info["action"] in ["update_executed", "update_simulated"]:
        is_simulation = update_info.get("simulation_mode", False)
        print("\n" + "="*80)
        print(f"DATABASE UPDATE - {'SIMULATION' if is_simulation else 'EXECUTED'}")
        print("="*80)
        print(f"SNOMED ID: {update_info['snomed_id']}")
        print(f"Entity searched: {update_info['entity_searched']}")
        print()
        print("CHANGES MADE:" if not is_simulation else "CHANGES TO BE MADE:")
        print("-" * 40)
        print(f"Original ConceptID_name:")
        print(f"  {update_info['original_concept_name']}")
        print()
        print(f"Most similar term found:")
        print(f"  {update_info.get('most_similar_term', 'N/A')}")
        print()
        print(f"Updated ConceptID_name:")
        print(f"  {update_info['updated_concept_name']}")
        print()
        print(f"New embeddings vector (first 10 values):")
        print(f"  {update_info['new_embeddings'][:10]}...")
        print(f"  (Total dimension: {update_info['embedding_dimension']})")
        
        # Display database result if available
        if not is_simulation and update_info.get('database_result'):
            db_result = update_info['database_result']
            print()
            print("DATABASE UPDATE RESULT:")
            print("-" * 40)
            print(f"Success: {db_result['success']}")
            print(f"Message: {db_result['message']}")
            print(f"Rows affected: {db_result.get('rows_affected', 'N/A')}")
            
            if db_result.get('updated_record'):
                updated_record = db_result['updated_record']
                print()
                print("UPDATED RECORD VERIFICATION:")
                print(f"  ConceptID: {updated_record['conceptid']}")
                print(f"  ConceptID_name: {updated_record['conceptid_name']}")
                print(f"  TopLevelHierarchy: {updated_record['toplevelhierarchy_name']}")
                print(f"  FHIR Resource: {updated_record['fhir_resource']}")
        
        print()
        print("SQL QUERY EXECUTED:" if not is_simulation else "SQL QUERY (NOT EXECUTED):")
        print("-" * 40)
        print(update_info.get('sql_query_executed', 'N/A'))
        print("="*80)


def process_manual_snomed_selection(snomed_id: str, snomed_name: str, entity: str, 
                                   execute_update: bool = True) -> Dict[str, Any]:
    """
    Main function to process a manual SNOMED selection and execute database updates.
    This is the entry point that would be called from the main application.
    
    Args:
        snomed_id: The SNOMED concept ID that was manually selected
        snomed_name: The concept name from the database
        entity: The original entity that was searched for
        execute_update: If True, actually update the database. If False, only simulate.
    
    Returns:
        Dictionary containing the processing results and update information
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING MANUAL SNOMED SELECTION")
    print(f"{'='*60}")
    print(f"Entity searched: {entity}")
    print(f"SNOMED ID selected: {snomed_id}")
    print(f"Current concept name: {snomed_name}")
    print(f"Execute update: {execute_update}")
    print(f"{'='*60}")
    
    # Process the update
    update_result = update_snomed_for_manual_lookup(
        snomed_id, entity, snomed_name, simulate_only=not execute_update
    )
    
    # Display the update information
    simulate_database_update(update_result)
    
    # Return comprehensive result
    result = {
        "processed_at": pd.Timestamp.now().isoformat(),
        "entity": entity,
        "snomed_id": snomed_id,
        "original_concept_name": snomed_name,
        "update_info": update_result,
        "success": update_result["action"] in ["update_executed", "update_simulated", "no_update_needed"],
        "database_updated": update_result["action"] == "update_executed" and 
                           update_result.get("database_result", {}).get("success", False)
    }
    
    return result
