"""
Database operations and data dictionary interactions.
"""

import psycopg2
import psycopg2.extras
from tqdm import tqdm
from config import DB_CONFIG
from utils.embedding import initialize_emb_model


def get_suggested_names_from_adarv_dict(columns_list):
    """
    Get suggested names from adarv_data_dict table using vector similarity.
    Returns a dictionary mapping column names to suggested names.
    """
    suggested_names = {}
    
    # Create connection config without table names
    connection_config = {
        'dbname': DB_CONFIG['dbname'],
        'user': DB_CONFIG['user'],
        'password': DB_CONFIG['password'],
        'host': DB_CONFIG['host'],
        'port': DB_CONFIG['port']
    }
    
    # Get table name from config
    adarv_dict_table = DB_CONFIG.get('adarv_dict_table', 'adarv_data_dict')
    
    # Establish fresh database connection
    try:
        conn = psycopg2.connect(**connection_config)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Initialize embedding model
        encoder = initialize_emb_model()[1]
        
        for col in tqdm(columns_list, desc="Getting suggested names from adarv_data_dict"):
            try:
                # Get embedding for the column name
                col_embedding = encoder.encode([col])[0]
                col_embedding_list = col_embedding.tolist()
                
                # SQL query to find similar columns using vector similarity
                similarity_query = f"""
                SELECT column_name, adarv_name, (embedding <=> %s) AS distance
                FROM {adarv_dict_table}
                ORDER BY distance
                LIMIT 1;
                """
                
                cur.execute(similarity_query, (col_embedding_list,))
                result = cur.fetchone()
                
                if result and result['distance'] < 0.5:  # Threshold for similarity
                    suggested_names[col] = result['adarv_name']
                    print(f"Found ADARV suggested name for '{col}': '{result['adarv_name']}' (distance: {result['distance']:.3f})")
                else:
                    print(f"No good match found for '{col}' in ADARV dict")
                    
            except Exception as e:
                print(f"Error processing column '{col}' for ADARV suggestions: {e}")
                continue
        
        # Close connection
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Database connection error in get_suggested_names_from_adarv_dict: {e}")
        return {}
    
    return suggested_names
