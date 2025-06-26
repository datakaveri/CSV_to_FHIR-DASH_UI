"""
Embedding model initialization and text processing utilities.
"""

import string
import psycopg2
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer


def initialize_emb_model():
    """Initialize database connection and embedding model."""
    conn_params = {
        'dbname': 'postgresdb',
        'user': 'postgres',
        'password': 'timescaledbpg',
        'host': '65.0.127.208',  
        'port': '32588' 
    }
    # conn_params = {
    #     'dbname': 'postgresdb',
    #     'user': 'postgres',
    #     'password': 'timescaledbpg',
    #     'host': 'localhost',  
    #     'port': '15432' 
    # }

    conn = psycopg2.connect(**conn_params)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=DictCursor)

    encoder = SentenceTransformer("paraphrase-mpnet-base-v2") #paraphrase-mpnet-base-v2, HierarchyTransformer.from_pretrained("Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT"), SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scitail-mednli-stsb")
    return cur, encoder


def remove_punctuation(input_string):
    """Remove punctuation from input string."""
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)
