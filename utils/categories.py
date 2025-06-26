"""
Category assignment utilities.
"""


def assign_categories(columns_list, adarv_mappings, similarity_threshold=0.75):
    """
    Assign categories to columns based on ADARV mappings from API response.
    If category is blank in API response, leave it blank for user to edit.
    
    Args:
        columns_list: List of column names
        adarv_mappings: ADARV mappings from API response containing category info
        similarity_threshold: Not used anymore since API provides categories
        
    Returns:
        Dictionary mapping column names to categories
    """
    category_map = {}
    
    # Use categories directly from API response
    for item in adarv_mappings:
        if isinstance(item, dict) and 'Original_Column_Name' in item:
            col_name = item['Original_Column_Name']
            # Use category from API response - if blank, leave blank for user to edit
            category_map[col_name] = item.get('category', "")
    
    # For any remaining columns not in ADARV mappings, set empty category
    for col in columns_list:
        if col not in category_map:
            category_map[col] = ""
    
    return category_map
