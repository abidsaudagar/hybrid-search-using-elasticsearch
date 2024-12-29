mappings = {
    "properties": {
        "product_name": {"type": "text", "analyzer": "standard"},
        "description": {"type": "text", "analyzer": "standard"},
        "price": {"type": "float"},
        "tags": {"type": "text", "analyzer": "standard"},
        "description_vector": {"type": "dense_vector", "dims": 384, "similarity": "l2_norm"},
    }
}
