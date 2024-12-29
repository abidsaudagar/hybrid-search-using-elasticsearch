from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from index_mapping import mappings
import pandas as pd

# Initialize Elasticsearch
try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "kBeV=iaa=5-grDa8s7BU"),
        ca_certs="/Users/abid/Documents/Personal_Projects/my_databases/elasticsearch-8.17.0/config/certs/http_ca.crt",
    )
except Exception as e:
    raise Exception(
        status_code=500, detail=f"Failed to connect to Elasticsearch: {str(e)}"
    )

# ========================= INDEXING =========================

# Initialize S-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to get embeddings using S-BERT
def get_embedding(text):
    try:
        text = text.replace("\n", " ")
        return model.encode(text).tolist()
    except Exception as e:
        print(f"Error fetching embedding for text: '{text[:50]}...'. Error: {str(e)}")
        return None

index_name = "hybrid_search_v1"

# Create Elasticsearch index with mappings
def create_index():
    try:
        # Delete index if it exists
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
        mapping = mappings
        es.indices.create(index=index_name, mappings=mapping)
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Error creating index '{index_name}': {str(e)}")

# Load data
data = pd.read_csv("ecommerce_data.csv")

# Convert data to Elasticsearch format
actions = [
    {
        "_index": index_name,
        "_id": row["ID"],
        "_source": {
            "product_name": row["Product Name"],
            "description": row["Description"],
            "price": row["Price"],
            "tags": row["Tags"],
            "description_vector": get_embedding(row["Description"]),
        },
    }
    for _, row in data.iterrows()
]

print(actions)

# Bulk index data
helpers.bulk(es, actions)
print("Data indexed successfully!")


# ========================= SEARCH =========================

### Lexical Search Function
def lexical_search(query: str, top_k: int):
    lexical_results = es.search(
        index=index_name,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["product_name", "description"],
                }
            },
            "size": top_k,
        },
        source_excludes=["description_vector"]
    )

    lexical_hits = lexical_results["hits"]["hits"]
    max_bm25_score = max([hit["_score"] for hit in lexical_hits], default=1.0)

    # Normalize lexical scores
    for hit in lexical_hits:
        hit["_normalized_score"] = hit["_score"] / max_bm25_score

    return lexical_hits


### Semantic Search Function
def semantic_search(query: str, top_k: int):
    # Generate embeddings for the query using S-BERT
    query_embedding = get_embedding(query)

    # Perform a cosine similarity search using the query embedding
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_embedding, 'description_vector') + 1.0",
                "params": {"query_embedding": query_embedding}
            }
        }
    }

    semantic_results = es.search(
        index=index_name,
        body={
            "query": script_query,
            "_source": {
                "excludes": ["content_vector"]
            },
            "size": top_k
        },
        source_excludes=["description_vector"]
    )

    semantic_hits = semantic_results['hits']['hits']
    max_semantic_score = max([hit['_score'] for hit in semantic_hits], default=1.0)

    # Normalize semantic scores
    for hit in semantic_hits:
        hit['_normalized_score'] = hit['_score'] / max_semantic_score

    return semantic_hits

# Combine lexical and semantic search results using Reciprocal Rank Fusion (RRF).
def reciprocal_rank_fusion(lexical_hits, semantic_hits, k=60):
    """
    k: The rank bias parameter (higher values reduce the impact of rank).
    """
    rrf_scores = {}

    # Process lexical results
    for rank, hit in enumerate(lexical_hits, start=1):
        doc_id = hit['_id']
        score = 1 / (k + rank)  # RRF
        if doc_id in rrf_scores:
            rrf_scores[doc_id]['rrf_score'] += score
        else:
            rrf_scores[doc_id] = {
                'product_name': hit['_source']['product_name'],
                'description': hit['_source']['description'],
                'lexical_score': hit['_normalized_score'],
                'semantic_score': 0,
                'rrf_score': score
            }

    # Process semantic results
    for rank, hit in enumerate(semantic_hits, start=1):
        doc_id = hit['_id']
        score = 1 / (k + rank)  # RRF formula
        if doc_id in rrf_scores:
            rrf_scores[doc_id]['rrf_score'] += score
            rrf_scores[doc_id]['semantic_score'] = hit['_normalized_score']
        else:
            rrf_scores[doc_id] = {
                'product_name': hit['_source']['product_name'],
                'description': hit['_source']['description'],
                'lexical_score': 0,
                'semantic_score': hit['_normalized_score'],
                'rrf_score': score
            }

    # Sort by the RRF score
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)

    return sorted_results

def remove_duplicates_and_rerank(lexical_hits, semantic_hits, rerank=False):
    combined_results = {}

    # Process lexical results
    for hit in lexical_hits:
        doc_id = hit['_id']
        combined_results[doc_id] = {
            'product_name': hit['_source']['product_name'],
            'description': hit['_source']['description'],
            '_normalized_score': hit['_normalized_score'],  # Store lexical normalized score
            'semantic_score': 0  # Default for semantic score
        }

    # Process semantic results, checking for duplicates
    for hit in semantic_hits:
        doc_id = hit['_id']
        if doc_id in combined_results:
            # If the document exists in both lexical and semantic results, combine the scores
            combined_results[doc_id]['semantic_score'] = hit['_normalized_score']
            combined_results[doc_id]['_normalized_score'] += hit['_normalized_score']  # Combine scores
        else:
            # If it's not a duplicate, add it with only the semantic score
            combined_results[doc_id] = {
                'product_name': hit['_source']['product_name'],
                'description': hit['_source']['description'],
                '_normalized_score': hit['_normalized_score'],  # Semantic normalized score
                'semantic_score': hit['_normalized_score']
            }

    # Convert the dictionary to a list of results
    results_list = list(combined_results.values())

    # Sort by combined _normalized_score if rerank is True
    if rerank:
        results_list = sorted(results_list, key=lambda x: x['_normalized_score'], reverse=True)

    return results_list

### Hybrid Search Function
def hybrid_search(query: str, lexical_top_k, semantic_top_k):
    # Get lexical and semantic results
    lexical_hits = lexical_search(query, lexical_top_k)
    semantic_hits = semantic_search(query, semantic_top_k)
    # Combine using RRF
    combined_results = reciprocal_rank_fusion(lexical_hits, semantic_hits, k=60)
    return combined_results


input_query = "Affordable wireless headphones with ANC"

lexical_search(input_query, 3)

semantic_search(input_query, 3)

hybrid_search(input_query, 3, 3)


print(es.count(index=index_name))