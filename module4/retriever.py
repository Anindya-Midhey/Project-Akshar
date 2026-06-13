from module4.vector_store import collection
from module4.embedder import embed_text

def retrieve(query):
    q_embed = embed_text([query])[0]

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=1
    )

    return results