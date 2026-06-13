from sentence_transformers import SentenceTransformer

# BAAI/bge-base-en-v1.5 — 768-dim, significantly stronger than MiniLM-L6-v2 (384-dim).
# normalize_embeddings=True is required for BGE models to produce unit-norm vectors
# suitable for cosine similarity search.
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

def embed_text(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()