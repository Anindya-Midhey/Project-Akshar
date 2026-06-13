from module4.embedder import embed_text
from module4.vector_store import store_chunks
from module4.retriever import retrieve


def prepare_index(module3_output, doc_id=0):
    chunks = []
    metadata = []

    for item in module3_output["visual_grounding"]:
        chunks.append(item["text"])

        metadata.append({
            "bbox": item["bbox"],
            "page": doc_id,
            "doc_id": doc_id
        })

    embeddings = embed_text(chunks)

    store_chunks(chunks, embeddings, metadata)


def answer_query(query):
    results = retrieve(query)

    doc = results["documents"][0][0]
    meta = results["metadatas"][0][0]

    return {
        "answer": doc,
        "bbox": meta["bbox"],
        "page": meta["page"]
    }