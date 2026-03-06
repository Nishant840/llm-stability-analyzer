from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return embedding_model.encode(texts)