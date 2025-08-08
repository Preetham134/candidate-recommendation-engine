from sentence_transformers import SentenceTransformer

def get_embedding_model():
    return SentenceTransformer('intfloat/e5-large-v2')
