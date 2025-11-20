import numpy as np
from functools import cache

@cache
def get_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    return model

def get_embedding(content):
    model = get_model()
    return model.encode(content)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def eval(dataset, response):
    ground_truth = get_embedding(dataset['answer'].split('####')[-1].strip())
    model_response = get_embedding(response['text'])
    
    return float(cosine_similarity(ground_truth, model_response))
