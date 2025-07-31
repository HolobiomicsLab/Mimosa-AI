import numpy as np
import openai
from numpy.linalg import norm
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def cosine_similarity_from_embeddings(embedding1, embedding2):
    """
    Calculate cosine similarity between two embedding vectors.
    
    Args:
        embedding1 (list): First embedding vector
        embedding2 (list): Second embedding vector
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    # Convert to numpy arrays if they aren't already
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    
    return similarity

def get_openai_embedding(text, model="text-embedding-3-small"):
    """
    Get embedding vector for a text string using OpenAI's API.
    
    Args:
        text (str): Text to get embedding for
        model (str): OpenAI embedding model to use
        
    Returns:
        list: Embedding vector
    """
    client = openai.OpenAI()
    
    response = client.embeddings.create(
        input=text,
        model=model
    )
    
    # Extract the embedding vector from the response
    embedding = response.data[0].embedding
    
    return embedding

def calculate_cosine_similarity_openai(text1, text2, model="text-embedding-3-small"):
    """
    Calculate cosine similarity between two text strings using OpenAI embeddings.
    
    Args:
        text1 (str): First text string
        text2 (str): Second text string
        model (str): OpenAI embedding model to use
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    # Get embeddings for both texts
    embedding1 = get_openai_embedding(text1, model)
    embedding2 = get_openai_embedding(text2, model)
    
    # Calculate cosine similarity
    similarity = cosine_similarity_from_embeddings(embedding1, embedding2)
    
    return similarity

if __name__ == "__main__":
    # Define the two text strings
    text1 = "Cluster samples"
    text2 = "Reduce dimensions"
    
    # Calculate similarity using OpenAI embeddings
    similarity = calculate_cosine_similarity_openai(text1, text2)
    print(f"Cosine similarity (OpenAI embeddings) {text1} / {text2}: {similarity:.4f}")