import os
from pinecone import Pinecone, ServerlessSpec
import streamlit as st 


PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

CLOUD = "aws"
REGION = "us-east-1"
INDEX_NAME = "resumes-index"

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_index(dimension):
    index_names = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in index_names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
    return pc.Index(INDEX_NAME)

def upsert_resume_embedding(index, embedding, metadata, filename):
    resume_id = f"resume_{filename.replace('.', '_').replace(' ', '_')}"
    index.upsert([(resume_id, embedding, metadata)])
    return resume_id

def query_similar_resumes(index, query_embedding, top_k):
    return index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )

def clear_all_resumes(index):
    try:
        index.delete(delete_all=True)
        import time
        time.sleep(2)
    except Exception as e:
        print(f"Error clearing index: {e}")

