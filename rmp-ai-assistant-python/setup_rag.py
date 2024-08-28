from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
import os
import json
from os import getenv
import requests
from retry import retry

pc = Pinecone(api_key=getenv("PINECONE_API_KEY"))

# Create a Pinecone index
# pc.create_index(
#     name="rag",
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
# )
# Hugging face setup
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = getenv("HF_TOKEN")

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

@retry(tries=3, delay=10)
def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()
# Load the review data
data = json.load(open("reviews.json"))

processed_data = []

# Create embeddings for each review
for review in data["reviews"]:
    response = query(review['review'])
    processed_data.append( {
        "values": response,
        "id": review["professor"],
        "metadata": {
            "review": review["review"],
            "subject": review["subject"],
            "stars": review["stars"]
        }
    })

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1"
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics'
print(index.describe_index_stats())