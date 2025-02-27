from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentrev.evaluator import evaluate_dense_retrieval
import os

# Load all the dense embedding models
encoder1 = SentenceTransformer('nvidia/embedding-v2')
encoder2 = SentenceTransformer('Alibaba-NLP/gte-Qwen2-7B-instruct')

# Create a list of the dense encoders
encoders = [encoder1, encoder2]

# Create a dictionary that maps each encoder to its name
encoder_to_names = {
    encoder1: 'nvidia-embed-v2',
    encoder2: 'gte-7b',
}

# Collect data
pdfs = [
    "template_3981_Mutual NDA .txt",

]

# Create Qdrant client
client = QdrantClient(
    url="https://41696e48-ee43-4811-901f-cc66429757dd.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.QG0B8gDPhf6LyLT-H1lQCVlb4CLv3YOJj2GFAiMpGlE",
)
# Set distances
distances = ["cosine", "dot", "euclid", "manhattan"]

# Loop through different chunking_size, text_percentage and distance options
for chunking_size in range(500, 2000, 500):
    for text_percentage in range(40, 100, 20):
        perc = text_percentage / 100
        for distance in distances:
            eval_dir = f"dense_eval/{chunking_size}_{text_percentage}_{distance}/"
            os.makedirs(eval_dir, exist_ok=True)
            csv_path = os.path.join(eval_dir, "stats.csv")
            evaluate_dense_retrieval(
                pdfs, encoders, encoder_to_names, client, csv_path,
                chunking_size, text_percentage=perc, distance=distance,
                mrr=10, carbon_tracking="AUT", plot=True
            )
