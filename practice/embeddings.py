from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Loads a small embedding model from HuggingFace that can be run locally
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)