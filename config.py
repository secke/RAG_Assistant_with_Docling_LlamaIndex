import os
from pathlib import Path

# Model Configuration
MODEL_NAME = "Akhenaton/sft_banking_model"
MODEL_FILE = "unsloth.Q4_K_M.gguf"  # Quantized model file
MODEL_PATH = "./models"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# llama.cpp Configuration
LLAMA_CPP_CONFIG = {
    "model_path": os.path.join(MODEL_PATH, MODEL_FILE),
    "n_ctx": 4096,
    "n_batch": 512,
    "n_threads": 8,
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "max_tokens": 2048,
    "verbose": False
}

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Vector Store Configuration
VECTOR_STORE_TYPE = "chroma"  # Options: chroma, faiss
PERSIST_DIR = "./vector_store"
COLLECTION_NAME = "document_collection"

# Document Processing Configuration
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx']
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 50

# Docling Configuration
DOCLING_CONFIG = {
    "do_ocr": True,  # Enable OCR for scanned PDFs
    "do_table_structure": True,
    "table_structure_options": {
        "do_cell_matching": True
    }
}

# Paths
DATA_DIR = "./data"
PROCESSED_DATA_DIR = "./processed_data"
LOGS_DIR = "./logs"

# Create directories if they don't exist
for dir_path in [MODEL_PATH, DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, PERSIST_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# RAG Configuration
RAG_CONFIG = {
    "similarity_top_k": 5,
    "response_mode": "compact",
    "streaming": True
}

# UI Configuration
GRADIO_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": False,
    "debug": True
}