# RAG Assistant with Docling & LlamaIndex

A comprehensive end-to-end RAG (Retrieval-Augmented Generation) application that processes various document formats including scanned PDFs with OCR capabilities.

## Features

### 🔧 Core Components
- **Orchestrator:** LlamaIndex for RAG pipeline management
- **Document Processing:** Docling with OCR support for scanned PDFs
- **LLM:** Custom Hugging Face model (`Akhenaton/sft_banking_model`) with quantized version
- **Serving:** llama.cpp for efficient model inference
- **Vector Store:** ChromaDB for embeddings storage
- **UI:** Gradio web interface

### 📄 Document Support
- **Supported formats:** PDF, DOCX, TXT, MD, CSV, XLSX
- **OCR capability:** Automatic text extraction from scanned PDFs
- **Table extraction:** Preserves table structure from documents
- **Large file handling:** Up to 50MB per file (configurable)

### 🚀 Key Features
- **Intelligent chunking:** Configurable chunk size and overlap
- **Semantic search:** Vector similarity search with configurable top-k
- **Source attribution:** Shows relevant document sources for each answer
- **Chat history:** Save and load conversation history
- **Real-time processing:** Streaming responses for better UX
- **System monitoring:** Comprehensive status and statistics

## Installation

### Prerequisites
- Python 3.8+
- CUDA GPU (optional, for faster processing)
- Minimum 16GB RAM recommended
- 20GB+ free disk space

### 1. Clone and Setup Environment
```bash
git clone <your-repo>
cd rag-assistant
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Hugging Face Token (Optional)
For private model access, set your Hugging Face token:
```bash
export HUGGINGFACE_TOKEN="your_hf_token_here"
```

### 4. Create Required Directories
```bash
mkdir -p data models vector_store processed_data logs
```

## Quick Start

### 1. Launch the Application
```bash
python app.py
```

The Gradio interface will be available at `http://localhost:7860`

### 2. Initialize the System
1. Go to the **"System Setup"** tab
2. Click **"Initialize System"** button
3. Wait for model download and initialization

### 3. Add Documents
**Option A: Directory Processing**
1. Go to **"Document Management"** → **"Process Directory"**
2. Enter the path to your documents folder (default: `./data`)
3. Click **"Process Directory"**

**Option B: File Upload**
1. Go to **"Document Management"** → **"Upload Files"**
2. Upload your documents
3. Click **"Process Uploaded Files"**

### 4. Start Chatting
1. Go to **"Chat with Documents"** tab
2. Ask questions about your documents
3. View sources and save chat history

## Configuration

### Model Configuration
Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "Akhenaton/sft_banking_model"
MODEL_FILE = "model.q4_k_m.gguf"

# Processing settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 50

# RAG settings
RAG_CONFIG = {
    "similarity_top_k": 5,
    "response_mode": "compact",
    "streaming": True
}
```

### Hardware Optimization
For GPU acceleration:
```bash
pip install llama-cpp-python[cuda]  # For CUDA
pip install llama-cpp-python[metal]  # For Apple Silicon
```

## File Structure

```
rag-assistant/
├── app.py                 # Main Gradio application
├── rag_engine.py         # RAG orchestration with LlamaIndex
├── document_processor.py # Document processing with Docling
├── model_setup.py        # Model management with llama.cpp
├── utils.py              # Utility functions
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── data/                 # Input documents (create this)
├── models/               # Model files (auto-created)
├── vector_store/         # ChromaDB storage (auto-created)
├── processed_data/       # Processed documents cache (auto-created)
└── logs/                 # Application logs (auto-created)
```

## API Usage

For programmatic usage:

```python
from rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine()
rag.initialize_system()

# Add documents
rag.process_and_add_directory("./your_documents/")

# Query
response = rag.query("What is the main topic discussed?")
print(response)

# Get sources
sources = rag.get_relevant_documents("your question", top_k=3)
```

## Advanced Configuration

### Custom Model Setup
To use a different quantized model:

1. Update `config.py`:
```python
MODEL_NAME = "your-model-name"
MODEL_FILE = "your-model.gguf"
```

2. The model will be automatically downloaded on first run

### Document Processing Options
Configure Docling processing in `config.py`:
```python
DOCLING_CONFIG = {
    "do_ocr": True,  # Enable OCR for scanned PDFs
    "do_table_structure": True,  # Extract table structure
    "table_structure_options": {
        "do_cell_matching": True
    }
}
```

### Vector Store Options
Switch between ChromaDB and FAISS:
```python
VECTOR_STORE_TYPE = "chroma"  # or "faiss"
```

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Verify Hugging Face token if using private models
   - Ensure sufficient disk space

2. **OCR Not Working**
   - Install tesseract: `sudo apt-get install tesseract-ocr`
   - Verify Docling installation: `pip install docling[full]`

3. **Out of Memory**
   - Reduce `CHUNK_SIZE` in config
   - Decrease `n_ctx` in llama.cpp config
   - Process files in smaller batches

4. **Slow Performance**
   - Use GPU acceleration if available
   - Reduce `similarity_top_k` for faster retrieval
   - Consider using smaller embedding models

### Debug Mode
Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python app.py
```

### System Requirements Check
Use the built-in system check:
1. Go to **"System Setup"** tab
2. Click **"Check Requirements"**

## Performance Optimization

### For Production Use
1. **Use GPU acceleration** for faster inference
2. **Increase RAM** allocation for larger document sets
3. **Use SSD storage** for vector database
4. **Configure chunking** based on your document types
5. **Monitor resource usage** via system status

### Scaling Considerations
- **Document volume:** ChromaDB handles millions of documents
- **Concurrent users:** Consider load balancing for multiple users
- **Memory usage:** Scales with document count and embedding dimensions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Docling](https://github.com/DS4SD/docling) for document processing
- [LlamaIndex](https://github.com/run-llama/llama_index) for RAG orchestration
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient LLM inference
- [Gradio](https://github.com/gradio-app/gradio) for the web interface
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review system requirements
3. Check application logs in `./logs/`
4. Open an issue on GitHub

---

**Built with ❤️ using open-source AI tools**