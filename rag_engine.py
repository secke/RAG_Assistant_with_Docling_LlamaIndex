from typing import List, Optional, Generator
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    Settings,
    get_response_synthesizer
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from document_processor import DocumentProcessor
from model_setup import ModelSetup
from loguru import logger
import config

class RAGEngine:
    """
    RAG Engine using LlamaIndex for orchestration
    """
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.model_setup = None
        self.vector_store = None
        self.index = None
        self.query_engine = None
        self.embedding_model = None
        
        # Setup LlamaIndex settings
        self.setup_llamaindex()
        
    def setup_llamaindex(self):
        """Setup LlamaIndex configuration"""
        try:
            # Setup embedding model
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedding_model = HuggingFaceEmbedding(
                model_name=config.EMBEDDING_MODEL
            )
            
            # Set global settings
            Settings.embed_model = self.embedding_model
            Settings.chunk_size = config.CHUNK_SIZE
            Settings.chunk_overlap = config.CHUNK_OVERLAP
            
            logger.info("LlamaIndex settings configured")
            
        except Exception as e:
            logger.error(f"Error setting up LlamaIndex: {str(e)}")
            raise
    
    def setup_model(self) -> bool:
        """Setup the LLM model"""
        try:
            # Initialize model setup
            self.model_setup = ModelSetup()
            
            # Download and load model
            if not self.model_setup.download_quantized_model():
                logger.error("Failed to download model")
                return False
            
            if not self.model_setup.load_model():
                logger.error("Failed to load model")
                return False
            
            # Create LlamaIndex LLM wrapper
            Settings.llm = LlamaCPP(
                model_path=config.LLAMA_CPP_CONFIG["model_path"],
                temperature=config.LLAMA_CPP_CONFIG["temperature"],
                max_new_tokens=config.LLAMA_CPP_CONFIG["max_tokens"],
                context_window=config.LLAMA_CPP_CONFIG["n_ctx"],
                generate_kwargs={
                    "top_p": config.LLAMA_CPP_CONFIG["top_p"],
                    "top_k": config.LLAMA_CPP_CONFIG["top_k"],
                    "repeat_penalty": config.LLAMA_CPP_CONFIG["repeat_penalty"]
                },
                verbose=config.LLAMA_CPP_CONFIG["verbose"]
            )
            
            logger.info("Model setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            return False
    
    def setup_vector_store(self):
        """Setup vector store (ChromaDB)"""
        try:
            logger.info("Setting up ChromaDB vector store")
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path=config.PERSIST_DIR)
            
            # Get or create collection
            chroma_collection = chroma_client.get_or_create_collection(
                name=config.COLLECTION_NAME
            )
            
            # Create ChromaDB vector store
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            logger.info("Vector store setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise
    
    def create_index(self, documents: List = None):
        """Create or load vector index"""
        try:
            if self.vector_store is None:
                self.setup_vector_store()
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            if documents:
                logger.info(f"Creating new index with {len(documents)} documents")
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True
                )
            else:
                logger.info("Loading existing index")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store
                )
            
            logger.info("Index created/loaded successfully")
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def setup_query_engine(self):
        """Setup the query engine with retrievers and post-processors"""
        try:
            if self.index is None:
                raise ValueError("Index not created. Please create index first.")
            
            # Configure retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=config.RAG_CONFIG["similarity_top_k"]
            )
            
            # Configure response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=config.RAG_CONFIG["response_mode"],
                streaming=config.RAG_CONFIG["streaming"]
            )
            
            # Configure post-processor for similarity filtering
            postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
            
            # Create query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[postprocessor]
            )
            
            logger.info("Query engine setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up query engine: {str(e)}")
            raise
    
    def add_documents(self, documents: List):
        """Add new documents to the index"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
            
            logger.info(f"Adding {len(documents)} documents to index")
            
            if self.index is None:
                # Create new index
                self.create_index(documents)
            else:
                # Add to existing index
                for doc in documents:
                    self.index.insert(doc)
            
            # Refresh query engine
            self.setup_query_engine()
            
            logger.info("Documents added successfully")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def process_and_add_directory(self, directory_path: str):
        """Process all files in directory and add to index"""
        try:
            logger.info(f"Processing directory: {directory_path}")
            documents = self.document_processor.process_directory(directory_path)
            
            if documents:
                self.add_documents(documents)
                logger.info(f"Successfully processed and added {len(documents)} documents")
            else:
                logger.warning("No documents were processed from the directory")
                
        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            raise
    
    def process_and_add_file(self, file_path: str):
        """Process single file and add to index"""
        try:
            logger.info(f"Processing file: {file_path}")
            documents = self.document_processor.process_file(file_path)
            
            if documents:
                self.add_documents(documents)
                logger.info(f"Successfully processed and added {len(documents)} document chunks")
            else:
                logger.warning("No documents were processed from the file")
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        try:
            if self.query_engine is None:
                return "RAG system not initialized. Please add documents first."
            
            logger.info(f"Processing query: {question}")
            response = self.query_engine.query(question)
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def query_streaming(self, question: str) -> Generator[str, None, None]:
        """Query the RAG system with streaming response"""
        try:
            if self.query_engine is None:
                yield "RAG system not initialized. Please add documents first."
                return
            
            logger.info(f"Processing streaming query: {question}")
            response = self.query_engine.query(question)
            
            # If streaming is enabled in the response synthesizer
            if hasattr(response, 'response_gen'):
                for chunk in response.response_gen:
                    yield chunk
            else:
                yield str(response)
                
        except Exception as e:
            logger.error(f"Error processing streaming query: {str(e)}")
            yield f"Error processing query: {str(e)}"
    
    def get_relevant_documents(self, question: str, top_k: int = 5) -> List[dict]:
        """Get relevant documents for a query without generating response"""
        try:
            if self.index is None:
                return []
            
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )
            
            nodes = retriever.retrieve(question)
            
            relevant_docs = []
            for node in nodes:
                relevant_docs.append({
                    "content": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                    "score": node.score if hasattr(node, 'score') else 0.0,
                    "metadata": node.metadata
                })
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {str(e)}")
            return []
    
    def get_index_stats(self) -> dict:
        """Get statistics about the index"""
        try:
            if self.index is None:
                return {"status": "No index created"}
            
            # Get basic stats
            stats = {
                "status": "Index loaded",
                "vector_store_type": config.VECTOR_STORE_TYPE,
                "embedding_model": config.EMBEDDING_MODEL,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP
            }
            
            # Try to get document count from vector store
            try:
                if hasattr(self.vector_store, '_collection'):
                    stats["document_count"] = self.vector_store._collection.count()
            except:
                stats["document_count"] = "Unknown"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"status": f"Error: {str(e)}"}
    
    def initialize_system(self) -> bool:
        """Initialize the complete RAG system"""
        try:
            logger.info("Initializing RAG system...")
            
            # Setup model
            if not self.setup_model():
                return False
            
            # Setup vector store
            self.setup_vector_store()
            
            # Try to load existing index
            try:
                self.create_index()
                self.setup_query_engine()
                logger.info("Loaded existing index")
            except:
                logger.info("No existing index found, will create when documents are added")
            
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            return False