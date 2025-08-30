import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger
import config

class DocumentProcessor:
    """
    Document processor using Docling for various file formats including scanned PDFs
    """
    
    def __init__(self):
        self.setup_docling()
        self.node_parser = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
    def setup_docling(self):
        """Setup Docling converter with OCR capabilities"""
        # Configure pipeline options for PDF processing
        pdf_options = PdfPipelineOptions(
            do_ocr=config.DOCLING_CONFIG["do_ocr"],
            do_table_structure=config.DOCLING_CONFIG["do_table_structure"],
            table_structure_options=config.DOCLING_CONFIG.get("table_structure_options", {})
        )
        
        # Create converter with PDF backend
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_options,
            }
        )
        
        logger.info("Docling converter initialized with OCR support")
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file extension is supported"""
        return Path(file_path).suffix.lower() in config.SUPPORTED_EXTENSIONS
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def process_with_docling(self, file_path: str) -> str:
        """Process document using Docling"""
        try:
            # Convert document
            result = self.converter.convert(file_path)
            
            # Extract text content
            text_content = ""
            for page in result.document.pages:
                for element in page.elements:
                    if hasattr(element, 'text') and element.text:
                        text_content += element.text + "\n"
            
            # Also extract tables if present
            if hasattr(result.document, 'tables') and result.document.tables:
                for table in result.document.tables:
                    if hasattr(table, 'data'):
                        text_content += f"\n[TABLE]\n{table.data}\n[/TABLE]\n"
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error processing {file_path} with Docling: {str(e)}")
            return self.fallback_processing(file_path)
    
    def fallback_processing(self, file_path: str) -> str:
        """Fallback text extraction for unsupported files"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_extension == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            else:
                logger.warning(f"No fallback processor for {file_extension}")
                return ""
                
        except Exception as e:
            logger.error(f"Fallback processing failed for {file_path}: {str(e)}")
            return ""
    
    def process_file(self, file_path: str) -> List[Document]:
        """Process a single file and return LlamaIndex Documents"""
        if not self.is_supported_file(file_path):
            logger.warning(f"Unsupported file type: {file_path}")
            return []
        
        if self.get_file_size_mb(file_path) > config.MAX_FILE_SIZE_MB:
            logger.warning(f"File too large: {file_path} ({self.get_file_size_mb(file_path):.2f} MB)")
            return []
        
        logger.info(f"Processing file: {file_path}")
        
        # Extract text content
        text_content = self.process_with_docling(file_path)
        
        if not text_content:
            logger.warning(f"No content extracted from {file_path}")
            return []
        
        # Create LlamaIndex document
        document = Document(
            text=text_content,
            metadata={
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_type": Path(file_path).suffix.lower(),
                "file_size": self.get_file_size_mb(file_path)
            }
        )
        
        # Split into chunks
        nodes = self.node_parser.get_nodes_from_documents([document])
        
        # Convert nodes back to documents for consistency
        documents = []
        for i, node in enumerate(nodes):
            doc = Document(
                text=node.text,
                metadata={
                    **document.metadata,
                    "chunk_id": i,
                    "total_chunks": len(nodes)
                }
            )
            documents.append(doc)
        
        logger.info(f"Successfully processed {file_path} into {len(documents)} chunks")
        return documents
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all supported files in a directory"""
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return documents
        
        # Find all supported files
        supported_files = []
        for ext in config.SUPPORTED_EXTENSIONS:
            supported_files.extend(directory.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
        
        for file_path in supported_files:
            try:
                file_documents = self.process_file(str(file_path))
                documents.extend(file_documents)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(documents)} document chunks from {len(supported_files)} files")
        return documents
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """Process an uploaded file (Gradio file upload)"""
        if uploaded_file is None:
            return []
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            # Copy uploaded file content
            with open(uploaded_file.name, 'rb') as src:
                tmp_file.write(src.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the temporary file
            documents = self.process_file(tmp_file_path)
            return documents
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass