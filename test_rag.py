#!/usr/bin/env python3
"""
Test script for RAG Assistant
Tests core functionality without UI
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from loguru import logger

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from rag_engine import RAGEngine
from document_processor import DocumentProcessor
from model_setup import ModelSetup
from utils import setup_logging, check_system_requirements

def create_test_documents():
    """Create test documents for validation"""
    test_docs = []
    
    # Create a simple text document
    text_content = """
    This is a test document for the RAG system.
    It contains information about artificial intelligence and machine learning.
    
    Machine Learning:
    Machine learning is a subset of artificial intelligence that focuses on 
    algorithms that can learn from and make predictions or decisions based on data.
    
    Deep Learning:
    Deep learning is a subset of machine learning that uses neural networks 
    with multiple layers to model and understand complex patterns in data.
    
    Natural Language Processing:
    NLP is a field of AI that focuses on the interaction between computers 
    and humans through natural language.
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text_content)
        test_docs.append(f.name)
    
    # Create a simple markdown document
    md_content = """
# RAG System Documentation

## Overview
This document explains the RAG (Retrieval-Augmented Generation) system.

## Components
- **Document Processor**: Handles various file formats
- **Vector Store**: Stores document embeddings
- **LLM**: Generates responses based on retrieved context

## Features
- OCR support for scanned documents
- Multiple file format support
- Semantic search capabilities
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(md_content)
        test_docs.append(f.name)
    
    return test_docs

def test_system_requirements():
    """Test system requirements"""
    logger.info("Testing system requirements...")
    
    try:
        requirements = check_system_requirements()
        
        if requirements.get("errors"):
            logger.error(f"System requirement errors: {requirements['errors']}")
            return False
        
        if requirements.get("warnings"):
            logger.warning(f"System warnings: {requirements['warnings']}")
        
        logger.info("✅ System requirements check passed")
        return True
        
    except Exception as e:
        logger.error(f"System requirements test failed: {str(e)}")
        return False

def test_document_processor():
    """Test document processing functionality"""
    logger.info("Testing document processor...")
    
    try:
        processor = DocumentProcessor()
        test_docs = create_test_documents()
        
        all_documents = []
        
        for doc_path in test_docs:
            logger.info(f"Processing test document: {doc_path}")
            documents = processor.process_file(doc_path)
            
            if not documents:
                logger.error(f"No documents extracted from {doc_path}")
                return False
            
            all_documents.extend(documents)
            logger.info(f"✅ Processed {len(documents)} chunks from {Path(doc_path).name}")
        
        logger.info(f"✅ Document processor test passed - {len(all_documents)} total chunks")
        
        # Cleanup
        for doc_path in test_docs:
            try:
                os.unlink(doc_path)
            except:
                pass
        
        return True, all_documents
        
    except Exception as e:
        logger.error(f"Document processor test failed: {str(e)}")
        return False, []

def test_model_setup():
    """Test model setup and loading"""
    logger.info("Testing model setup...")
    
    try:
        model_setup = ModelSetup()
        
        # Test model download
        logger.info("Testing model download...")
        if not model_setup.download_quantized_model():
            logger.error("Model download failed")
            return False
        
        # Test model loading
        logger.info("Testing model loading...")
        if not model_setup.load_model():
            logger.error("Model loading failed")
            return False
        
        # Test model inference
        logger.info("Testing model inference...")
        if not model_setup.test_model():
            logger.warning("Model test failed, but model loaded")
        
        logger.info("✅ Model setup test passed")
        return True, model_setup
        
    except Exception as e:
        logger.error(f"Model setup test failed: {str(e)}")
        return False, None

def test_rag_engine():
    """Test complete RAG engine"""
    logger.info("Testing RAG engine...")
    
    try:
        rag = RAGEngine()
        
        # Initialize system
        logger.info("Initializing RAG system...")
        if not rag.initialize_system():
            logger.error("RAG system initialization failed")
            return False
        
        # Test with sample documents
        test_docs = create_test_documents()
        
        # Process and add documents
        logger.info("Adding test documents...")
        for doc_path in test_docs:
            rag.process_and_add_file(doc_path)
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "Explain deep learning",
            "What are the components of the RAG system?",
            "What file formats are supported?"
        ]
        
        logger.info("Testing queries...")
        for query in test_queries:
            logger.info(f"Query: {query}")
            response = rag.query(query)
            
            if not response or "error" in response.lower():
                logger.warning(f"Query failed or returned error: {query}")
            else:
                logger.info(f"✅ Query successful: {response[:100]}...")
        
        # Test retrieval
        logger.info("Testing document retrieval...")
        relevant_docs = rag.get_relevant_documents("machine learning", top_k=3)
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
        else:
            logger.info(f"✅ Retrieved {len(relevant_docs)} relevant documents")
        
        # Get system stats
        stats = rag.get_index_stats()
        logger.info(f"Index stats: {stats}")
        
        logger.info("✅ RAG engine test passed")
        
        # Cleanup
        for doc_path in test_docs:
            try:
                os.unlink(doc_path)
            except:
                pass
        
        return True
        
    except Exception as e:
        logger.error(f"RAG engine test failed: {str(e)}")
        return False

def test_end_to_end():
    """Run end-to-end test"""
    logger.info("Starting end-to-end RAG system test...")
    
    results = {
        "system_requirements": False,
        "document_processor": False,
        "model_setup": False,
        "rag_engine": False,
        "overall": False
    }
    
    try:
        # Test 1: System requirements
        results["system_requirements"] = test_system_requirements()
        
        # Test 2: Document processor
        doc_test_result, test_documents = test_document_processor()
        results["document_processor"] = doc_test_result
        
        # Test 3: Model setup (optional - can be skipped if model not available)
        model_test_result, model_setup = test_model_setup()
        results["model_setup"] = model_test_result
        
        # Test 4: RAG engine (only if model setup succeeded)
        if model_test_result:
            results["rag_engine"] = test_rag_engine()
        else:
            logger.warning("Skipping RAG engine test due to model setup failure")
        
        # Overall result
        results["overall"] = all([
            results["system_requirements"],
            results["document_processor"],
            results["model_setup"],
            results["rag_engine"]
        ])
        
        # Print results
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("="*50)
        
        if results["overall"]:
            print("🎉 ALL TESTS PASSED! RAG system is working correctly.")
        else:
            print("⚠️ Some tests failed. Check logs for details.")
            
            # Provide troubleshooting hints
            print("\nTroubleshooting hints:")
            if not results["system_requirements"]:
                print("- Check system dependencies and disk space")
            if not results["document_processor"]:
                print("- Verify Docling installation: pip install docling")
            if not results["model_setup"]:
                print("- Check model download and HUGGINGFACE_TOKEN")
                print("- Verify llama-cpp-python installation")
            if not results["rag_engine"]:
                print("- Check vector store setup (ChromaDB)")
                print("- Verify embedding model download")
        
        return results["overall"]
        
    except Exception as e:
        logger.error(f"End-to-end test failed: {str(e)}")
        print(f"❌ Test suite failed with error: {str(e)}")
        return False

def main():
    """Main test function"""
    # Setup logging
    setup_logging()
    logger.info("Starting RAG Assistant test suite...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test RAG Assistant functionality")
    parser.add_argument("--component", choices=[
        "requirements", "processor", "model", "rag", "all"
    ], default="all", help="Component to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    try:
        if args.component == "requirements":
            success = test_system_requirements()
        elif args.component == "processor":
            success, _ = test_document_processor()
        elif args.component == "model":
            success, _ = test_model_setup()
        elif args.component == "rag":
            success = test_rag_engine()
        else:  # all
            success = test_end_to_end()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()