#!/usr/bin/env python3
"""
Command Line Interface for RAG Assistant
Provides CLI access to all RAG functionality
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional
import readline  # For better input handling
from loguru import logger

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from rag_engine import RAGEngine
from utils import setup_logging, format_sources, save_chat_history, create_system_info
import config

class RAGCLIApp:
    """Command Line Interface for RAG Assistant"""
    
    def __init__(self):
        self.rag_engine = None
        self.chat_history = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for CLI"""
        setup_logging()
        # Reduce log verbosity for CLI unless debug mode
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    
    def initialize_system(self) -> bool:
        """Initialize the RAG system"""
        print("🚀 Initializing RAG system...")
        
        try:
            self.rag_engine = RAGEngine()
            
            if self.rag_engine.initialize_system():
                print("✅ RAG system initialized successfully!")
                return True
            else:
                print("❌ Failed to initialize RAG system")
                return False
                
        except Exception as e:
            print(f"❌ Error initializing RAG system: {str(e)}")
            return False
    
    def add_documents(self, paths: List[str], recursive: bool = False):
        """Add documents from paths"""
        if not self.rag_engine:
            print("❌ RAG system not initialized")
            return
        
        total_docs = 0
        
        for path in paths:
            path_obj = Path(path)
            
            if path_obj.is_file():
                print(f"📄 Processing file: {path}")
                try:
                    self.rag_engine.process_and_add_file(path)
                    print(f"✅ Added file: {path_obj.name}")
                    total_docs += 1
                except Exception as e:
                    print(f"❌ Error processing {path}: {str(e)}")
            
            elif path_obj.is_dir():
                print(f"📁 Processing directory: {path}")
                try:
                    self.rag_engine.process_and_add_directory(path)
                    print(f"✅ Processed directory: {path}")
                    total_docs += 1
                except Exception as e:
                    print(f"❌ Error processing directory {path}: {str(e)}")
            
            else:
                print(f"⚠️ Path not found: {path}")
        
        if total_docs > 0:
            # Show updated stats
            stats = self.rag_engine.get_index_stats()
            print(f"\n📊 Index updated - Document count: {stats.get('document_count', 'Unknown')}")
    
    def query_single(self, question: str) -> str:
        """Process a single query"""
        if not self.rag_engine or not self.rag_engine.query_engine:
            return "❌ RAG system not initialized or no documents loaded"
        
        try:
            response = self.rag_engine.query(question)
            
            # Get sources
            relevant_docs = self.rag_engine.get_relevant_documents(question, top_k=3)
            
            # Format response
            result = f"🤖 Answer: {response}\n"
            
            if relevant_docs:
                result += f"\n📚 Sources:\n"
                for i, doc in enumerate(relevant_docs, 1):
                    metadata = doc.get("metadata", {})
                    file_name = metadata.get("file_name", "Unknown")
                    score = doc.get("score", 0.0)
                    result += f"  {i}. {file_name} (similarity: {score:.3f})\n"
            
            # Add to chat history
            self.chat_history.append({
                "question": question,
                "response": response,
                "sources": relevant_docs
            })
            
            return result
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"
    
    def interactive_chat(self):
        """Start interactive chat session"""
        if not self.rag_engine or not self.rag_engine.query_engine:
            print("❌ RAG system not initialized or no documents loaded")
            return
        
        print("\n💬 Interactive Chat Mode")
        print("Type 'quit', 'exit', or 'q' to end the session")
        print("Type '/help' for available commands")
        print("Type '/stats' for system statistics")
        print("Type '/save' to save chat history")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n🗣️ You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                # Handle special commands
                if question.startswith('/'):
                    self.handle_chat_command(question)
                    continue
                
                print("\n🤔 Processing...")
                response = self.query_single(question)
                print(f"\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Chat ended. Goodbye!")
                break
    
    def handle_chat_command(self, command: str):
        """Handle special chat commands"""
        if command == '/help':
            print("""
📋 Available Commands:
/help    - Show this help message
/stats   - Show system statistics
/save    - Save chat history to file
/clear   - Clear chat history
/info    - Show system information
quit/exit/q - Exit chat mode
""")
        
        elif command == '/stats':
            if self.rag_engine:
                stats = self.rag_engine.get_index_stats()
                print("\n📊 System Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            else:
                print("❌ RAG system not initialized")
        
        elif command == '/save':
            if self.chat_history:
                filename = save_chat_history(self.chat_history)
                if filename:
                    print(f"✅ Chat history saved to: {filename}")
                else:
                    print("❌ Failed to save chat history")
            else:
                print("⚠️ No chat history to save")
        
        elif command == '/clear':
            self.chat_history = []
            print("✅ Chat history cleared")
        
        elif command == '/info':
            info = create_system_info()
            print("\n🔧 System Information:")
            print(json.dumps(info, indent=2))
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Type '/help' for available commands")
    
    def batch_query(self, questions_file: str, output_file: Optional[str] = None):
        """Process batch queries from file"""
        if not self.rag_engine or not self.rag_engine.query_engine:
            print("❌ RAG system not initialized or no documents loaded")
            return
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f.readlines() if line.strip()]
            
            results = []
            
            print(f"📝 Processing {len(questions)} questions from {questions_file}")
            
            for i, question in enumerate(questions, 1):
                print(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                try:
                    response = self.rag_engine.query(question)
                    sources = self.rag_engine.get_relevant_documents(question, top_k=3)
                    
                    result = {
                        "question": question,
                        "response": response,
                        "sources": [
                            {
                                "file_name": src.get("metadata", {}).get("file_name", "Unknown"),
                                "score": src.get("score", 0.0)
                            } for src in sources
                        ]
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"❌ Error processing question {i}: {str(e)}")
                    results.append({
                        "question": question,
                        "response": f"Error: {str(e)}",
                        "sources": []
                    })
            
            # Save results
            output_path = output_file or f"batch_results_{len(questions)}_questions.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Batch processing complete. Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error in batch processing: {str(e)}")
    
    def show_status(self):
        """Show system status"""
        print("\n📊 RAG Assistant Status")
        print("-" * 30)
        
        if not self.rag_engine:
            print("❌ RAG system not initialized")
            return
        
        print(f"✅ RAG system initialized")
        print(f"✅ Model loaded: {self.rag_engine.model_setup is not None}")
        print(f"✅ Index created: {self.rag_engine.index is not None}")
        
        if self.rag_engine.index:
            stats = self.rag_engine.get_index_stats()
            print(f"\n📈 Index Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        print(f"\n💬 Chat history: {len(self.chat_history)} messages")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="RAG Assistant Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                           # Initialize system
  %(prog)s add ./data                     # Add documents from directory
  %(prog)s add file1.pdf file2.docx      # Add specific files
  %(prog)s query "What is machine learning?"  # Single query
  %(prog)s chat                          # Interactive chat
  %(prog)s batch questions.txt           # Batch processing
  %(prog)s status                        # Show system status
        """
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize RAG system')
    
    # Add documents command
    add_parser = subparsers.add_parser('add', help='Add documents to index')
    add_parser.add_argument('paths', nargs='+', help='File or directory paths')
    add_parser.add_argument('--recursive', '-r', action='store_true', help='Recursive directory processing')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Single query')
    query_parser.add_argument('question', help='Question to ask')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch query processing')
    batch_parser.add_argument('input_file', help='File containing questions (one per line)')
    batch_parser.add_argument('--output', '-o', help='Output file for results')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    elif args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    # Create CLI app
    app = RAGCLIApp()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'init':
            app.initialize_system()
        
        elif args.command == 'add':
            if not app.initialize_system():
                sys.exit(1)
            app.add_documents(args.paths, args.recursive)
        
        elif args.command == 'query':
            if not app.initialize_system():
                sys.exit(1)
            response = app.query_single(args.question)
            print(f"\n{response}")
        
        elif args.command == 'chat':
            if not app.initialize_system():
                sys.exit(1)
            app.interactive_chat()
        
        elif args.command == 'batch':
            if not app.initialize_system():
                sys.exit(1)
            app.batch_query(args.input_file, args.output)
        
        elif args.command == 'status':
            app.show_status()
    
    except KeyboardInterrupt:
        print("\n👋 Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()