import gradio as gr
import os
from pathlib import Path
from typing import List, Tuple
import json
from datetime import datetime

from rag_engine import RAGEngine
from utils import (
    setup_logging, 
    get_directory_info, 
    validate_file_upload,
    format_sources,
    save_chat_history,
    create_system_info,
    check_system_requirements,
    estimate_processing_time
)
from loguru import logger
import config

# Setup logging
setup_logging()

# Global RAG engine instance
rag_engine = None

# Chat history storage
chat_history = []

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_engine
    
    try:
        logger.info("Initializing RAG system...")
        rag_engine = RAGEngine()
        
        if rag_engine.initialize_system():
            return "✅ RAG system initialized successfully!", True
        else:
            return "❌ Failed to initialize RAG system. Check logs for details.", False
            
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return f"❌ Error initializing RAG system: {str(e)}", False

def process_directory(directory_path: str):
    """Process all files in a directory"""
    global rag_engine
    
    if not rag_engine:
        return "❌ RAG system not initialized", "", ""
    
    if not directory_path or not os.path.exists(directory_path):
        return "❌ Invalid directory path", "", ""
    
    try:
        # Get directory info first
        dir_info = get_directory_info(directory_path)
        info_text = f"""
📁 **Directory Information:**
- Path: {dir_info['path']}
- Total files: {dir_info['total_files']}
- Supported files: {dir_info['supported_files']}
- Total size: {dir_info.get('total_size_formatted', 'Unknown')}

📋 **File types found:**
"""
        for ext, count in dir_info.get('file_types', {}).items():
            supported = "✅" if ext in config.SUPPORTED_EXTENSIONS else "❌"
            info_text += f"- {ext}: {count} files {supported}\n"
        
        if dir_info['supported_files'] == 0:
            return "⚠️ No supported files found in directory", info_text, ""
        
        # Process directory
        logger.info(f"Processing directory: {directory_path}")
        rag_engine.process_and_add_directory(directory_path)
        
        # Get updated stats
        index_stats = rag_engine.get_index_stats()
        stats_text = f"""
📊 **Index Statistics:**
- Status: {index_stats.get('status', 'Unknown')}
- Document count: {index_stats.get('document_count', 'Unknown')}
- Vector store: {index_stats.get('vector_store_type', 'Unknown')}
- Embedding model: {index_stats.get('embedding_model', 'Unknown')}
"""
        
        success_msg = f"✅ Successfully processed {dir_info['supported_files']} files from directory!"
        
        return success_msg, info_text, stats_text
        
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        return f"❌ Error processing directory: {str(e)}", "", ""

def process_uploaded_files(files):
    """Process uploaded files"""
    global rag_engine
    
    if not rag_engine:
        return "❌ RAG system not initialized", ""
    
    if not files:
        return "❌ No files uploaded", ""
    
    try:
        processed_count = 0
        error_count = 0
        total_size = 0
        processing_info = []
        
        for file in files:
            try:
                # Validate file
                validation = validate_file_upload(file.name)
                
                if not validation["valid"]:
                    error_count += 1
                    processing_info.append(f"❌ {Path(file.name).name}: {', '.join(validation['errors'])}")
                    continue
                
                # Get file info
                file_info = validation["info"]
                total_size += file_info["size"]
                
                # Process file
                documents = rag_engine.document_processor.process_uploaded_file(file)
                
                if documents:
                    rag_engine.add_documents(documents)
                    processed_count += 1
                    processing_info.append(f"✅ {file_info['name']}: {len(documents)} chunks ({file_info['size_formatted']})")
                else:
                    error_count += 1
                    processing_info.append(f"⚠️ {file_info['name']}: No content extracted")
                    
            except Exception as e:
                error_count += 1
                processing_info.append(f"❌ {Path(file.name).name}: {str(e)}")
        
        # Create summary
        summary = f"""
📋 **Processing Summary:**
- Files processed: {processed_count}
- Errors: {error_count}
- Total size: {total_size / (1024*1024):.1f} MB

📄 **File Details:**
""" + "\n".join(processing_info)
        
        if processed_count > 0:
            # Get updated stats
            index_stats = rag_engine.get_index_stats()
            summary += f"""

📊 **Updated Index Statistics:**
- Document count: {index_stats.get('document_count', 'Unknown')}
- Status: {index_stats.get('status', 'Unknown')}
"""
        
        return summary, f"✅ Processed {processed_count} files successfully!"
        
    except Exception as e:
        logger.error(f"Error processing uploaded files: {str(e)}")
        return f"❌ Error processing files: {str(e)}", ""

def chat_with_rag(message: str, history: List[Tuple[str, str]]):
    """Chat with the RAG system"""
    global rag_engine, chat_history
    
    if not rag_engine or not rag_engine.query_engine:
        return history + [(message, "❌ RAG system not initialized or no documents loaded. Please add documents first.")]
    
    if not message.strip():
        return history + [(message, "Please enter a question.")]
    
    try:
        logger.info(f"Processing chat query: {message}")
        
        # Query the RAG system
        response = rag_engine.query(message)
        
        # Get relevant sources
        relevant_docs = rag_engine.get_relevant_documents(message, top_k=3)
        
        # Format response with sources
        if relevant_docs:
            sources_text = format_sources(relevant_docs)
            full_response = f"{response}\n\n---\n**Sources:**\n{sources_text}"
        else:
            full_response = f"{response}\n\n---\n*No relevant sources found.*"
        
        # Update chat history
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": message,
            "response": response,
            "sources": relevant_docs
        }
        chat_history.append(chat_entry)
        
        # Update gradio history
        new_history = history + [(message, full_response)]
        
        return new_history
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        error_response = f"❌ Error processing your question: {str(e)}"
        return history + [(message, error_response)]

def get_system_status():
    """Get current system status"""
    global rag_engine
    
    try:
        status_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rag_initialized": rag_engine is not None,
            "model_loaded": False,
            "index_created": False
        }
        
        if rag_engine:
            status_info["model_loaded"] = rag_engine.model_setup is not None
            status_info["index_created"] = rag_engine.index is not None
            
            # Get index stats
            index_stats = rag_engine.get_index_stats()
            status_info.update(index_stats)
            
            # Get model info
            if rag_engine.model_setup:
                model_info = rag_engine.model_setup.get_model_info()
                status_info["model_info"] = model_info
        
        # Format for display
        status_text = f"""
🕒 **Status as of:** {status_info['timestamp']}

🤖 **RAG System:**
- Initialized: {"✅" if status_info['rag_initialized'] else "❌"}
- Model loaded: {"✅" if status_info['model_loaded'] else "❌"}
- Index created: {"✅" if status_info['index_created'] else "❌"}

📊 **Index Information:**
- Status: {status_info.get('status', 'Unknown')}
- Document count: {status_info.get('document_count', 'Unknown')}
- Vector store: {status_info.get('vector_store_type', 'Unknown')}
- Embedding model: {status_info.get('embedding_model', 'Unknown')}

⚙️ **Configuration:**
- Chunk size: {status_info.get('chunk_size', config.CHUNK_SIZE)}
- Chunk overlap: {status_info.get('chunk_overlap', config.CHUNK_OVERLAP)}
- Max file size: {config.MAX_FILE_SIZE_MB} MB
"""
        
        if "model_info" in status_info:
            model_info = status_info["model_info"]
            status_text += f"""

🔧 **Model Information:**
- Model file exists: {"✅" if model_info.get('file_exists') else "❌"}
- Model size: {model_info.get('file_size_mb', 0):.1f} MB
- Context size: {model_info.get('n_ctx', 'Unknown')}
- Vocabulary size: {model_info.get('n_vocab', 'Unknown')}
"""
        
        return status_text
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return f"❌ Error getting system status: {str(e)}"

def save_current_chat_history():
    """Save current chat history"""
    global chat_history
    
    try:
        if not chat_history:
            return "No chat history to save."
        
        filepath = save_chat_history(chat_history)
        if filepath:
            return f"✅ Chat history saved to: {filepath}"
        else:
            return "❌ Failed to save chat history."
            
    except Exception as e:
        return f"❌ Error saving chat history: {str(e)}"

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="RAG Assistant with Docling & LlamaIndex", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 RAG Assistant
        **Powered by:** Docling + LlamaIndex + llama.cpp + Gradio
        
        This system can process various document types (PDF, DOCX, TXT, etc.) including scanned PDFs with OCR.
        """)
        
        with gr.Tabs():
            # System Setup Tab
            with gr.Tab("🔧 System Setup"):
                gr.Markdown("### Initialize RAG System")
                
                with gr.Row():
                    init_button = gr.Button("Initialize System", variant="primary")
                    system_status_button = gr.Button("Check Status")
                
                init_output = gr.Textbox(
                    label="Initialization Status",
                    lines=3,
                    interactive=False
                )
                
                system_status_output = gr.Textbox(
                    label="System Status",
                    lines=15,
                    interactive=False
                )
                
                # System requirements check
                gr.Markdown("### System Requirements")
                req_button = gr.Button("Check Requirements")
                req_output = gr.Textbox(
                    label="Requirements Check",
                    lines=8,
                    interactive=False
                )
                
                # Event handlers
                init_button.click(
                    fn=lambda: initialize_rag_system()[0],
                    outputs=init_output
                )
                
                system_status_button.click(
                    fn=get_system_status,
                    outputs=system_status_output
                )
                
                req_button.click(
                    fn=lambda: json.dumps(check_system_requirements(), indent=2),
                    outputs=req_output
                )
            
            # Document Management Tab
            with gr.Tab("📁 Document Management"):
                gr.Markdown("### Add Documents to Knowledge Base")
                
                with gr.Tabs():
                    # Directory processing
                    with gr.Tab("📂 Process Directory"):
                        directory_input = gr.Textbox(
                            label="Directory Path",
                            placeholder="Enter path to directory containing documents",
                            value=config.DATA_DIR
                        )
                        
                        process_dir_button = gr.Button("Process Directory", variant="primary")
                        
                        dir_status = gr.Textbox(
                            label="Processing Status",
                            lines=2,
                            interactive=False
                        )
                        
                        dir_info = gr.Textbox(
                            label="Directory Information",
                            lines=8,
                            interactive=False
                        )
                        
                        dir_stats = gr.Textbox(
                            label="Index Statistics",
                            lines=6,
                            interactive=False
                        )
                        
                        process_dir_button.click(
                            fn=process_directory,
                            inputs=directory_input,
                            outputs=[dir_status, dir_info, dir_stats]
                        )
                    
                    # File upload
                    with gr.Tab("📄 Upload Files"):
                        file_upload = gr.Files(
                            label="Upload Documents",
                            file_types=[".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"],
                            file_count="multiple"
                        )
                        
                        process_files_button = gr.Button("Process Uploaded Files", variant="primary")
                        
                        upload_status = gr.Textbox(
                            label="Processing Status",
                            lines=2,
                            interactive=False
                        )
                        
                        upload_details = gr.Textbox(
                            label="Processing Details",
                            lines=10,
                            interactive=False
                        )
                        
                        process_files_button.click(
                            fn=process_uploaded_files,
                            inputs=file_upload,
                            outputs=[upload_details, upload_status]
                        )
                
                # Supported formats info
                gr.Markdown(f"""
                ### ℹ️ Supported Formats
                **File types:** {', '.join(config.SUPPORTED_EXTENSIONS)}
                
                **Max file size:** {config.MAX_FILE_SIZE_MB} MB
                
                **Special features:**
                - 🔍 OCR for scanned PDFs
                - 📊 Table extraction
                - 📝 Document structure preservation
                """)
            
            # Chat Tab
            with gr.Tab("💬 Chat with Documents"):
                gr.Markdown("### Ask Questions About Your Documents")
                
                chatbot = gr.Chatbot(
                    label="RAG Assistant",
                    height=500,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about your documents...",
                        lines=2,
                        scale=4
                    )
                    submit_button = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_button = gr.Button("Clear Chat")
                    save_chat_button = gr.Button("Save Chat History")
                
                save_status = gr.Textbox(
                    label="Save Status",
                    lines=1,
                    interactive=False
                )
                
                # Event handlers for chat
                def submit_message(message, history):
                    return chat_with_rag(message, history), ""
                
                submit_button.click(
                    fn=submit_message,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input]
                )
                
                msg_input.submit(
                    fn=submit_message,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input]
                )
                
                clear_button.click(
                    fn=lambda: [],
                    outputs=chatbot
                )
                
                save_chat_button.click(
                    fn=save_current_chat_history,
                    outputs=save_status
                )
            
            # System Info Tab
            with gr.Tab("ℹ️ System Information"):
                gr.Markdown("### Configuration and System Details")
                
                config_info = gr.Textbox(
                    label="System Configuration",
                    value=json.dumps(create_system_info(), indent=2),
                    lines=20,
                    interactive=False
                )
                
                refresh_config_button = gr.Button("Refresh Configuration")
                
                refresh_config_button.click(
                    fn=lambda: json.dumps(create_system_info(), indent=2),
                    outputs=config_info
                )
        
        gr.Markdown("""
        ---
        ### 📚 Usage Tips:
        1. **Initialize the system** first using the System Setup tab
        2. **Add documents** via directory processing or file upload
        3. **Start chatting** with your documents in the Chat tab
        4. **Monitor system status** for performance insights
        
        ### 🔧 Technical Details:
        - **Orchestrator:** LlamaIndex
        - **Document Processing:** Docling (with OCR support)
        - **LLM:** Custom Hugging Face model (quantized)
        - **Serving:** llama.cpp
        - **Vector Store:** ChromaDB
        - **UI:** Gradio
        """)
    
    return demo

def main():
    """Main function to run the application"""
    try:
        logger.info("Starting RAG Assistant application")
        
        # Create Gradio interface
        demo = create_gradio_interface()
        
        # Launch the app
        demo.launch(
            server_name=config.GRADIO_CONFIG["server_name"],
            server_port=config.GRADIO_CONFIG["server_port"],
            share=config.GRADIO_CONFIG["share"],
            debug=config.GRADIO_CONFIG["debug"]
        )
        
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        raise

if __name__ == "__main__":
    main()