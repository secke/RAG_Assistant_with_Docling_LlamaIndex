import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import config

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory
    Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Configure loguru
    logger.add(
        os.path.join(config.LOGS_DIR, "rag_app_{time}.log"),
        rotation="10 MB",
        retention="10 days",
        level="INFO",
        format="{time} | {level} | {message}"
    )
    
    logger.info("Logging setup completed")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def get_directory_info(directory_path: str) -> Dict[str, Any]:
    """Get information about files in a directory"""
    directory = Path(directory_path)
    
    if not directory.exists():
        return {"error": f"Directory does not exist: {directory_path}"}
    
    info = {
        "path": str(directory),
        "exists": True,
        "total_files": 0,
        "supported_files": 0,
        "file_types": {},
        "total_size": 0,
        "files": []
    }
    
    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                info["total_files"] += 1
                file_size = file_path.stat().st_size
                info["total_size"] += file_size
                
                file_ext = file_path.suffix.lower()
                info["file_types"][file_ext] = info["file_types"].get(file_ext, 0) + 1
                
                if file_ext in config.SUPPORTED_EXTENSIONS:
                    info["supported_files"] += 1
                    info["files"].append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "extension": file_ext,
                        "size": file_size,
                        "size_formatted": format_file_size(file_size),
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
        
        info["total_size_formatted"] = format_file_size(info["total_size"])
        
    except Exception as e:
        logger.error(f"Error scanning directory {directory_path}: {str(e)}")
        info["error"] = str(e)
    
    return info

def validate_file_upload(file_path: str) -> Dict[str, Any]:
    """Validate uploaded file"""
    validation = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    try:
        file_path_obj = Path(file_path)
        
        # Check if file exists
        if not file_path_obj.exists():
            validation["errors"].append("File does not exist")
            return validation
        
        # Check file extension
        file_ext = file_path_obj.suffix.lower()
        if file_ext not in config.SUPPORTED_EXTENSIONS:
            validation["errors"].append(f"Unsupported file type: {file_ext}")
        
        # Check file size
        file_size = file_path_obj.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            validation["errors"].append(f"File too large: {file_size_mb:.1f} MB (max: {config.MAX_FILE_SIZE_MB} MB)")
        elif file_size_mb > config.MAX_FILE_SIZE_MB * 0.8:
            validation["warnings"].append(f"Large file: {file_size_mb:.1f} MB")
        
        # File info
        validation["info"] = {
            "name": file_path_obj.name,
            "extension": file_ext,
            "size": file_size,
            "size_formatted": format_file_size(file_size),
            "size_mb": file_size_mb
        }
        
        # Mark as valid if no errors
        validation["valid"] = len(validation["errors"]) == 0
        
    except Exception as e:
        validation["errors"].append(f"Error validating file: {str(e)}")
    
    return validation

def create_chat_prompt(question: str, context: str = "") -> str:
    """Create a formatted prompt for the chat model"""
    if context:
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"""Question: {question}

Answer:"""
    
    return prompt

def extract_answer_from_response(response: str) -> str:
    """Extract clean answer from model response"""
    # Remove common prefixes
    prefixes_to_remove = [
        "Answer:",
        "Response:",
        "Based on the context",
        "According to the information provided"
    ]
    
    cleaned_response = response.strip()
    
    for prefix in prefixes_to_remove:
        if cleaned_response.lower().startswith(prefix.lower()):
            cleaned_response = cleaned_response[len(prefix):].strip()
            if cleaned_response.startswith(":"):
                cleaned_response = cleaned_response[1:].strip()
    
    return cleaned_response

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format source documents for display"""
    if not sources:
        return "No sources found."
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        file_name = metadata.get("file_name", "Unknown")
        chunk_id = metadata.get("chunk_id", "")
        score = source.get("score", 0.0)
        
        formatted_sources.append(
            f"**Source {i}:** {file_name}"
            f"{f' (chunk {chunk_id})' if chunk_id != '' else ''}"
            f" (similarity: {score:.3f})"
        )
    
    return "\n".join(formatted_sources)

def save_chat_history(chat_history: List[Dict[str, str]], filename: Optional[str] = None):
    """Save chat history to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
    
    filepath = os.path.join(config.LOGS_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
        logger.info(f"Chat history saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        return None

def load_chat_history(filepath: str) -> List[Dict[str, str]]:
    """Load chat history from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_history = json.load(f)
        logger.info(f"Chat history loaded from {filepath}")
        return chat_history
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
        return []

def estimate_processing_time(file_size_mb: float) -> str:
    """Estimate processing time based on file size"""
    # Rough estimates based on file size and processing complexity
    if file_size_mb < 1:
        return "< 30 seconds"
    elif file_size_mb < 5:
        return "30 seconds - 2 minutes"
    elif file_size_mb < 20:
        return "2-10 minutes"
    else:
        return "10+ minutes"

def create_system_info() -> Dict[str, Any]:
    """Create system information summary"""
    return {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": config.MODEL_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "vector_store": config.VECTOR_STORE_TYPE,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "max_file_size_mb": config.MAX_FILE_SIZE_MB,
            "supported_extensions": config.SUPPORTED_EXTENSIONS
        },
        "directories": {
            "data_dir": config.DATA_DIR,
            "model_path": config.MODEL_PATH,
            "persist_dir": config.PERSIST_DIR,
            "logs_dir": config.LOGS_DIR
        }
    }

def cleanup_old_logs(days_to_keep: int = 7):
    """Clean up old log files"""
    try:
        logs_dir = Path(config.LOGS_DIR)
        if not logs_dir.exists():
            return
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in logs_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                logger.info(f"Deleted old log file: {log_file}")
                
    except Exception as e:
        logger.error(f"Error cleaning up old logs: {str(e)}")

def check_system_requirements() -> Dict[str, Any]:
    """Check if system meets requirements"""
    requirements = {
        "status": "checking",
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    try:
        import torch
        requirements["info"]["torch_available"] = True
        requirements["info"]["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            requirements["info"]["cuda_device_count"] = torch.cuda.device_count()
            requirements["info"]["cuda_device_name"] = torch.cuda.get_device_name()
        
    except ImportError:
        requirements["warnings"].append("PyTorch not available - CPU only mode")
    
    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage(config.MODEL_PATH)
    free_gb = free / (1024**3)
    
    requirements["info"]["free_disk_space_gb"] = free_gb
    
    if free_gb < 10:
        requirements["errors"].append(f"Low disk space: {free_gb:.1f} GB free")
    elif free_gb < 20:
        requirements["warnings"].append(f"Limited disk space: {free_gb:.1f} GB free")
    
    # Check if model directories exist
    for dir_path in [config.MODEL_PATH, config.DATA_DIR, config.PERSIST_DIR]:
        if not os.path.exists(dir_path):
            requirements["warnings"].append(f"Directory does not exist: {dir_path}")
    
    requirements["status"] = "completed"
    return requirements