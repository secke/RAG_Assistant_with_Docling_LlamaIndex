import os
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama
from loguru import logger
import config

class ModelSetup:
    """
    Setup and manage the quantized LLM model with llama.cpp
    """
    
    def __init__(self):
        self.model_path = config.LLAMA_CPP_CONFIG["model_path"]
        self.model = None
        
    def download_quantized_model(self) -> bool:
        """Download the quantized model from Hugging Face"""
        try:
            logger.info(f"Checking for quantized model: {config.MODEL_NAME}")
            
            # List all files in the repository
            repo_files = list_repo_files(
                repo_id=config.MODEL_NAME,
                token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None
            )
            
            # Look for GGUF files (quantized models)
            gguf_files = [f for f in repo_files if f.endswith('.gguf')]
            
            if not gguf_files:
                logger.error(f"No GGUF files found in {config.MODEL_NAME}")
                return False
            
            # Prefer the specified model file, or pick the first available
            model_file = None
            if config.MODEL_FILE in gguf_files:
                model_file = config.MODEL_FILE
            else:
                # Look for common quantization patterns
                for pattern in ['q4_k_m', 'q4_0', 'q5_k_m', 'q8_0']:
                    candidates = [f for f in gguf_files if pattern in f.lower()]
                    if candidates:
                        model_file = candidates[0]
                        break
                
                if not model_file:
                    model_file = gguf_files[0]  # Fallback to first GGUF file
            
            logger.info(f"Selected model file: {model_file}")
            
            # Download the model if it doesn't exist
            if not os.path.exists(self.model_path):
                logger.info("Downloading quantized model...")
                downloaded_path = hf_hub_download(
                    repo_id=config.MODEL_NAME,
                    filename=model_file,
                    local_dir=config.MODEL_PATH,
                    token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None
                )
                
                # Move to expected path if different
                if downloaded_path != self.model_path:
                    os.rename(downloaded_path, self.model_path)
                
                logger.info(f"Model downloaded to: {self.model_path}")
            else:
                logger.info(f"Model already exists at: {self.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """Load the model with llama.cpp"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info("Loading model with llama.cpp...")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=config.LLAMA_CPP_CONFIG["n_ctx"],
                n_batch=config.LLAMA_CPP_CONFIG["n_batch"],
                n_threads=config.LLAMA_CPP_CONFIG["n_threads"],
                verbose=config.LLAMA_CPP_CONFIG["verbose"]
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using the loaded model"""
        if self.model is None:
            return "Model not loaded. Please load the model first."
        
        try:
            # Merge default config with provided kwargs
            generation_config = {
                "temperature": config.LLAMA_CPP_CONFIG["temperature"],
                "top_p": config.LLAMA_CPP_CONFIG["top_p"],
                "top_k": config.LLAMA_CPP_CONFIG["top_k"],
                "repeat_penalty": config.LLAMA_CPP_CONFIG["repeat_penalty"],
                "max_tokens": config.LLAMA_CPP_CONFIG["max_tokens"],
                **kwargs
            }
            
            # Generate response
            response = self.model(
                prompt,
                **generation_config
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_streaming_response(self, prompt: str, **kwargs):
        """Generate streaming response using the loaded model"""
        if self.model is None:
            yield "Model not loaded. Please load the model first."
            return
        
        try:
            # Merge default config with provided kwargs
            generation_config = {
                "temperature": config.LLAMA_CPP_CONFIG["temperature"],
                "top_p": config.LLAMA_CPP_CONFIG["top_p"],
                "top_k": config.LLAMA_CPP_CONFIG["top_k"],
                "repeat_penalty": config.LLAMA_CPP_CONFIG["repeat_penalty"],
                "max_tokens": config.LLAMA_CPP_CONFIG["max_tokens"],
                "stream": True,
                **kwargs
            }
            
            # Generate streaming response
            for chunk in self.model(prompt, **generation_config):
                if 'choices' in chunk and chunk['choices']:
                    token = chunk['choices'][0].get('text', '')
                    if token:
                        yield token
                        
        except Exception as e:
            logger.error(f"Error generating streaming response: {str(e)}")
            yield f"Error generating response: {str(e)}"
    
    def test_model(self) -> bool:
        """Test the loaded model with a simple prompt"""
        try:
            test_prompt = "Hello, how are you?"
            response = self.generate_response(test_prompt, max_tokens=50)
            
            if response and len(response) > 0:
                logger.info(f"Model test successful. Response: {response[:100]}...")
                return True
            else:
                logger.error("Model test failed: empty response")
                return False
                
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        info = {
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "file_exists": os.path.exists(self.model_path),
            "file_size_mb": 0
        }
        
        if os.path.exists(self.model_path):
            info["file_size_mb"] = os.path.getsize(self.model_path) / (1024 * 1024)
        
        if self.model:
            try:
                # Get context size and other model parameters
                info.update({
                    "n_ctx": self.model.n_ctx(),
                    "n_vocab": self.model.n_vocab(),
                    "model_type": "GGUF (Quantized)"
                })
            except:
                pass
        
        return info

def setup_model() -> ModelSetup:
    """Setup and return the model instance"""
    model_setup = ModelSetup()
    
    # Download model if needed
    if not model_setup.download_quantized_model():
        logger.error("Failed to download model")
        return None
    
    # Load model
    if not model_setup.load_model():
        logger.error("Failed to load model")
        return None
    
    # Test model
    if not model_setup.test_model():
        logger.warning("Model test failed, but continuing...")
    
    return model_setup