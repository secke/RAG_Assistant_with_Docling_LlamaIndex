#!/usr/bin/env python3
"""
Setup script for RAG Assistant application
Handles initial setup, directory creation, and dependency checking
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import importlib.util

def print_colored(message, color="white"):
    """Print colored messages"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m", 
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "white": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{message}{colors['white']}")

def check_python_version():
    """Check if Python version is supported"""
    print_colored("🐍 Checking Python version...", "blue")
    
    if sys.version_info < (3, 8):
        print_colored("❌ Python 3.8+ required. Current version: {}.{}".format(
            sys.version_info.major, sys.version_info.minor), "red")
        return False
    
    print_colored(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected", "green")
    return True

def check_system_requirements():
    """Check system requirements"""
    print_colored("💻 Checking system requirements...", "blue")
    
    # Check OS
    system = platform.system()
    print_colored(f"Operating System: {system}", "white")
    
    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage('.')
    free_gb = free / (1024**3)
    print_colored(f"Available disk space: {free_gb:.1f} GB", "white")
    
    if free_gb < 10:
        print_colored("⚠️ Warning: Less than 10GB free space available", "yellow")
    
    # Check for GPU (CUDA)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print_colored("✅ NVIDIA GPU detected", "green")
        else:
            print_colored("ℹ️ No NVIDIA GPU detected (CPU mode will be used)", "yellow")
    except FileNotFoundError:
        print_colored("ℹ️ nvidia-smi not found (CPU mode will be used)", "yellow")
    
    return True

def create_directories():
    """Create required directories"""
    print_colored("📁 Creating required directories...", "blue")
    
    directories = [
        "data",
        "models", 
        "vector_store",
        "processed_data",
        "logs"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_colored(f"✅ Created directory: {dir_name}", "green")
        else:
            print_colored(f"ℹ️ Directory already exists: {dir_name}", "white")

def install_dependencies():
    """Install Python dependencies"""
    print_colored("📦 Installing Python dependencies...", "blue")
    
    if not Path("requirements.txt").exists():
        print_colored("❌ requirements.txt not found!", "red")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print_colored("✅ Dependencies installed successfully", "green")
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install dependencies: {e}", "red")
        return False

def install_system_dependencies():
    """Install system dependencies for OCR"""
    print_colored("🔧 Checking system dependencies for OCR...", "blue")
    
    system = platform.system()
    
    if system == "Linux":
        print_colored("Installing tesseract for OCR support...", "white")
        try:
            # Check if tesseract is already installed
            subprocess.check_call(["which", "tesseract"], stdout=subprocess.DEVNULL)
            print_colored("✅ Tesseract already installed", "green")
        except subprocess.CalledProcessError:
            print_colored("ℹ️ Tesseract not found. Please install it manually:", "yellow")
            print_colored("   Ubuntu/Debian: sudo apt-get install tesseract-ocr", "white")
            print_colored("   CentOS/RHEL: sudo yum install tesseract", "white")
    
    elif system == "Darwin":  # macOS
        print_colored("ℹ️ For macOS, install tesseract using:", "yellow")
        print_colored("   brew install tesseract", "white")
    
    elif system == "Windows":
        print_colored("ℹ️ For Windows, download tesseract from:", "yellow")
        print_colored("   https://github.com/tesseract-ocr/tesseract", "white")

def check_configuration():
    """Check and validate configuration"""
    print_colored("⚙️ Checking configuration...", "blue")
    
    try:
        import config
        print_colored("✅ Configuration file loaded successfully", "green")
        
        # Check critical config values
        required_configs = [
            'MODEL_NAME',
            'EMBEDDING_MODEL', 
            'CHUNK_SIZE',
            'SUPPORTED_EXTENSIONS'
        ]
        
        missing_configs = []
        for config_name in required_configs:
            if not hasattr(config, config_name):
                missing_configs.append(config_name)
        
        if missing_configs:
            print_colored(f"⚠️ Missing configurations: {', '.join(missing_configs)}", "yellow")
        else:
            print_colored("✅ All required configurations present", "green")
            
        return True
        
    except ImportError as e:
        print_colored(f"❌ Failed to import config: {e}", "red")
        return False

def setup_huggingface_token():
    """Setup Hugging Face token if needed"""
    print_colored("🤗 Setting up Hugging Face access...", "blue")
    
    if os.getenv("HUGGINGFACE_TOKEN"):
        print_colored("✅ HUGGINGFACE_TOKEN environment variable found", "green")
        return True
    
    print_colored("ℹ️ No HUGGINGFACE_TOKEN found", "white")
    print_colored("If you need access to private models, set your token:", "white")
    print_colored("   export HUGGINGFACE_TOKEN='your_token_here'", "white")
    
    response = input("Do you have a Hugging Face token to set now? (y/n): ").lower()
    if response == 'y':
        token = input("Enter your Hugging Face token: ").strip()
        if token:
            os.environ["HUGGINGFACE_TOKEN"] = token
            print_colored("✅ Token set for this session", "green")
            print_colored("To make it permanent, add to your shell profile:", "white")
            print_colored(f"   export HUGGINGFACE_TOKEN='{token}'", "white")
    
    return True

def test_imports():
    """Test critical imports"""
    print_colored("🧪 Testing critical imports...", "blue")
    
    critical_imports = [
        "torch",
        "transformers", 
        "llama_index",
        "gradio",
        "docling",
        "chromadb"
    ]
    
    failed_imports = []
    
    for module_name in critical_imports:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                failed_imports.append(module_name)
            else:
                print_colored(f"  ✅ {module_name}", "green")
        except ImportError:
            failed_imports.append(module_name)
    
    if failed_imports:
        print_colored(f"❌ Failed imports: {', '.join(failed_imports)}", "red")
        print_colored("Please run: pip install -r requirements.txt", "yellow")
        return False
    
    print_colored("✅ All critical imports successful", "green")
    return True

def create_sample_data():
    """Create sample data directory with instructions"""
    print_colored("📄 Setting up sample data...", "blue")
    
    data_dir = Path("data")
    readme_file = data_dir / "README.txt"
    
    if not readme_file.exists():
        readme_content = """
# Sample Data Directory

Place your documents here for processing by the RAG system.

Supported formats:
- PDF files (.pdf) - including scanned PDFs with OCR
- Microsoft Word documents (.docx)
- Text files (.txt)
- Markdown files (.md)
- CSV files (.csv)
- Excel files (.xlsx)

Usage:
1. Copy your documents to this directory
2. Use the "Process Directory" option in the web interface
3. Or drag and drop files using the "Upload Files" option

Tips:
- Keep files under 50MB for optimal performance
- Use descriptive filenames
- Organize in subdirectories if needed

The RAG system will automatically:
- Extract text from all supported formats
- Perform OCR on scanned PDFs
- Extract tables and preserve structure
- Split content into optimized chunks
- Create vector embeddings for search
"""
        
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print_colored(f"✅ Created sample data instructions: {readme_file}", "green")

def main():
    """Main setup function"""
    print_colored("=" * 60, "blue")
    print_colored("🚀 RAG Assistant Setup Script", "blue")
    print_colored("=" * 60, "blue")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check system requirements
    check_system_requirements()
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Install system dependencies
    install_system_dependencies()
    
    # Step 5: Install Python dependencies
    if not install_dependencies():
        print_colored("❌ Setup failed during dependency installation", "red")
        sys.exit(1)
    
    # Step 6: Test imports
    if not test_imports():
        print_colored("❌ Setup failed during import testing", "red")
        sys.exit(1)
    
    # Step 7: Check configuration
    if not check_configuration():
        print_colored("❌ Setup failed during configuration check", "red")
        sys.exit(1)
    
    # Step 8: Setup Hugging Face token
    setup_huggingface_token()
    
    # Step 9: Create sample data instructions
    create_sample_data()
    
    # Success message
    print_colored("=" * 60, "green")
    print_colored("✅ Setup completed successfully!", "green")
    print_colored("=" * 60, "green")
    
    print_colored("\n🚀 Next steps:", "blue")
    print_colored("1. Place your documents in the 'data' directory", "white")
    print_colored("2. Run the application: python app.py", "white")
    print_colored("3. Open your browser to: http://localhost:7860", "white")
    print_colored("4. Initialize the system in the 'System Setup' tab", "white")
    print_colored("5. Add documents and start chatting!", "white")
    
    print_colored("\n📚 For more information, see README.md", "blue")

if __name__ == "__main__":
    main()