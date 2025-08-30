#!/bin/bash
# RAG Assistant Installation Script
# Automated setup for the RAG Assistant application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.8"
VENV_NAME="rag_env"
REQUIREMENTS_FILE="requirements.txt"

# Function to print colored output
print_color() {
    printf "${2:-$NC}%s${NC}\n" "$1"
}

print_success() {
    print_color "✅ $1" "$GREEN"
}

print_error() {
    print_color "❌ $1" "$RED"
}

print_warning() {
    print_color "⚠️ $1" "$YELLOW"
}

print_info() {
    print_color "ℹ️ $1" "$BLUE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare versions
version_compare() {
    printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1
}

# Function to check Python version
check_python_version() {
    print_info "Checking Python version..."
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        print_info "Please install Python 3.8+ and try again"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    if [ "$(version_compare "$PYTHON_VERSION" "$PYTHON_MIN_VERSION")" != "$PYTHON_MIN_VERSION" ]; then
        print_error "Python $PYTHON_MIN_VERSION+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION detected"
}

# Function to check system dependencies
check_system_dependencies() {
    print_info "Checking system dependencies..."
    
    # Check OS
    OS_NAME=$(uname -s)
    print_info "Operating System: $OS_NAME"
    
    # Check for build essentials
    if [[ "$OS_NAME" == "Linux" ]]; then
        if ! command_exists gcc; then
            print_warning "gcc not found. Some packages may fail to install."
            print_info "Install build essentials: sudo apt-get install build-essential"
        fi
        
        # Check for tesseract (OCR)
        if ! command_exists tesseract; then
            print_warning "Tesseract not found. OCR functionality will be limited."
            print_info "Install tesseract: sudo apt-get install tesseract-ocr"
        else
            print_success "Tesseract found"
        fi
        
    elif [[ "$OS_NAME" == "Darwin" ]]; then
        # macOS
        if ! command_exists brew; then
            print_warning "Homebrew not found. Some dependencies may need manual installation."
        fi
        
        if ! command_exists tesseract; then
            print_warning "Tesseract not found. Install with: brew install tesseract"
        else
            print_success "Tesseract found"
        fi
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        print_warning "Low disk space: ${AVAILABLE_SPACE}GB available"
        print_info "At least 10GB recommended for models and data"
    else
        print_success "Sufficient disk space: ${AVAILABLE_SPACE}GB available"
    fi
}

# Function to create virtual environment
create_virtual_environment() {
    print_info "Creating virtual environment: $VENV_NAME"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment already exists"
        read -p "Remove existing environment and create new one? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            print_info "Removed existing virtual environment"
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    python3 -m venv "$VENV_NAME"
    print_success "Virtual environment created: $VENV_NAME"
}

# Function to activate virtual environment
activate_virtual_environment() {
    print_info "Activating virtual environment..."
    
    if [ ! -f "$VENV_NAME/bin/activate" ]; then
        print_error "Virtual environment not found: $VENV_NAME"
        exit 1
    fi
    
    source "$VENV_NAME/bin/activate"
    print_success "Virtual environment activated"
}

# Function to install Python dependencies
install_python_dependencies() {
    print_info "Installing Python dependencies..."
    
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install requirements
    pip install -r "$REQUIREMENTS_FILE"
    
    print_success "Python dependencies installed"
}

# Function to create directories
create_directories() {
    print_info "Creating required directories..."
    
    directories=("data" "models" "vector_store" "processed_data" "logs")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
    
    # Create .gitkeep files to preserve empty directories
    for dir in "${directories[@]}"; do
        if [ ! -f "$dir/.gitkeep" ]; then
            touch "$dir/.gitkeep"
        fi
    done
}

# Function to setup environment file
setup_environment_file() {
    print_info "Setting up environment file..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp ".env.example" ".env"
            print_success "Created .env file from template"
            print_warning "Please edit .env file to configure your settings"
        else
            print_warning ".env.example not found. Creating basic .env file"
            cat > .env << EOF
# RAG Assistant Environment Configuration
HUGGINGFACE_TOKEN=
LOG_LEVEL=INFO
GRADIO_SERVER_PORT=7860
EOF
            print_success "Created basic .env file"
        fi
    else
        print_info ".env file already exists"
    fi
}

# Function to run setup script
run_setup_script() {
    print_info "Running setup script..."
    
    if [ -f "setup.py" ]; then
        python setup.py
        print_success "Setup script completed"
    else
        print_warning "setup.py not found, skipping setup script"
    fi
}

# Function to run tests
run_tests() {
    print_info "Running system tests..."
    
    if [ -f "test_rag.py" ]; then
        python test_rag.py --component requirements
        print_success "Basic tests completed"
    else
        print_warning "test_rag.py not found, skipping tests"
    fi
}

# Function to print final instructions
print_final_instructions() {
    print_success "Installation completed successfully!"
    
    echo
    print_info "Next steps:"
    echo "1. Edit .env file to configure your settings (especially HUGGINGFACE_TOKEN)"
    echo "2. Activate virtual environment: source $VENV_NAME/bin/activate"
    echo "3. Run the application: python app.py"
    echo "4. Open your browser to: http://localhost:7860"
    echo
    
    print_info "Alternative usage:"
    echo "• CLI mode: python cli.py --help"
    echo "• Run tests: python test_rag.py"
    echo "• Docker: docker-compose up"
    echo
    
    print_info "Documentation:"
    echo "• README.md - Full documentation"
    echo "• config.py - Configuration options"
    echo "• logs/ - Application logs"
    echo
}

# Function to handle errors
handle_error() {
    print_error "Installation failed at step: $1"
    print_info "Check the error messages above for details"
    print_info "You can re-run this script to continue from where it failed"
    exit 1
}

# Main installation function
main() {
    echo "======================================"
    print_info "RAG Assistant Installation Script"
    echo "======================================"
    echo
    
    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Parse command line arguments
    SKIP_TESTS=false
    SKIP_SETUP=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-tests    Skip running tests"
                echo "  --skip-setup    Skip running setup.py"
                echo "  -h, --help      Show this help message"
                exit 0
                ;;
            *)
                print_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Installation steps
    check_python_version || handle_error "Python version check"
    check_system_dependencies || handle_error "System dependencies check"
    create_virtual_environment || handle_error "Virtual environment creation"
    activate_virtual_environment || handle_error "Virtual environment activation"
    install_python_dependencies || handle_error "Python dependencies installation"
    create_directories || handle_error "Directory creation"
    setup_environment_file || handle_error "Environment file setup"
    
    if [ "$SKIP_SETUP" = false ]; then
        run_setup_script || handle_error "Setup script execution"
    fi
    
    if [ "$SKIP_TESTS" = false ]; then
        run_tests || handle_error "Test execution"
    fi
    
    print_final_instructions
}

# Run main function
main "$@"