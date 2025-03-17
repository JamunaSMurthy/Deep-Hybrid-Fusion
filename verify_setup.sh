#!/bin/bash

# Deep Hybrid Fusion - Verification & Setup Checklist
# This script helps verify that everything is set up correctly

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                   DHF VERIFICATION CHECKLIST                      ║"
echo "║   This script verifies your DHF installation and setup            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print success
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to print warning
warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Counter for checks
total_checks=0
passed_checks=0

# ============================================================================
# Section 1: Python Environment
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. PYTHON ENVIRONMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python installation
total_checks=$((total_checks + 1))
if command_exists python3; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    success "Python 3 installed: $python_version"
    passed_checks=$((passed_checks + 1))
else
    error "Python 3 not found (required: 3.8+)"
fi

# Check pip
total_checks=$((total_checks + 1))
if command_exists pip3; then
    pip_version=$(pip3 --version 2>&1 | awk '{print $2}')
    success "pip3 installed: $pip_version"
    passed_checks=$((passed_checks + 1))
else
    error "pip3 not found"
fi

echo ""

# ============================================================================
# Section 2: Directory Structure
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. DIRECTORY STRUCTURE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

directories=(
    "config"
    "Preprocessing"
    "TextFeatureExtraction"
    "ImageFeatureExtraction"
    "AudioFeatureExtraction"
    "ModalityRepresentation"
    "DeepHybridFusion"
)

for dir in "${directories[@]}"; do
    total_checks=$((total_checks + 1))
    if [ -d "$dir" ]; then
        success "Directory exists: $dir"
        passed_checks=$((passed_checks + 1))
    else
        error "Directory missing: $dir"
    fi
done

echo ""

# ============================================================================
# Section 3: Key Files
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. KEY FILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

files=(
    "requirements.txt"
    "config/config.yaml"
    "config/config_loader.py"
    "train.py"
    "inference.py"
    "evaluate.py"
    "example_usage.py"
    "README.md"
    "SETUP.md"
)

for file in "${files[@]}"; do
    total_checks=$((total_checks + 1))
    if [ -f "$file" ]; then
        success "File exists: $file"
        passed_checks=$((passed_checks + 1))
    else
        error "File missing: $file"
    fi
done

echo ""

# ============================================================================
# Section 4: Python Packages
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. PYTHON PACKAGES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check core packages
packages=("torch" "transformers" "librosa" "cv2" "sklearn" "numpy" "pyyaml")
package_names=("PyTorch" "Transformers" "Librosa" "OpenCV" "Scikit-learn" "NumPy" "PyYAML")

for i in "${!packages[@]}"; do
    total_checks=$((total_checks + 1))
    if python3 -c "import ${packages[$i]}" 2>/dev/null; then
        version=$(python3 -c "import ${packages[$i]}; print(getattr(${packages[$i]}, '__version__', 'unknown'))" 2>/dev/null)
        success "${package_names[$i]}: $version"
        passed_checks=$((passed_checks + 1))
    else
        warning "${package_names[$i]}: not installed"
    fi
done

echo ""

# ============================================================================
# Section 5: GPU Setup (Optional)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. GPU SETUP (Optional)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    total_checks=$((total_checks + 1))
    cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    success "CUDA available: $cuda_version"
    success "Number of GPUs: $gpu_count"
    passed_checks=$((passed_checks + 2))
else
    total_checks=$((total_checks + 1))
    warning "CUDA not available (CPU mode will be used)"
fi

echo ""

# ============================================================================
# Section 6: Configuration
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. CONFIGURATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if config loads
total_checks=$((total_checks + 1))
if python3 -c "from config import get_config; config = get_config(); print(config['model']['text_dim'])" 2>/dev/null | grep -q "768"; then
    success "Configuration loads successfully"
    passed_checks=$((passed_checks + 1))
else
    error "Configuration loading failed"
fi

echo ""

# ============================================================================
# Section 7: Module Imports
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. MODULE IMPORTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

modules=(
    "TextFeatureExtraction"
    "ImageFeatureExtraction"
    "AudioFeatureExtraction"
    "ModalityRepresentation"
    "DeepHybridFusion"
)

for module in "${modules[@]}"; do
    total_checks=$((total_checks + 1))
    if python3 -c "import ${module}" 2>/dev/null; then
        success "Module import: $module"
        passed_checks=$((passed_checks + 1))
    else
        error "Module import failed: $module"
    fi
done

echo ""

# ============================================================================
# Section 8: Disk Space
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. DISK SPACE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

total_checks=$((total_checks + 1))
available_space=$(df . | awk 'NR==2 {print $4}')
if [ "$available_space" -gt 2097152 ]; then  # 2GB in KB
    available_gb=$(echo "scale=2; $available_space / (1024*1024)" | bc)
    success "Available disk space: ${available_gb}GB (sufficient)"
    passed_checks=$((passed_checks + 1))
else
    warning "Low disk space available"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                        VERIFICATION SUMMARY                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

percentage=$((passed_checks * 100 / total_checks))

echo ""
echo "Checks passed: $passed_checks / $total_checks ($percentage%)"
echo ""

if [ $passed_checks -eq $total_checks ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python example_usage.py"
    echo "  2. Read: README.md"
    echo "  3. Train: python train.py"
    echo ""
elif [ $passed_checks -ge $((total_checks * 80 / 100)) ]; then
    echo -e "${YELLOW}⚠ MOST CHECKS PASSED (80%+)${NC}"
    echo ""
    echo "You can proceed, but some features may not work optimally."
    echo "See above for missing items marked with ✗"
    echo ""
else
    echo -e "${RED}✗ CRITICAL CHECKS FAILED${NC}"
    echo ""
    echo "Installation incomplete. Please fix the issues marked with ✗"
    echo "Refer to SETUP.md for installation instructions."
    echo ""
fi

# Suggestions
if ! python3 -c "import torch" 2>/dev/null; then
    echo "📌 Suggestion: Install PyTorch"
    echo "   pip install torch torchvision torchaudio"
    echo ""
fi

if ! python3 -c "import transformers" 2>/dev/null; then
    echo "📌 Suggestion: Install Transformers"
    echo "   pip install transformers"
    echo ""
fi

if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "📌 Note: GPU not available. Using CPU mode."
    echo "   To use GPU: Install CUDA and cuDNN"
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "For help, refer to SETUP.md or README.md"
echo "═══════════════════════════════════════════════════════════════════════"
