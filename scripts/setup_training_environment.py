"""
Training Environment Setup Script

This script sets up the training environment for the Discord personality bot.
It creates a virtual environment with Python 3.11, installs all training
dependencies (Unsloth, TRL, PyTorch), and verifies GPU/CUDA setup.

Usage:
    # After installing Python 3.11 alongside Python 3.14:
    py -3.11 scripts/setup_training_environment.py

    # Or if Python 3.11 is default:
    python scripts/setup_training_environment.py

Requirements:
    - Python 3.9-3.13 (NOT 3.14) - Unsloth compatibility
    - NVIDIA GPU with CUDA support (RTX 3070 8GB or better)
    - ~10GB free disk space
    - Internet connection for downloads
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, List, Tuple


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(project_root)


# ============================================================================
# Helper Functions
# ============================================================================

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step: int, text: str):
    """Print a step number and description"""
    print(f"\n{'─' * 70}")
    print(f"  STEP {step}: {text}")
    print(f"{'─' * 70}\n")


def print_success(text: str):
    """Print a success message"""
    print(f"✅ {text}")


def print_error(text: str):
    """Print an error message"""
    print(f"❌ {text}")


def print_warning(text: str):
    """Print a warning message"""
    print(f"⚠️  {text}")


def print_info(text: str):
    """Print an info message"""
    print(f"ℹ️  {text}")


def run_command(
    cmd: List[str],
    description: str,
    check: bool = True,
    capture: bool = False
) -> Optional[subprocess.CompletedProcess]:
    """
    Run a command with error handling

    Args:
        cmd: Command and arguments as list
        description: Human-readable description
        check: Raise error if command fails
        capture: Capture output

    Returns:
        CompletedProcess if capture=True, else None
    """
    try:
        print(f"  → {description}...")

        if capture:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True
            )
            return result
        else:
            subprocess.run(cmd, check=check)
            return None

    except subprocess.CalledProcessError as e:
        print_error(f"Failed: {description}")
        if capture and e.stderr:
            print(f"\n{e.stderr}")
        raise
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        raise


# ============================================================================
# Python Version Detection
# ============================================================================

def check_python_version() -> Tuple[int, int, int]:
    """
    Check if current Python version is compatible with Unsloth

    Returns:
        Tuple of (major, minor, micro) version
    """
    version = sys.version_info
    major, minor, micro = version.major, version.minor, version.micro

    print_info(f"Detected Python {major}.{minor}.{micro}")

    # Unsloth requires Python 3.9-3.13 (NOT 3.14)
    if major != 3:
        print_error(f"Python 3 required (found {major}.{minor}.{micro})")
        sys.exit(1)

    if minor < 9:
        print_error(f"Python 3.9+ required for Unsloth (found {major}.{minor}.{micro})")
        print_info("Please install Python 3.11: https://www.python.org/downloads/")
        sys.exit(1)

    if minor >= 14:
        print_error(f"Python 3.14+ not supported by Unsloth (found {major}.{minor}.{micro})")
        print_warning("Unsloth requires Python 3.9-3.13")
        print("\nIf you have Python 3.11 installed alongside 3.14:")
        print("  Run this script with: py -3.11 scripts/setup_training_environment.py")
        print("\nOtherwise, install Python 3.11:")
        print("  1. Download: https://www.python.org/downloads/release/python-3118/")
        print("  2. Run installer, check 'Add to PATH'")
        print("  3. Install to: C:\\Python311\\")
        print("  4. Then run: py -3.11 scripts/setup_training_environment.py")
        sys.exit(1)

    print_success(f"Python {major}.{minor}.{micro} is compatible with Unsloth")
    return (major, minor, micro)


def detect_python_installations() -> List[Tuple[str, str]]:
    """
    Detect all installed Python versions (Windows only)

    Returns:
        List of (version, path) tuples
    """
    installations = []

    if platform.system() != "Windows":
        return installations

    try:
        result = subprocess.run(
            ["py", "-0"],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.strip() and not line.startswith("Installed"):
                    # Parse output like: " -3.11-64  C:\Python311\python.exe"
                    parts = line.split()
                    if len(parts) >= 2:
                        version = parts[0].strip().lstrip('-')
                        path = parts[1] if len(parts) > 1 else ""
                        installations.append((version, path))
    except FileNotFoundError:
        pass

    return installations


# ============================================================================
# Virtual Environment Setup
# ============================================================================

def create_virtual_environment(venv_path: Path) -> bool:
    """
    Create virtual environment with current Python

    Args:
        venv_path: Path to virtual environment directory

    Returns:
        True if created, False if already exists
    """
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        choice = input("\nOptions:\n  1. Use existing venv (skip creation)\n  2. Delete and recreate venv\nChoose option (1 or 2): ").strip()

        if choice == '1':
            print_info("Using existing virtual environment")
            return False
        elif choice == '2':
            print_info("Deleting existing virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print_error("Invalid choice. Exiting.")
            sys.exit(0)

    print_info(f"Creating virtual environment at: {venv_path}")

    run_command(
        [sys.executable, "-m", "venv", str(venv_path)],
        f"Creating venv with Python {sys.version_info.major}.{sys.version_info.minor}"
    )

    print_success("Virtual environment created successfully")
    return True


def get_venv_python(venv_path: Path) -> Path:
    """Get path to Python executable in virtual environment"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def get_venv_pip(venv_path: Path) -> Path:
    """Get path to pip executable in virtual environment"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"


# ============================================================================
# Dependency Installation
# ============================================================================

def upgrade_pip(pip_path: Path):
    """Upgrade pip to latest version"""
    print_info("Upgrading pip to latest version...")
    run_command(
        [str(pip_path), "install", "--upgrade", "pip"],
        "Upgrading pip"
    )
    print_success("Pip upgraded successfully")


def install_pytorch(pip_path: Path):
    """Install PyTorch with CUDA support"""
    print_info("Installing PyTorch with CUDA support...")
    print_warning("This may take 5-10 minutes (downloading ~2GB)")

    # Install PyTorch with CUDA 11.8 or 12.1 support
    # Use official PyTorch index for CUDA builds
    run_command(
        [
            str(pip_path), "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ],
        "Installing PyTorch with CUDA 12.1 support"
    )

    print_success("PyTorch installed successfully")


def install_transformers_ecosystem(pip_path: Path):
    """Install transformers, datasets, and related packages"""
    print_info("Installing transformers ecosystem...")

    run_command(
        [str(pip_path), "install", "transformers", "datasets", "accelerate"],
        "Installing transformers, datasets, accelerate"
    )

    print_success("Transformers ecosystem installed")


def install_unsloth(pip_path: Path):
    """Install Unsloth for efficient QLoRA training"""
    print_info("Installing Unsloth...")
    print_warning("This may take 5-10 minutes (building from source)")

    run_command(
        [
            str(pip_path), "install",
            "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        ],
        "Installing Unsloth from GitHub"
    )

    print_success("Unsloth installed successfully")


def install_trl(pip_path: Path):
    """Install TRL for DPO training"""
    print_info("Installing TRL (Transformer Reinforcement Learning)...")

    run_command(
        [str(pip_path), "install", "trl"],
        "Installing TRL"
    )

    print_success("TRL installed successfully")


def install_additional_deps(pip_path: Path):
    """Install additional training dependencies"""
    print_info("Installing additional dependencies...")

    run_command(
        [str(pip_path), "install", "peft", "bitsandbytes", "sentencepiece"],
        "Installing PEFT, bitsandbytes, sentencepiece"
    )

    print_success("Additional dependencies installed")


# ============================================================================
# GPU and CUDA Verification
# ============================================================================

def verify_gpu_cuda(python_path: Path):
    """
    Verify GPU and CUDA availability

    Args:
        python_path: Path to Python executable in venv
    """
    print_info("Verifying GPU and CUDA setup...")

    # Test PyTorch CUDA
    test_script = """
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Get GPU memory
    total_mem = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_mem / (1024**3)
    print(f"GPU memory: {total_gb:.1f} GB")

    # Check compute capability
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {major}.{minor}")

    # Verify can allocate memory
    try:
        x = torch.randn(1000, 1000, device='cuda')
        print("✅ GPU memory allocation: SUCCESS")
    except Exception as e:
        print(f"❌ GPU memory allocation: FAILED - {e}")
        sys.exit(1)
else:
    print("❌ CUDA not available!")
    print("\\nTroubleshooting:")
    print("  1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("  2. Update GPU drivers: https://www.nvidia.com/download/index.aspx")
    print("  3. Verify GPU: Run 'nvidia-smi' in command prompt")
    sys.exit(1)
"""

    result = subprocess.run(
        [str(python_path), "-c", test_script],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print_error("GPU/CUDA verification failed!")
        if result.stderr:
            print(result.stderr)
        print("\n⚠️  Training requires NVIDIA GPU with CUDA support")
        print("If you're on RTX 3070, ensure:")
        print("  1. Latest GPU drivers installed")
        print("  2. CUDA Toolkit 11.8 or 12.1 installed")
        print("  3. Run 'nvidia-smi' to verify GPU is detected")
        sys.exit(1)

    print_success("GPU and CUDA verified successfully")


# ============================================================================
# Installation Testing
# ============================================================================

def test_installation(python_path: Path):
    """
    Test that all training packages can be imported

    Args:
        python_path: Path to Python executable in venv
    """
    print_info("Testing installation...")

    test_script = """
import sys

packages = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('datasets', 'Datasets'),
    ('accelerate', 'Accelerate'),
    ('unsloth', 'Unsloth'),
    ('trl', 'TRL'),
    ('peft', 'PEFT'),
]

failed = []

for module_name, display_name in packages:
    try:
        __import__(module_name)
        print(f"✅ {display_name}")
    except ImportError as e:
        print(f"❌ {display_name}: {e}")
        failed.append(display_name)

if failed:
    print(f"\\n❌ Failed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\\n✅ All packages imported successfully!")
"""

    result = subprocess.run(
        [str(python_path), "-c", test_script],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print_error("Installation test failed!")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    print_success("Installation test passed")


# ============================================================================
# Helper Scripts
# ============================================================================

def create_activation_script(venv_path: Path):
    """
    Create easy activation scripts for the virtual environment

    Args:
        venv_path: Path to virtual environment directory
    """
    print_info("Creating activation helper scripts...")

    # Windows batch script
    if platform.system() == "Windows":
        activate_bat = project_root / "activate_training.bat"
        activate_bat.write_text(f"""@echo off
echo Activating training environment...
call "{venv_path}\\Scripts\\activate.bat"
echo.
echo ✅ Training environment activated!
echo.
echo You can now run:
echo   python scripts/2_prepare_training_data.py
echo   python scripts/3_train_model.py --mode sft+dpo
echo.
""")
        print_success(f"Created: {activate_bat.name}")

    # Linux/Mac shell script
    else:
        activate_sh = project_root / "activate_training.sh"
        activate_sh.write_text(f"""#!/bin/bash
echo "Activating training environment..."
source "{venv_path}/bin/activate"
echo ""
echo "✅ Training environment activated!"
echo ""
echo "You can now run:"
echo "  python scripts/2_prepare_training_data.py"
echo "  python scripts/3_train_model.py --mode sft+dpo"
echo ""
""")
        activate_sh.chmod(0o755)
        print_success(f"Created: {activate_sh.name}")


def print_next_steps(venv_path: Path):
    """Print instructions for next steps"""
    print_header("Setup Complete! 🎉")

    print("Your training environment is ready to use!\n")

    if platform.system() == "Windows":
        print("To activate the environment:")
        print("  • Double-click: activate_training.bat")
        print(f"  • Or run: {venv_path}\\Scripts\\activate.bat\n")
    else:
        print("To activate the environment:")
        print("  • Run: source activate_training.sh")
        print(f"  • Or: source {venv_path}/bin/activate\n")

    print("Next steps:")
    print("  1. Activate the training environment (see above)")
    print("  2. Run: python scripts/2_prepare_training_data.py")
    print("  3. Run: python scripts/3_train_model.py --mode sft+dpo")
    print("  4. Wait 5-7 hours for training to complete")
    print("  5. Convert to GGUF and deploy to laptop\n")

    print("Training Resources:")
    print(f"  • RTX 3070 8GB: ✅ Comfortable")
    print(f"  • Training time: 5-7 hours (SFT + DPO)")
    print(f"  • Peak VRAM: 7.2-7.5 GB")
    print(f"  • Model: Qwen2.5-3B-Instruct\n")

    print("Documentation:")
    print("  • README.md: Quick start guide")
    print("  • CLAUDE.md: Comprehensive training guide")
    print("  • TODO.md: Project roadmap\n")


# ============================================================================
# Main Setup Flow
# ============================================================================

def main():
    """Main setup flow"""
    print_header("Training Environment Setup - Discord Personality Bot")

    print("This script will set up your training environment with:")
    print("  • Python 3.11 virtual environment")
    print("  • PyTorch with CUDA support")
    print("  • Transformers, Datasets, Accelerate")
    print("  • Unsloth (efficient QLoRA training)")
    print("  • TRL (Direct Preference Optimization)")
    print("  • GPU/CUDA verification")
    print()
    print("Requirements:")
    print("  • Python 3.9-3.13 (current version will be checked)")
    print("  • NVIDIA GPU with CUDA (RTX 3070 8GB recommended)")
    print("  • ~10GB free disk space")
    print("  • Internet connection")
    print()
    print("Estimated time: 15-20 minutes")
    print()

    ready = input("Ready to begin? (y/n): ").strip().lower()
    if ready != 'y':
        print("\nSetup cancelled. Run this script again when you're ready!")
        sys.exit(0)

    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    version = check_python_version()

    installations = detect_python_installations()
    if installations:
        print_info("Detected Python installations:")
        for ver, path in installations:
            print(f"  • Python {ver}: {path}")

    # Step 2: Create virtual environment
    print_step(2, "Creating virtual environment")
    venv_path = project_root / "venv_training"
    created = create_virtual_environment(venv_path)

    python_path = get_venv_python(venv_path)
    pip_path = get_venv_pip(venv_path)

    # Verify venv Python
    print_info(f"Virtual environment Python: {python_path}")

    # Step 3: Install dependencies
    if created:
        print_step(3, "Installing training dependencies")
        print_warning("This will take 15-20 minutes. Installing:")
        print("  • PyTorch with CUDA (~2GB download)")
        print("  • Transformers ecosystem")
        print("  • Unsloth (builds from source)")
        print("  • TRL and other training tools")
        print()

        upgrade_pip(pip_path)
        install_pytorch(pip_path)
        install_transformers_ecosystem(pip_path)
        install_unsloth(pip_path)
        install_trl(pip_path)
        install_additional_deps(pip_path)

        print_success("All dependencies installed")
    else:
        print_step(3, "Skipping dependency installation")
        print_info("Using existing packages in virtual environment")

    # Step 4: Verify GPU and CUDA
    print_step(4, "Verifying GPU and CUDA")
    verify_gpu_cuda(python_path)

    # Step 5: Test installation
    print_step(5, "Testing installation")
    test_installation(python_path)

    # Step 6: Create helper scripts
    print_step(6, "Creating helper scripts")
    create_activation_script(venv_path)

    # Done!
    print_next_steps(venv_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
