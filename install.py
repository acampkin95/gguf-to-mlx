#!/usr/bin/env python3
"""
GGUF-to-MLX Release Installer
==============================
Comprehensive installer for macOS Tahoe (Sonoma/Sequoia)

This script:
1. Runs pre-flight checks (dependencies, disk space, network)
2. Installs all required packages
3. Clones/configures repositories
4. Sets up CLI alias
5. Verifies installation

Usage:
    curl -sSL https://raw.githubusercontent.com/acampkin95/gguf-to-mlx/main/install.py | python3
    python3 install.py
"""

import sys
import os
import subprocess
import shutil
import argparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_VERSION = "1.0.0"
REPO_URL = "https://github.com/acampkin95/gguf-to-mlx.git"
FORK_URL = "https://github.com/acampkin95/gguf2mlx.git"
UPSTREAM_URL = "https://github.com/barrontang/gguf2mlx.git"

MIN_DISK_GB = 10
MIN_RAM_GB = 8

REQUIRED_PACKAGES = [
    ("Python 3.10+", ["python3", "--version"]),
    ("Git", ["git", "--version"]),
    ("Homebrew", ["brew", "--version"]),
]

REQUIRED_PYTHON_PKGS = [
    "gguf>=0.18.0",
    "mlx>=0.18.0", 
    "mlx-lm>=0.18.0",
    "safetensors>=0.4.0",
    "transformers>=4.40.0",
    "rich>=13.0.0",
    "tqdm>=4.0.0",
    "psutil>=5.9.0",
]

# ═══════════════════════════════════════════════════════════════════════════
# ANSI Colors
# ═══════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def color(text: str, code: str) -> str:
    return f"{code}{text}{Colors.RESET}"


def success(msg: str):
    print(f"  {color('✓', Colors.GREEN)} {msg}")


def error(msg: str):
    print(f"  {color('✗', Colors.RED)} {msg}")


def info(msg: str):
    print(f"  {color('●', Colors.BLUE)} {msg}")


def warn(msg: str):
    print(f"  {color('⚠', Colors.YELLOW)} {msg}")


def header(msg: str):
    print()
    print(color(f"═══ {msg} ═══", Colors.CYAN))


def step(num: int, total: int, msg: str):
    print()
    print(color(f"[{num}/{total}] {msg}", Colors.BOLD + Colors.CYAN))


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: list[str], capture: bool = True, check: bool = True) -> tuple:
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=300
        )
        if check and result.returncode != 0:
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def get_disk_space(path: str = "/") -> float:
    """Get available disk space in GB."""
    try:
        import shutil
        stat = shutil.disk_usage(path)
        return stat.free / (1024**3)
    except:
        return 0


def get_ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except:
        return 0


def get_chip() -> str:
    """Get Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except:
        return "Unknown"


def get_macos_version() -> str:
    """Get macOS version."""
    try:
        result = subprocess.run(
            ["sw_vers", "-productVersion"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except:
        return "Unknown"


def check_internet() -> bool:
    """Check internet connectivity."""
    success, _ = run_cmd(["curl", "-s", "--max-time", "10", "https://github.com"])
    return success


# ═══════════════════════════════════════════════════════════════════════════
# Pre-Flight Checks
# ═══════════════════════════════════════════════════════════════════════════

def check_system(args) -> bool:
    """Run system pre-flight checks."""
    header("Pre-Flight Checks")
    
    all_passed = True
    
    # macOS Version
    version = get_macos_version()
    info(f"macOS Version: {version}")
    if version.startswith("14.") or version.startswith("15."):
        success(f"macOS Tahoe ({version}) - Supported")
    else:
        warn(f"macOS {version} - May work but Tahoe (14/15) recommended")
    
    # Apple Silicon Chip
    chip = get_chip()
    info(f"Processor: {chip}")
    if "Apple" in chip or "M1" in chip or "M2" in chip or "M3" in chip or "M4" in chip:
        success("Apple Silicon - MLX will work optimally")
    else:
        warn("Intel or unknown - MLX may not be available")
    
    # RAM
    ram = get_ram_gb()
    info(f"RAM: {ram:.1f} GB")
    if ram >= MIN_RAM_GB:
        success(f"Sufficient RAM ({ram:.0f} GB >= {MIN_RAM_GB} GB minimum)")
    else:
        error(f"Insufficient RAM ({ram:.0f} GB < {MIN_RAM_GB} GB minimum)")
        all_passed = False
    
    # Disk Space
    disk = get_disk_space()
    info(f"Free Disk: {disk:.1f} GB")
    if disk >= MIN_DISK_GB:
        success(f"Sufficient disk space ({disk:.0f} GB >= {MIN_DISK_GB} GB minimum)")
    else:
        error(f"Insufficient disk space ({disk:.0f} GB < {MIN_DISK_GB} GB minimum)")
        all_passed = False
    
    # Internet
    if check_internet():
        success("Internet connectivity - OK")
    else:
        error("No internet connectivity - Cannot download packages")
        all_passed = False
    
    print()
    return all_passed


def check_dependencies(args) -> bool:
    """Check required system dependencies."""
    header("Dependency Checks")
    
    all_found = True
    
    for name, cmd in REQUIRED_PACKAGES:
        success_flag, output = run_cmd(cmd, check=False)
        if success_flag:
            version = output.strip().split("\n")[0] if output else "OK"
            success(f"{name}: {version}")
        else:
            error(f"{name}: Not found or error")
            all_found = False
    
    # Python version
    try:
        version = sys.version_info
        if version.major >= 3 and version.minor >= 10:
            success(f"Python: {version.major}.{version.minor}.{version.micro}")
        else:
            error(f"Python: {version.major}.{version.minor} - Need 3.10+")
            all_found = False
    except:
        error("Python: Not found")
        all_found = False
    
    return all_found


# ═══════════════════════════════════════════════════════════════════════════
# Installation Steps
# ═══════════════════════════════════════════════════════════════════════════

def install_packages(args) -> bool:
    """Install Python packages."""
    step(1, 6, "Installing Python Packages")
    
    print()
    info("Installing required Python packages...")
    
    # First upgrade pip
    success_flag, _ = run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    if success_flag:
        success("Upgraded pip")
    else:
        warn("Could not upgrade pip")
    
    # Install packages
    packages = " ".join(REQUIRED_PYTHON_PKGS)
    print()
    info(f"Installing: {', '.join(REQUIRED_PYTHON_PKGS)}")
    
    success_flag, output = run_cmd(
        [sys.executable, "-m", "pip", "install", packages],
        check=False
    )
    
    if success_flag:
        success("Python packages installed successfully")
        return True
    else:
        error(f"Failed to install packages: {output}")
        return False


def clone_repos(args) -> bool:
    """Clone or update repositories."""
    step(2, 6, "Cloning Repositories")
    
    install_dir = Path.home() / "Projects" / "gguf-to-mlx"
    fork_dir = Path.home() / "Projects" / "gguf2mlx-fork"
    
    # Create Projects directory
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Clone main repo
    if install_dir.exists():
        info(f"Main repo exists at {install_dir}")
        success("Using existing repository")
    else:
        print()
        info(f"Cloning main repo to {install_dir}...")
        success_flag, output = run_cmd(
            ["git", "clone", REPO_URL, str(install_dir)],
            check=False
        )
        if success_flag:
            success("Main repo cloned")
        else:
            error(f"Failed to clone main repo: {output}")
            return False
    
    # Clone/setup fork
    if fork_dir.exists():
        info(f"Fork exists at {fork_dir}")
        success("Using existing fork")
    else:
        print()
        info(f"Cloning fork to {fork_dir}...")
        success_flag, output = run_cmd(
            ["git", "clone", FORK_URL, str(fork_dir)],
            check=False
        )
        if success_flag:
            success("Fork cloned")
            
            # Add upstream
            run_cmd(["git", "remote", "add", "upstream", UPSTREAM_URL], check=False)
            success("Added upstream remote")
        else:
            error(f"Failed to clone fork: {output}")
            return False
    
    return True


def install_fork(args) -> bool:
    """Install the gguf2mlx fork in editable mode."""
    step(3, 6, "Installing gguf2mlx Fork")
    
    fork_dir = Path.home() / "Projects" / "gguf2mlx-fork"
    
    if not fork_dir.exists():
        error(f"Fork not found at {fork_dir}")
        return False
    
    print()
    info(f"Installing fork from {fork_dir}...")
    
    success_flag, output = run_cmd(
        [sys.executable, "-m", "pip", "install", "-e", str(fork_dir)],
        check=False
    )
    
    if success_flag:
        success("gguf2mlx fork installed successfully")
        
        # Verify
        success_flag, output = run_cmd(
            [sys.executable, "-c", "import gguf2mlx; print(gguf2mlx.__version__)"],
            check=False
        )
        if success_flag:
            success(f"gguf2mlx version: {output.strip()}")
        return True
    else:
        error(f"Failed to install fork: {output}")
        return False


def setup_cli(args) -> bool:
    """Set up the CLI wrapper."""
    step(4, 6, "Setting Up CLI")
    
    install_dir = Path.home() / "Projects" / "gguf-to-mlx"
    convert_script = install_dir / "convert.py"
    
    if not convert_script.exists():
        error(f"convert.py not found at {convert_script}")
        return False
    
    print()
    info(f"CLI script: {convert_script}")
    
    # Make executable
    convert_script.chmod(0o755)
    success("Made convert.py executable")
    
    return True


def setup_alias(args) -> bool:
    """Set up shell alias."""
    step(5, 6, "Setting Up Alias")
    
    install_dir = Path.home() / "Projects" / "gguf-to-mlx"
    convert_script = str(install_dir / "convert.py")
    
    # Determine alias name
    if args.alias:
        alias_name = args.alias
        info(f"Using custom alias: {alias_name}")
    else:
        # Prompt user
        print()
        print(color("  Choose an alias name for the converter:", Colors.BOLD))
        print()
        print(f"    1. convertgguf (recommended)")
        print(f"    2. cpmm (short)")
        print(f"    3. Custom name")
        print()
        
        choice = input(color("  Enter choice (1/2/3) or alias name: ", Colors.CYAN)).strip()
        
        if choice == "1":
            alias_name = "convertgguf"
        elif choice == "2":
            alias_name = "cpmm"
        elif choice == "3":
            alias_name = input(color("  Enter custom alias: ", Colors.CYAN)).strip()
        else:
            alias_name = choice if choice else "convertgguf"
        
        print()
        info(f"Using alias: {alias_name}")
    
    # Detect shell
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        rc_file = Path.home() / ".zshrc"
    elif "bash" in shell:
        rc_file = Path.home() / ".bashrc"
    else:
        rc_file = Path.home() / ".zshrc"  # Default to zsh
    
    # Read existing aliases
    existing = {}
    if rc_file.exists():
        for line in rc_file.read_text().splitlines():
            if line.strip().startswith("alias ") and "=" in line:
                try:
                    name = line.split("=")[0].replace("alias ", "").strip()
                    existing[name] = line
                except:
                    pass
    
    # Create alias line
    alias_line = f'alias {alias_name}="python3 {convert_script}"'
    
    # Check if already exists
    if alias_name in existing:
        print()
        warn(f"Alias '{alias_name}' already exists in {rc_file}")
        if args.yes or args.force:
            info("Force flag set - overwriting")
        else:
            overwrite = input(color("  Overwrite? (y/N): ", Colors.CYAN)).strip().lower()
            if overwrite != "y":
                info("Keeping existing alias")
                return True
    
    # Add alias
    with rc_file.open("a") as f:
        f.write(f"\n# GGUF to MLX converter alias\n")
        f.write(f"{alias_line}\n")
    
    success(f"Added alias '{alias_name}' to {rc_file}")
    
    # Apply to current session
    os.system(f'eval "$(alias {alias_name})" 2>/dev/null || true')
    
    return True


def verify_installation(args) -> bool:
    """Verify the installation."""
    step(6, 6, "Verifying Installation")
    
    all_ok = True
    
    # Check Python packages
    print()
    info("Checking Python packages...")
    for pkg in ["gguf", "mlx", "mlx_lm", "rich", "tqdm", "psutil"]:
        success_flag, output = run_cmd(
            [sys.executable, "-c", f"import {pkg.replace('-', '_')}; print('OK')"],
            check=False
        )
        if success_flag:
            success(f"{pkg}: OK")
        else:
            error(f"{pkg}: Not installed")
            all_ok = False
    
    # Check gguf2mlx
    print()
    info("Checking gguf2mlx...")
    success_flag, output = run_cmd(
        [sys.executable, "-c", "import gguf2mlx; print(gguf2mlx.__version__)"],
        check=False
    )
    if success_flag:
        success(f"gguf2mlx: {output.strip()}")
    else:
        error("gguf2mlx: Not installed")
        all_ok = False
    
    # Check CLI script
    print()
    info("Checking CLI script...")
    install_dir = Path.home() / "Projects" / "gguf-to-mlx"
    convert_script = install_dir / "convert.py"
    
    if convert_script.exists():
        success(f"convert.py: {convert_script}")
        # Test help
        success_flag, output = run_cmd(
            [sys.executable, str(convert_script), "--help"],
            check=False
        )
        if success_flag:
            success("CLI help: OK")
        else:
            warn("CLI help: Failed")
    else:
        error(f"convert.py: Not found")
        all_ok = False
    
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Main Installer
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GGUF-to-MLX Release Installer for macOS Tahoe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                    # Interactive (prompts for alias)
    %(prog)s --alias cpmm      # Use 'cpmm' as alias
    %(prog)s --skip-checks      # Skip pre-flight checks
    %(prog)s --force            # Force reinstallation
        """
    )
    parser.add_argument("--alias", "-a", help="Set alias name for CLI")
    parser.add_argument("--skip-checks", "-s", action="store_true", help="Skip pre-flight checks")
    parser.add_argument("--force", "-f", action="store_true", help="Force reinstallation")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip prompts (assume yes)")
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    
    args = parser.parse_args()
    
    # Banner
    print()
    print(color("=" * 60, Colors.CYAN))
    print(color("  GGUF-to-MLX Release Installer", Colors.BOLD + Colors.CYAN))
    print(color(f"  Version {SCRIPT_VERSION} for macOS Tahoe", Colors.DIM))
    print(color("=" * 60, Colors.CYAN))
    print()
    
    # Pre-flight checks
    if not args.skip_checks:
        if not check_system(args):
            print()
            error("System check failed. Fix issues and try again.")
            sys.exit(1)
        
        if not check_dependencies(args):
            print()
            error("Dependency check failed. Install missing dependencies.")
            sys.exit(1)
    
    # Summary
    print()
    header("Installation Summary")
    
    install_dir = Path.home() / "Projects" / "gguf-to-mlx"
    print(f"  Installation directory: {install_dir}")
    print(f"  Target alias: {args.alias or 'convertgguf (default)'}")
    print(f"  macOS: {get_macos_version()}")
    print(f"  Chip: {get_chip()}")
    print(f"  RAM: {get_ram_gb():.1f} GB")
    print(f"  Free disk: {get_disk_space():.1f} GB")
    
    # User confirmation
    if not args.yes:
        print()
        response = input(color("\n  Proceed with installation? [Y/n]: ", Colors.CYAN)).strip().lower()
        if response and response != "y" and response != "yes":
            print()
            info("Installation cancelled.")
            sys.exit(0)
    
    # Run installation steps
    print()
    header("Installing")
    
    steps = [
        ("Install packages", lambda: install_packages(args)),
        ("Clone repositories", lambda: clone_repos(args)),
        ("Install gguf2mlx fork", lambda: install_fork(args)),
        ("Set up CLI", lambda: setup_cli(args)),
        ("Set up alias", lambda: setup_alias(args)),
        ("Verify installation", lambda: verify_installation(args)),
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        if not func():
            print()
            error(f"Installation failed at step: {name}")
            print()
            print("  To retry from scratch, run:")
            print(color(f"    rm -rf ~/Projects/gguf-to-mlx ~/Projects/gguf2mlx-fork", Colors.DIM))
            sys.exit(1)
    
    # Success
    print()
    print(color("=" * 60, Colors.GREEN))
    print(color("  ✓ Installation Complete!", Colors.BOLD + Colors.GREEN))
    print(color("=" * 60, Colors.GREEN))
    print()
    
    # Post-install instructions
    alias_name = args.alias or "convertgguf"
    rc_file = Path.home() / (".zshrc" if "zsh" in os.environ.get("SHELL", "") else ".bashrc")
    
    print(color("  Next Steps:", Colors.BOLD))
    print()
    print(f"  1. Restart your terminal or run:")
    print(color(f"     source {rc_file}", Colors.DIM))
    print()
    print(f"  2. Use the converter:")
    print(color(f"     {alias_name} model.gguf", Colors.DIM))
    print()
    print(f"  3. For help:")
    print(color(f"     {alias_name} --help", Colors.DIM))
    print()
    print(color("  Repository:", Colors.DIM))
    print(color("    https://github.com/acampkin95/gguf-to-mlx", Colors.DIM))
    print()


if __name__ == "__main__":
    main()