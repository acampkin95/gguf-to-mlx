#!/usr/bin/env python3
"""
GGUF-to-MLX Release Installer & Updater
========================================
Comprehensive installer for macOS Tahoe (Sonoma/Sequoia)

Features:
- Pre-flight checks (dependencies, disk space, network)
- Auto-update checker (checks latest version from GitHub)
- Interactive menu (Install, Update, Verify, Remove)
- One-command installation
- Alias setup with custom name prompt

Usage:
    curl -sSL https://raw.githubusercontent.com/acampkin95/gguf-to-mlx/main/install.py | python3
    python3 install.py
    python3 install.py --update      # Check for updates
    python3 install.py --verify      # Verify installation
    python3 install.py --menu        # Interactive menu
"""

import sys
import os
import re
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
RAW_INSTALL_URL = "https://raw.githubusercontent.com/acampkin95/gguf-to-mlx/main/install.py"

MIN_DISK_GB = 10
MIN_RAM_GB = 8

INSTALL_DIR = Path.home() / "Projects" / "gguf-to-mlx"
FORK_DIR = Path.home() / "Projects" / "gguf2mlx-fork"

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


def prompt(msg: str) -> str:
    return input(color(f"  {msg}: ", Colors.CYAN)).strip()


def confirm(msg: str, default: str = "y", auto_yes: bool = False) -> bool:
    if auto_yes:
        return True
    try:
        response = prompt(f"{msg} [{default}]").strip().lower()
        if not response:
            return default == "y"
        return response in ("y", "yes")
    except EOFError:
        return default == "y"


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: list[str], capture: bool = True, check: bool = True, cwd: str = None) -> tuple:
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=300,
            cwd=cwd
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


def get_installed_version() -> str:
    """Get the installed version of the script."""
    if INSTALL_DIR.exists():
        git_dir = INSTALL_DIR / ".git"
        if git_dir.exists():
            success_flag, output = run_cmd(
                ["git", "-C", str(INSTALL_DIR), "rev-parse", "--short", "HEAD"],
                check=False
            )
            if success_flag:
                return output.strip()
    return "Not installed"


def get_latest_version() -> tuple[bool, str]:
    """Check for latest version from GitHub."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", REPO_URL, "main"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            commit = result.stdout.split()[0][:7] if result.stdout else "unknown"
            return True, commit
        return False, "Could not fetch"
    except:
        return False, "Network error"


def get_current_alias() -> str:
    """Get the current configured alias."""
    rc_file = Path.home() / ".zshrc"
    if not rc_file.exists():
        rc_file = Path.home() / ".bashrc"
    
    if rc_file.exists():
        for line in rc_file.read_text().splitlines():
            if "gguf-to-mlx" in line and line.strip().startswith("alias "):
                match = re.search(r'alias\s+(\w+)=', line)
                if match:
                    return match.group(1)
    return "cpmm"  # Default


# ═══════════════════════════════════════════════════════════════════════════
# Auto-Checks
# ═══════════════════════════════════════════════════════════════════════════

def run_auto_checks() -> dict:
    """Run all automatic checks and return results."""
    results = {
        "macos_version": get_macos_version(),
        "chip": get_chip(),
        "ram_gb": get_ram_gb(),
        "disk_gb": get_disk_space(),
        "internet": check_internet(),
        "installed_version": get_installed_version(),
        "alias": get_current_alias(),
    }
    
    # Check for updates
    has_update, latest = get_latest_version()
    results["has_update"] = has_update
    results["latest_version"] = latest if has_update else "Unknown"
    results["update_available"] = (
        has_update and 
        results["installed_version"] != "Not installed" and
        results["installed_version"] != latest
    )
    
    # Check installed packages
    results["packages_ok"] = {}
    for pkg in ["gguf", "mlx", "mlx_lm", "rich", "tqdm", "psutil", "gguf2mlx"]:
        success_flag, _ = run_cmd(
            [sys.executable, "-c", f"import {pkg.replace('-', '_')}; print('OK')"],
            check=False
        )
        results["packages_ok"][pkg] = success_flag
    
    return results


def display_auto_checks(checks: dict):
    """Display auto-check results."""
    header("Auto-Check Results")
    
    print(f"\n  {'System Info':<25}")
    print(color("  ─" * 30, Colors.DIM))
    print(f"  {'macOS:':<20} {checks['macos_version']}")
    print(f"  {'Chip:':<20} {checks['chip']}")
    print(f"  {'RAM:':<20} {checks['ram_gb']:.1f} GB")
    print(f"  {'Free Disk:':<20} {checks['disk_gb']:.1f} GB")
    
    print(f"\n  {'Installation Status':<25}")
    print(color("  ─" * 30, Colors.DIM))
    print(f"  {'Alias:':<20} {checks['alias']}")
    print(f"  {'Version:':<20} {checks['installed_version']}")
    
    if checks["update_available"]:
        print()
        warn(f"  Update available: {checks['latest_version']} (you have {checks['installed_version']})")
    elif checks["installed_version"] != "Not installed":
        success("  Up to date")
    else:
        info("  Not installed")
    
    print(f"\n  {'Internet:':<20} {'✓ Connected' if checks['internet'] else '✗ Offline'}")
    
    # Package status
    all_ok = all(checks["packages_ok"].values())
    print(f"\n  {'Packages:':<20} {'✓ All installed' if all_ok else '⚠ Some missing'}")
    for pkg, ok in checks["packages_ok"].items():
        status = "✓" if ok else "✗"
        print(f"    {status} {pkg}")
    
    return checks["ram_gb"] >= MIN_RAM_GB and checks["disk_gb"] >= MIN_DISK_GB and checks["internet"]


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
    if version.startswith("14.") or version.startswith("15.") or version.startswith("16."):
        success(f"macOS Tahoe ({version}) - Supported")
    else:
        warn(f"macOS {version} - May work but Tahoe (14/15/16) recommended")
    
    # Apple Silicon Chip
    chip = get_chip()
    info(f"Processor: {chip}")
    if "Apple" in chip or any(c in chip for c in ["M1", "M2", "M3", "M4", "M5"]):
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
# Updater Functions
# ═══════════════════════════════════════════════════════════════════════════

def update_main_repo(args) -> bool:
    """Update main repository from GitHub."""
    header("Updating Main Repository")
    
    if not INSTALL_DIR.exists():
        error(f"Installation directory not found: {INSTALL_DIR}")
        return False
    
    print(f"\n  Current: {get_installed_version()}")
    print(f"  Latest: {get_latest_version()[1]}")
    print()
    
    if not confirm("Pull latest changes", default="y"):
        info("Update cancelled")
        return True
    
    print()
    info("Pulling from origin...")
    
    success_flag, output = run_cmd(
        ["git", "-C", str(INSTALL_DIR), "pull", "origin", "main"],
        check=False
    )
    
    if success_flag:
        success("Main repo updated")
        new_version = get_installed_version()
        print(f"  Now at: {new_version}")
        return True
    else:
        error(f"Failed to update: {output}")
        return False


def update_fork(args) -> bool:
    """Update gguf2mlx fork from upstream."""
    header("Updating gguf2mlx Fork")
    
    if not FORK_DIR.exists():
        info("Fork not installed, skipping")
        return True
    
    # Fetch upstream
    info("Fetching from upstream...")
    success_flag, _ = run_cmd(
        ["git", "-C", str(FORK_DIR), "fetch", "upstream"],
        cwd=str(FORK_DIR),
        check=False
    )
    
    if not success_flag:
        warn("Could not fetch from upstream")
    
    # Pull from origin
    print()
    info("Pulling from origin...")
    success_flag, output = run_cmd(
        ["git", "-C", str(FORK_DIR), "pull", "origin", "main"],
        cwd=str(FORK_DIR),
        check=False
    )
    
    if success_flag:
        success("Fork updated")
        
        # Reinstall
        print()
        info("Reinstalling fork...")
        success_flag, output = run_cmd(
            [sys.executable, "-m", "pip", "install", "-e", str(FORK_DIR)],
            check=False
        )
        if success_flag:
            success("Fork reinstalled")
        return True
    else:
        warn(f"Could not pull: {output}")
        return False


def update_all(args) -> bool:
    """Update everything."""
    header("Updating All")
    
    if not update_main_repo(args):
        return False
    
    if not update_fork(args):
        return False
    
    # Verify
    print()
    return verify_installation(args)


def check_for_updates(args) -> bool:
    """Check if updates are available."""
    header("Checking for Updates")
    
    print()
    
    installed = get_installed_version()
    has_update, latest = get_latest_version()
    
    print(f"  Installed version: {installed}")
    print(f"  Latest version:    {latest}")
    print()
    
    if installed == "Not installed":
        info("No installation found. Use --install to set up.")
        return False
    
    if not has_update:
        warn("Could not check for updates. Network issue?")
        return False
    
    if installed != latest:
        success(f"Update available: {latest}")
        print()
        print(f"  To update, run: {sys.argv[0]} --update")
        return True
    else:
        success("You have the latest version")
        return True


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
    
    # Create Projects directory
    INSTALL_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # Clone main repo
    if INSTALL_DIR.exists():
        info(f"Main repo exists at {INSTALL_DIR}")
        if args.force:
            print()
            if confirm("Re-clone main repo", default="n"):
                shutil.rmtree(INSTALL_DIR)
            else:
                success("Using existing repository")
        else:
            success("Using existing repository")
    else:
        print()
        info(f"Cloning main repo to {INSTALL_DIR}...")
        success_flag, output = run_cmd(
            ["git", "clone", REPO_URL, str(INSTALL_DIR)],
            check=False
        )
        if success_flag:
            success("Main repo cloned")
        else:
            error(f"Failed to clone main repo: {output}")
            return False
    
    # Clone/setup fork
    if FORK_DIR.exists():
        info(f"Fork exists at {FORK_DIR}")
        success("Using existing fork")
    else:
        print()
        info(f"Cloning fork to {FORK_DIR}...")
        success_flag, output = run_cmd(
            ["git", "clone", FORK_URL, str(FORK_DIR)],
            check=False
        )
        if success_flag:
            success("Fork cloned")
            
            # Add upstream
            run_cmd(["git", "-C", str(FORK_DIR), "remote", "add", "upstream", UPSTREAM_URL], check=False)
            success("Added upstream remote")
        else:
            error(f"Failed to clone fork: {output}")
            return False
    
    return True


def install_fork(args) -> bool:
    """Install the gguf2mlx fork in editable mode."""
    step(3, 6, "Installing gguf2mlx Fork")
    
    if not FORK_DIR.exists():
        error(f"Fork not found at {FORK_DIR}")
        return False
    
    print()
    info(f"Installing fork from {FORK_DIR}...")
    
    success_flag, output = run_cmd(
        [sys.executable, "-m", "pip", "install", "-e", str(FORK_DIR)],
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
    
    convert_script = INSTALL_DIR / "convert.py"
    
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
    
    convert_script = str(INSTALL_DIR / "convert.py")
    
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
        
        choice = prompt("Enter choice (1/2/3) or alias name")
        
        if choice == "1":
            alias_name = "convertgguf"
        elif choice == "2":
            alias_name = "cpmm"
        elif choice == "3":
            alias_name = prompt("Enter custom alias name")
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
        if args.force:
            info("Force flag set - overwriting")
        else:
            if confirm("Overwrite existing alias"):
                pass
            else:
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
    convert_script = INSTALL_DIR / "convert.py"
    
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


def remove_installation(args) -> bool:
    """Remove the installation."""
    header("Removing Installation")
    
    print()
    warn("This will remove:")
    print(f"  - {INSTALL_DIR}")
    print(f"  - {FORK_DIR}")
    print(f"  - Alias from ~/.zshrc or ~/.bashrc")
    print()
    
    if not confirm("Are you sure you want to remove", default="n"):
        info("Removal cancelled")
        return True
    
    # Remove directories
    if INSTALL_DIR.exists():
        print()
        info(f"Removing {INSTALL_DIR}...")
        shutil.rmtree(INSTALL_DIR)
        success("Main repo removed")
    
    if FORK_DIR.exists():
        print()
        info(f"Removing {FORK_DIR}...")
        shutil.rmtree(FORK_DIR)
        success("Fork removed")
    
    # Remove alias
    rc_file = Path.home() / ".zshrc"
    if not rc_file.exists():
        rc_file = Path.home() / ".bashrc"
    
    if rc_file.exists():
        content = rc_file.read_text()
        new_content = "\n".join([
            line for line in content.splitlines()
            if "gguf-to-mlx" not in line
        ])
        rc_file.write_text(new_content)
        success("Alias removed from shell config")
    
    print()
    success("Installation removed")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Interactive Menu
# ═══════════════════════════════════════════════════════════════════════════

def show_menu():
    """Display interactive menu."""
    # Run auto-checks first
    checks = run_auto_checks()
    display_auto_checks(checks)
    
    print()
    header("Options")
    print()
    print(color("  1. ", Colors.CYAN) + "Install / Reinstall")
    print(color("  2. ", Colors.CYAN) + "Update all")
    print(color("  3. ", Colors.CYAN) + "Check for updates")
    print(color("  4. ", Colors.CYAN) + "Verify installation")
    print(color("  5. ", Colors.CYAN) + "Change alias")
    print(color("  6. ", Colors.CYAN) + "Remove installation")
    print()
    print(color("  Q. ", Colors.YELLOW) + "Quit")
    print()


def run_menu(args):
    """Run interactive menu."""
    while True:
        print()
        print(color("=" * 60, Colors.CYAN))
        print(color("  GGUF-to-MLX Installer & Updater", Colors.BOLD + Colors.CYAN))
        print(color(f"  Version {SCRIPT_VERSION}", Colors.DIM))
        print(color("=" * 60, Colors.CYAN))
        
        show_menu()
        
        choice = prompt("Select option").strip().lower()
        
        if choice in ("q", "quit", "exit"):
            print()
            info("Goodbye!")
            break
        
        elif choice == "1":
            print()
            if not check_system(args) or not check_dependencies(args):
                print()
                if not confirm("Continue anyway", default="n"):
                    continue
            args.force = True  # Force reinstall
            args.alias = None  # Prompt for alias
            
            # Run install steps
            if not install_packages(args):
                continue
            if not clone_repos(args):
                continue
            if not install_fork(args):
                continue
            if not setup_cli(args):
                continue
            if not setup_alias(args):
                continue
            verify_installation(args)
        
        elif choice == "2":
            update_all(args)
        
        elif choice == "3":
            check_for_updates(args)
        
        elif choice == "4":
            print()
            header("Verification")
            # Run quick verify
            checks = run_auto_checks()
            all_ok = all(checks["packages_ok"].values())
            if all_ok and checks["installed_version"] != "Not installed":
                success("Installation looks good!")
            else:
                warn("Installation has issues. Run option 1 to fix.")
        
        elif choice == "5":
            print()
            header("Change Alias")
            new_alias = prompt("Enter new alias name")
            if new_alias:
                args.alias = new_alias
                args.force = True
                setup_alias(args)
        
        elif choice == "6":
            remove_installation(args)
        
        else:
            warn("Invalid option. Try again.")


# ═══════════════════════════════════════════════════════════════════════════
# Main Installer
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GGUF-to-MLX Release Installer & Updater for macOS Tahoe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                    # Interactive menu
    %(prog)s --install          # Full installation
    %(prog)s --update           # Check and apply updates
    %(prog)s --verify           # Verify installation
    %(prog)s --menu             # Interactive menu
    %(prog)s --remove           # Remove installation
    %(prog)s --check            # Auto-checks only
    %(prog)s --alias cpmm       # Custom alias
        """
    )
    parser.add_argument("--install", action="store_true", help="Run full installation")
    parser.add_argument("--update", action="store_true", help="Check and apply updates")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--menu", "-m", action="store_true", help="Interactive menu")
    parser.add_argument("--check", action="store_true", help="Run auto-checks only")
    parser.add_argument("--remove", action="store_true", help="Remove installation")
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
    
    # Handle different modes
    if args.check:
        # Auto-checks only
        checks = run_auto_checks()
        display_auto_checks(checks)
        sys.exit(0)
    
    if args.menu:
        # Interactive menu
        run_menu(args)
        sys.exit(0)
    
    if args.remove:
        # Remove installation
        remove_installation(args)
        sys.exit(0)
    
    if args.verify:
        # Verify only
        checks = run_auto_checks()
        display_auto_checks(checks)
        if all(checks["packages_ok"].values()) and checks["installed_version"] != "Not installed":
            print()
            success("Installation verified!")
        else:
            print()
            warn("Installation needs attention. Run with --install to fix.")
        sys.exit(0)
    
    if args.update:
        # Update
        update_all(args)
        sys.exit(0)
    
    if not args.install:
        # No specific action, show auto-checks then menu
        checks = run_auto_checks()
        display_auto_checks(checks)
        print()
        if confirm("Run full installation", default="y"):
            args.install = True
        else:
            if confirm("Show interactive menu", default="n"):
                run_menu(args)
                sys.exit(0)
            info("Exiting. Run with --help for options.")
            sys.exit(0)
    
    # Run installation
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
    
    print(f"  Installation directory: {INSTALL_DIR}")
    print(f"  Target alias: {args.alias or 'convertgguf (default)'}")
    print(f"  macOS: {get_macos_version()}")
    print(f"  Chip: {get_chip()}")
    print(f"  RAM: {get_ram_gb():.1f} GB")
    print(f"  Free disk: {get_disk_space():.1f} GB")
    
    # User confirmation
    if not args.yes:
        print()
        response = prompt("Proceed with installation").strip().lower()
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
            print(color(f"    {sys.argv[0]} --remove && {sys.argv[0]} --install", Colors.DIM))
            sys.exit(1)
    
    # Success
    print()
    print(color("=" * 60, Colors.GREEN))
    print(color("  ✓ Installation Complete!", Colors.BOLD + Colors.GREEN))
    print(color("=" * 60, Colors.GREEN))
    print()
    
    # Post-install instructions
    alias_name = args.alias or get_current_alias()
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