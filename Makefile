# GGUF-to-MLX Release Builder
# ==============================
# Build macOS release installer

.PHONY: help install test clean verify release check run-installer

# Default target
help:
	@echo "GGUF-to-MLX Release Builder"
	@echo ""
	@echo "Available targets:"
	@echo "  make install    - Run the installer"
	@echo "  make check      - Run pre-flight checks only"
	@echo "  make verify     - Verify installation"
	@echo "  make test       - Run tests"
	@echo "  make release    - Create release package"
	@echo "  make clean      - Clean build artifacts"

# Run the installer
install:
	@echo "Running GGUF-to-MLX installer..."
	@chmod +x install.py
	@python3 install.py

# Interactive installer (with prompts)
run-installer:
	@python3 install.py

# Pre-flight checks only
check:
	@python3 install.py --skip-checks

# Verify installation
verify:
	@echo "Verifying GGUF-to-MLX installation..."
	@python3 -c "import gguf2mlx; print(f'gguf2mlx: {gguf2mlx.__version__}')" && \
	python3 -c "import rich; print('rich: OK')" && \
	python3 -c "import mlx; print('mlx: OK')" && \
	python3 convert.py --help > /dev/null && \
	echo "✓ All checks passed"

# Run tests
test:
	@echo "Running tests..."
	@python3 -m pytest test_convert.py -v
	@python3 -m mypy convert.py --ignore-missing-imports

# Lint and format
lint:
	@echo "Running linters..."
	@python3 -m ruff check convert.py
	@python3 -m black --check convert.py

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache .benchmarks
	@rm -rf *.egg-info build dist
	@find . -name "*.pyc" -delete
	@echo "✓ Clean complete"

# Create release package
release: test clean
	@echo "Creating release package..."
	@git add -A
	@git status
	@echo ""
	@echo "To create a release:"
	@echo "  1. git commit -m 'Release vX.Y.Z'"
	@echo "  2. git tag vX.Y.Z"
	@echo "  3. git push origin main --tags"
	@echo ""
	@echo "Or use GitHub Actions for automated releases."

# Quick test of the converter
run:
	@echo "Testing convert.py..."
	@python3 convert.py --help

# Development setup
dev-setup:
	@echo "Setting up development environment..."
	@pip install -e ".[dev]" --quiet && \
	pip install -e ~/Projects/gguf2mlx-fork --quiet && \
	echo "✓ Development environment ready"

# Update from upstream
update:
	@echo "Updating from upstream..."
	@cd ~/Projects/gguf2mlx-fork && \
	git fetch upstream && \
	git merge upstream/main && \
	git push && \
	echo "✓ Fork updated from upstream"