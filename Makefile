.PHONY: help venv-dir venv activate install system-deps check-gdal setup

VENV_DIR := $(HOME)/venvs
VENV_NAME := ai_archaeotopia
VENV_PATH := $(VENV_DIR)/$(VENV_NAME)
PYTHON := python3
PIP := $(VENV_PATH)/bin/pip

help:
	@echo "Available commands:"
	@echo "  make setup              - Complete setup (system deps, venv, installs packages)"
	@echo "  make venv-dir           - Create ~/venvs directory if it doesn't exist"
	@echo "  make system-deps        - Update system and install GDAL dependencies"
	@echo "  make check-gdal         - Check GDAL version"
	@echo "  make venv               - Create virtual environment"
	@echo "  make activate           - Show activation command"
	@echo "  make install            - Install packages from requirements.txt"

venv-dir:
	@mkdir -p $(VENV_DIR)
	@echo "Created $(VENV_DIR) directory"

system-deps:
	@echo "Updating system packages..."
	sudo apt-get update
	@echo "Installing GDAL system dependencies..."
	sudo apt-get install -y gdal-bin libgdal-dev python3-gdal
	@echo "System dependencies installed"

check-gdal:
	@echo "Checking GDAL version..."
	gdalinfo --version

venv: venv-dir
	@if [ ! -d "$(VENV_PATH)" ]; then \
		$(PYTHON) -m venv $(VENV_PATH); \
		echo "Created virtual environment at $(VENV_PATH)"; \
	else \
		echo "Virtual environment already exists at $(VENV_PATH)"; \
	fi

activate: venv
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV_PATH)/bin/activate"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Packages installed successfully"

setup: system-deps check-gdal venv install
	@echo "Setup complete! To activate, run: source $(VENV_PATH)/bin/activate"