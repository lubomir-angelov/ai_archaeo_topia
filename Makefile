.PHONY: help venv-dir venv activate install setup

VENV_DIR := $(HOME)/venvs
VENV_NAME := ai_archaeotopia
VENV_PATH := $(VENV_DIR)/$(VENV_NAME)
PYTHON := python3

help:
    @echo "Available commands:"
    @echo "  make setup     - Complete setup (creates venv, activates, installs packages)"
    @echo "  make venv-dir  - Create ~/venvs directory if it doesn't exist"
    @echo "  make venv      - Create virtual environment"
    @echo "  make activate  - Activate virtual environment"
    @echo "  make install   - Install packages from requirements.txt"

venv-dir:
    @mkdir -p $(VENV_DIR)
    @echo "Created $(VENV_DIR) directory"

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
    @. $(VENV_PATH)/bin/activate && pip install -r requirements.txt
    @echo "Packages installed successfully"

setup: venv-dir venv install
    @echo "Setup complete! To activate, run: source $(VENV_PATH)/bin/activate"