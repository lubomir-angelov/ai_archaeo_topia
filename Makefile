SHELL := /usr/bin/env bash

.PHONY: help venv-dir venv activate install system-deps check-gdal setup
.PHONY: wsl-system-deps cce-install cce-init cce-uninstall
.PHONY: lint format-check test compile
.PHONY: sam2-mcp-run sam2-mcp-health sam2-mcp-smoke sam2-mcp-contract-test sam2-mcp-mock-smoke sam2-mcp-live-health
.PHONY: sam2-backend-run sam2-backend-health sam2-backend-mock-smoke sam2-backend-contract-test
.PHONY: annotation-pdf-sam2-mock-dry-run annotation-pdf-sam2-mock-run annotation-pdf-sam2-validate
.PHONY: mapsam-prepare mapsam-prepare-v0 mapsam-summary

VENV_DIR := $(HOME)/venvs
VENV_NAME := ai_archaeo_topia
VENV_PATH := $(VENV_DIR)/$(VENV_NAME)
PYTHON := python3
PIP := $(VENV_PATH)/bin/pip

help:
	@echo "Available commands:"
	@echo ""
	@echo "Python setup:"
	@echo "  make setup              - Complete setup (system deps, venv, installs packages)"
	@echo "  make venv-dir           - Create ~/venvs directory if it doesn't exist"
	@echo "  make system-deps        - Update system and install GDAL dependencies"
	@echo "  make check-gdal         - Check GDAL version"
	@echo "  make venv               - Create virtual environment"
	@echo "  make activate           - Show activation command"
	@echo "  make install            - Install packages from requirements.txt"
	@echo ""
	@echo "SAM2 MCP service:"
	@echo "  make sam2-mcp-run             - Start the SAM2 MCP service (stdio, standalone)"
	@echo "  make sam2-mcp-health          - Quick health check of SAM2 MCP service"
	@echo "  make sam2-mcp-smoke           - Run SAM2 MCP smoke tests"
	@echo "  make sam2-mcp-mock-smoke      - Run SAM2 MCP mock-mode smoke tests"
	@echo "  make sam2-mcp-contract-test   - Run backend contract validation tests"
	@echo "  make sam2-mcp-live-health     - Check live SAM2 backend health via curl"
	@echo ""
	@echo "SAM2 Backend service:"
	@echo "  make sam2-backend-run             - Start the SAM2 backend HTTP service"
	@echo "  make sam2-backend-health          - Check SAM2 backend health via curl"
	@echo "  make sam2-backend-mock-smoke      - Run SAM2 backend mock-mode smoke test"
	@echo "  make sam2-backend-contract-test   - Run SAM2 backend contract tests"
	@echo ""
	@echo "PDF → SAM2 Annotation Pipeline:"
	@echo "  make annotation-pdf-sam2-mock-dry-run PDF_PATH=/path/to/file.pdf [OUTPUT_ROOT=...] [RUN_ID=...] [SAM2_MCP_URL=mock]"
	@echo "  make annotation-pdf-sam2-mock-run     PDF_PATH=/path/to/file.pdf [OUTPUT_ROOT=...] [RUN_ID=...] [SAM2_MCP_URL=mock]"
	@echo "  make annotation-pdf-sam2-validate     CSV_PATH=/path/to/annotation_results.csv"
	@echo ""
	@echo "SAM2 proposals (from extracted images + CSV):"
	@echo "  make sam2-proposals-run   - Run SAM2 proposals against extracted clips"
	@echo "    INPUT_DIR=...           Path to extraction output (raw/images/, annotations/)"
	@echo "    OUTPUT_DIR=...          Output directory for masks + annotation CSV"
	@echo "    BACKEND_URL=...         SAM2 backend URL (default: http://127.0.0.1:8181)"
	@echo "    MAX_PROPOSALS=...       Max proposals per image (default: 10)"
	@echo "  make sam2-proposals-mock  - Same but with mock backend"
	@echo ""
	@echo "MapSAM dataset preparation:"
	@echo "  make mapsam-prepare              - Convert CVAT COCO → MapSAM dataset"
	@echo "    COCO_JSON=...                  Path to COCO instances_default.json"
	@echo "    IMAGES_DIR=...                 Path to CVAT images directory"
	@echo "    OUTPUT_DIR=...                 Output directory for MapSAM dataset"
	@echo "  make mapsam-prepare-v0           - Quick run with default v0.0.1 paths"
	@echo "  make mapsam-summary              - Print dataset_summary.json"
	@echo ""
	@echo "Quality gates:"
	@echo "  make lint               - Run ruff check + format check"
	@echo "  make test               - Run pytest"
	@echo "  make compile            - Compile all source files"
	@echo ""
	@echo "WSL / CCE:"
	@echo "  make wsl-system-deps    - Install WSL system dependencies (build-essential, cmake, git, python3, pipx)"
	@echo "  make cce-install        - Install Code Context Engine via pipx"
	@echo "  make cce-init           - Initialize CCE (backs up existing opencode.json)"
	@echo "  make cce-uninstall      - Uninstall Code Context Engine via pipx"

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

# ---- WSL / CCE ----

wsl-system-deps:
	@echo "Installing WSL system dependencies..."
	set -euxo pipefail; \
	sudo apt update; \
	sudo apt install -y \
		build-essential \
		cmake \
		git \
		python3 \
		python3-pip \
		pipx; \
	python3 --version; \
	echo "WSL system dependencies installed"

cce-install:
	@echo "Installing Code Context Engine via pipx..."
	set -euxo pipefail; \
	pipx ensurepath; \
	pipx install "code-context-engine[local]"; \
	cce --version; \
	echo "CCE installed. Reload your shell or run: exec \"\${SHELL}\" -l"

cce-init:
	@echo "Initializing CCE in this project..."
	@if [[ ! -f "opencode.json" ]]; then \
		echo "opencode.json not found. Nothing to back up."; \
	elif [[ -f "opencode.json.bak" ]]; then \
		echo "Backing up existing opencode.json to opencode.json.bak"; \
		cp opencode.json opencode.json.bak; \
	else \
		echo "Backing up existing opencode.json to opencode.json.bak"; \
		cp opencode.json opencode.json.bak; \
	fi
	cce init

cce-uninstall:
	@echo "Uninstalling Code Context Engine via pipx..."
	set -euxo pipefail; \
	pipx uninstall code-context-engine; \
	echo "CCE uninstalled"

# ---- MCP / Annotation tools ----

PYTHON := $(VENV_PATH)/bin/python3

lint:
	$(VENV_PATH)/bin/ruff check src/services tests
	$(VENV_PATH)/bin/ruff format --check src/services tests
	$(VENV_PATH)/bin/ruff check src/services/sam2_backend
	$(VENV_PATH)/bin/ruff check src/services/annotation_pipeline

format-check:
	$(VENV_PATH)/bin/ruff format src/services tests src/services/sam2_backend

test:
	$(VENV_PATH)/bin/pytest tests/ -v

compile:
	$(VENV_PATH)/bin/python -m compileall src/services src/services/sam2_backend

# ---- SAM2 MCP service ----

sam2-mcp-run:
	@echo "Starting SAM2 MCP server (stdio)..."
	@echo "Backend: $$(echo ${SAM2_MCP_BACKEND_URL:-mock})"
	$(PYTHON) -m services.sam2_mcp.server

sam2-mcp-health:
	@echo "Running SAM2 MCP health check..."
	@$(PYTHON) -c "\
	import json, os, sys; \
	os.environ.setdefault('SAM2_MCP_BACKEND_URL', 'mock'); \
	from services.sam2_mcp.service import Sam2Service; \
	from services.sam2_mcp.settings import reset_settings; \
	reset_settings(); \
	svc = Sam2Service(); \
	r = svc.health_check(); \
	print(json.dumps(r.model_dump(), indent=2)); \
	sys.exit(0 if r.ok else 1); \
	"

sam2-mcp-smoke:
	@echo "Running SAM2 MCP smoke tests..."
	$(VENV_PATH)/bin/pytest tests/test_sam2_mcp.py -v --tb=short

sam2-mcp-mock-smoke:
	@echo "Running SAM2 MCP mock-mode smoke tests..."
	@$(PYTHON) -c "\
	import json, os, sys; \
	os.environ['SAM2_MCP_BACKEND_URL'] = 'mock'; \
	os.environ['SAM2_MCP_OUTPUT_DIR'] = '/tmp/sam2_mcp_smoke'; \
	from services.sam2_mcp.settings import reset_settings; \
	reset_settings(); \
	from services.sam2_mcp.service import Sam2Service; \
	from services.sam2_mcp.schemas import SegmentBoxInput, SegmentPointsInput, GenerateProposalsInput; \
	svc = Sam2Service(); \
	h = svc.health_check(); \
	assert h.ok, 'health check failed'; \
	print('  health: OK'); \
	print(json.dumps(h.model_dump(), indent=2)); \
	"

sam2-mcp-contract-test:
	@echo "Running SAM2 MCP backend contract tests..."
	$(VENV_PATH)/bin/pytest tests/test_sam2_mcp.py -v --tb=short -k "LiveBackendErrors or BackendSchemas"

sam2-mcp-live-health:
	@echo "Checking live SAM2 backend health..."
	@backend_url="$${SAM2_MCP_BACKEND_URL:-http://127.0.0.1:8181}"; \
	echo "Backend URL: $$backend_url"; \
	curl -sf --max-time 10 "$$backend_url/health" 2>/dev/null | $(PYTHON) -m json.tool && \
		echo "Backend is healthy" || \
		echo "Backend is unreachable or returned an error"

# ---- SAM2 Backend service ----

sam2-backend-run:
	@echo "Starting SAM2 backend service (HTTP)..."
	@echo "Mode: $${SAM2_BACKEND_MODE:-mock}"
	@echo "Host: $${SAM2_BACKEND_HOST:-0.0.0.0}"
	@echo "Port: $${SAM2_BACKEND_PORT:-8181}"
	$(PYTHON) -m services.sam2_backend.main

sam2-backend-health:
	@echo "Checking SAM2 backend health..."
	@backend_url="$${SAM2_BACKEND_URL:-http://127.0.0.1:$${SAM2_BACKEND_PORT:-8181}}"; \
	echo "Backend URL: $$backend_url"; \
	curl -sf --max-time 10 "$$backend_url/health" 2>/dev/null | $(PYTHON) -m json.tool && \
		echo "Backend is healthy" || \
		echo "Backend is unreachable or returned an error"

sam2-backend-mock-smoke:
	@echo "Running SAM2 backend mock-mode smoke tests..."
	@$(PYTHON) -c "\
	import json, os, sys; \
	os.environ['SAM2_BACKEND_MODE'] = 'mock'; \
	os.environ['SAM2_BACKEND_OUTPUT_DIR'] = '/tmp/sam2_backend_smoke'; \
	from services.sam2_backend.settings import reset_settings; \
	reset_settings(); \
	from services.sam2_backend.predictor import SAM2PredictorBackend; \
	p = SAM2PredictorBackend(); \
	p.ensure_loaded(); \
	assert p.is_loaded, 'model should be loaded in mock mode'; \
	assert p.device == 'mock', 'device should be mock'; \
	print('  predictor: OK (mode=mock)'); \
	print(json.dumps({'ok': True, 'device': p.device, 'loaded': p.is_loaded}, indent=2)); \
	"

sam2-backend-contract-test:
	@echo "Running SAM2 backend contract tests..."
	$(VENV_PATH)/bin/pytest tests/test_sam2_backend.py -v --tb=short

# ---- PDF → SAM2 Annotation Pipeline ----

annotation-pdf-sam2-mock-dry-run:
	@if [ -z "${PDF_PATH}" ]; then \
		echo "Usage: make annotation-pdf-sam2-mock-dry-run PDF_PATH=/path/to/file.pdf [OUTPUT_ROOT=...] [RUN_ID=...] [SAM2_MCP_URL=mock]"; \
		exit 1; \
	fi
	@set -euxo pipefail; \
	output_root="${OUTPUT_ROOT:-./data/annotations/runs}"; \
	run_id="${RUN_ID:-dry_run_$$(date +%Y%m%d_%H%M%S)}"; \
	sam2_mcp_url="${SAM2_MCP_URL:-mock}"; \
	echo "PDF: ${PDF_PATH}"; \
	echo "Output: ${output_root}"; \
	echo "Run ID: ${run_id}"; \
	echo "SAM2 MCP URL: ${sam2_mcp_url}"; \
	$(PYTHON) -m services.annotation_pipeline.pdf_sam2_runner \
		--pdf "${PDF_PATH}" \
		--output-root "${output_root}" \
		--run-id "${run_id}" \
		--sam2-mcp-url "${sam2_mcp_url}" \
		--sam2-mode mock \
		--dry-run true \
		--log-level INFO

annotation-pdf-sam2-mock-run:
	@if [ -z "${PDF_PATH}" ]; then \
		echo "Usage: make annotation-pdf-sam2-mock-run PDF_PATH=/path/to/file.pdf [OUTPUT_ROOT=...] [RUN_ID=...] [SAM2_MCP_URL=mock]"; \
		exit 1; \
	fi
	@set -euxo pipefail; \
	output_root="${OUTPUT_ROOT:-./data/annotations/runs}"; \
	run_id="${RUN_ID:-run_$$(date +%Y%m%d_%H%M%S)}"; \
	sam2_mcp_url="${SAM2_MCP_URL:-mock}"; \
	echo "PDF: ${PDF_PATH}"; \
	echo "Output: ${output_root}"; \
	echo "Run ID: ${run_id}"; \
	echo "SAM2 MCP URL: ${sam2_mcp_url}"; \
	$(PYTHON) -m services.annotation_pipeline.pdf_sam2_runner \
		--pdf "${PDF_PATH}" \
		--output-root "${output_root}" \
		--run-id "${run_id}" \
		--sam2-mcp-url "${sam2_mcp_url}" \
		--sam2-mode mock \
		--dry-run false \
		--log-level INFO

annotation-pdf-sam2-validate:
	@if [ -z "${CSV_PATH}" ]; then \
		echo "Usage: make annotation-pdf-sam2-validate CSV_PATH=/path/to/annotation_results.csv"; \
		exit 1; \
	fi
	@set -euxo pipefail; \
	echo "Validating: ${CSV_PATH}"; \
	$(PYTHON) -m src.validate_annotation_run --csv "${CSV_PATH}"

# ---- SAM2 Proposals from extracted images + CSV ----

sam2-proposals-run:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Usage: make sam2-proposals-run INPUT_DIR=path/to/extracted [OUTPUT_DIR=...] [BACKEND_URL=...] [MAX_PROPOSALS=10]"; \
		exit 1; \
	fi
	@set -euo pipefail; \
	output_dir="$(OUTPUT_DIR)"; \
	if [ -z "$$output_dir" ]; then output_dir="$(INPUT_DIR)/automated_mcp"; fi; \
	backend_url="$(BACKEND_URL)"; \
	if [ -z "$$backend_url" ]; then backend_url="http://127.0.0.1:8181"; fi; \
	max_proposals="$(MAX_PROPOSALS)"; \
	if [ -z "$$max_proposals" ]; then max_proposals="10"; fi; \
	echo "Input:    $(INPUT_DIR)"; \
	echo "Output:   $$output_dir"; \
	echo "Backend:  $$backend_url"; \
	echo "Max props: $$max_proposals"; \
	$(PYTHON) src/sam2_mcp_proposal_run.py \
		--input-dir "$(INPUT_DIR)" \
		--output-dir "$$output_dir" \
		--backend-url "$$backend_url" \
		--max-proposals "$$max_proposals"

sam2-proposals-mock:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Usage: make sam2-proposals-mock INPUT_DIR=path/to/extracted [OUTPUT_DIR=...] [MAX_PROPOSALS=10]"; \
		exit 1; \
	fi
	@set -euo pipefail; \
	output_dir="$(OUTPUT_DIR)"; \
	if [ -z "$$output_dir" ]; then output_dir="$(INPUT_DIR)/automated_mcp_mock"; fi; \
	max_proposals="$(MAX_PROPOSALS)"; \
	if [ -z "$$max_proposals" ]; then max_proposals="10"; fi; \
	echo "Input:    $(INPUT_DIR)"; \
	echo "Output:   $$output_dir"; \
	echo "Backend:  mock"; \
	echo "Max props: $$max_proposals"; \
	$(PYTHON) src/sam2_mcp_proposal_run.py \
		--input-dir "$(INPUT_DIR)" \
		--output-dir "$$output_dir" \
		--backend-url mock \
		--max-proposals "$$max_proposals"

# ---- MapSAM dataset preparation ----

mapsam-prepare:
	@if [ -z "$(COCO_JSON)" ]; then \
		echo "Usage: make mapsam-prepare COCO_JSON=path/to/instances_default.json IMAGES_DIR=path/to/images [OUTPUT_DIR=...]"; \
		exit 1; \
	fi
	@set -euo pipefail; \
	output_dir="$(OUTPUT_DIR)"; \
	if [ -z "$$output_dir" ]; then output_dir="data/curated/datasets/mapsam_v0"; fi; \
	echo "COCO JSON:  $(COCO_JSON)"; \
	echo "Images dir: $(IMAGES_DIR)"; \
	echo "Output dir: $$output_dir"; \
	$(PYTHON) -m src.archeo_topia.datasets.prepare_mapsam_coco \
		--coco-json "$(COCO_JSON)" \
		--images-dir "$(IMAGES_DIR)" \
		--output-dir "$$output_dir" \
		--positive-label mound \
		--ignore-label uncertain_ignore \
		--hard-negative-label hard_negative_symbol \
		--split-by sheet

mapsam-prepare-v0:
	@$(MAKE) mapsam-prepare \
		COCO_JSON=data/curated/datasets/cvat/v0.0.1/annotations/instances_default.json \
		IMAGES_DIR=data/curated/datasets/cvat/v0.0.1/images/default \
		OUTPUT_DIR=data/curated/datasets/mapsam_v0

mapsam-summary:
	@summary="data/curated/datasets/mapsam_v0/metadata/dataset_summary.json"; \
	if [ -f "$$summary" ]; then \
		$(PYTHON) -m json.tool "$$summary"; \
	else \
		echo "Summary not found at $$summary"; \
		echo "Run: make mapsam-prepare-v0"; \
		exit 1; \
	fi
