SHELL := /usr/bin/env bash

.PHONY: help venv-dir venv activate install system-deps check-gdal setup
.PHONY: cvat-clone cvat-init-env cvat-up cvat-up-serverless cvat-down cvat-down-volumes
.PHONY: cvat-health cvat-ps cvat-logs cvat-superuser cvat-reset
.PHONY: cvat-install-nuctl cvat-deploy-sam2-cpu cvat-deploy-sam2-gpu
.PHONY: cvat-undeploy-sam2 cvat-functions cvat-open
.PHONY: wsl-system-deps cce-install cce-init cce-uninstall
.PHONY: cvat-mcp-run cvat-mcp-health cvat-mcp-smoke
.PHONY: annotation-dry-run annotation-run
.PHONY: lint format-check test compile
.PHONY: sam2-mcp-run sam2-mcp-health sam2-mcp-smoke sam2-mcp-contract-test sam2-mcp-mock-smoke sam2-mcp-live-health
.PHONY: sam2-backend-run sam2-backend-health sam2-backend-mock-smoke sam2-backend-contract-test
.PHONY: annotation-pdf-sam2-mock-dry-run annotation-pdf-sam2-mock-run annotation-pdf-sam2-validate

VENV_DIR := $(HOME)/venvs
VENV_NAME := ai_archaeo_topia
VENV_PATH := $(VENV_DIR)/$(VENV_NAME)
PYTHON := python3
PIP := $(VENV_PATH)/bin/pip

# ---- CVAT ----
CVAT_DIR ?= tools/cvat
CVAT_REPO ?= https://github.com/openvinotoolkit/cvat.git
CVAT_ENV ?= .env.cvat

# Nuclio version — read from CVAT compose at runtime (after clone)
NUCTL_VERSION ?= 1.15.9
NUCTL_DIR := /tmp/cvat-nuctl
NUCTL_BIN_NAME := nuctl-$(NUCTL_VERSION)-linux-amd64
NUCTL_BIN_PATH := $(NUCTL_DIR)/$(NUCTL_BIN_NAME)

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
	@echo "CVAT infrastructure:"
	@echo "  make cvat-clone         - Clone CVAT repo into tools/cvat/"
	@echo "  make cvat-init-env      - Create .env.cvat from template"
	@echo "  make cvat-up            - Start CVAT (no serverless)"
	@echo "  make cvat-up-serverless - Start CVAT + Nuclio (for SAM2 auto-annotation)"
	@echo "  make cvat-down          - Stop CVAT containers"
	@echo "  make cvat-down-volumes  - Stop + remove volumes (destructive)"
	@echo "  make cvat-health        - Wait for CVAT readiness"
	@echo "  make cvat-ps            - List CVAT containers"
	@echo "  make cvat-logs          - Follow CVAT logs"
	@echo "  make cvat-superuser     - Create CVAT superuser"
	@echo "  make cvat-reset         - Wipe all CVAT data (requires confirmation)"
	@echo "  make cvat-open          - Print CVAT URL"
	@echo ""
	@echo "CVAT serverless / SAM2:"
	@echo "  make cvat-install-nuctl - Download nuctl matching CVAT's Nuclio version"
	@echo "  make cvat-deploy-sam2-cpu - Deploy SAM2 CPU function"
	@echo "  make cvat-deploy-sam2-gpu - Deploy SAM2 GPU function (opt-in)"
	@echo "  make cvat-undeploy-sam2 - Remove SAM2 function"
	@echo "  make cvat-functions     - List deployed Nuclio functions"
	@echo ""
	@echo "MCP / Annotation pipeline:"
	@echo "  make cvat-mcp-run       - Start the CVAT SAM2 MCP server (stdio)"
	@echo "  make cvat-mcp-health    - Quick health check of CVAT/SAM2 connectivity"
	@echo "  make cvat-mcp-smoke     - Run pre-flight smoke tests (no CVAT required)"
	@echo "  make sam2-mcp-run             - Start the SAM2 MCP service (stdio, standalone)"
	@echo "  make sam2-mcp-health          - Quick health check of SAM2 MCP service"
	@echo "  make sam2-mcp-smoke           - Run SAM2 MCP smoke tests"
	@echo "  make sam2-mcp-mock-smoke      - Run SAM2 MCP mock-mode smoke tests"
	@echo "  make sam2-mcp-contract-test   - Run backend contract validation tests"
	@echo "  make sam2-mcp-live-health     - Check live SAM2 backend health via curl"
	@echo "  make sam2-backend-run             - Start the SAM2 backend HTTP service"
	@echo "  make sam2-backend-health          - Check SAM2 backend health via curl"
	@echo "  make sam2-backend-mock-smoke      - Run SAM2 backend mock-mode smoke test"
	@echo "  make sam2-backend-contract-test   - Run SAM2 backend contract tests"
	@echo "  make annotation-dry-run - Dry-run the annotation pipeline (INPUT_DIR=...)"
	@echo "  make annotation-run     - Run the full annotation pipeline (INPUT_DIR=...)"
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

# ---- CVAT Infrastructure ----

cvat-clone:
	@if [[ ! -d "$(CVAT_DIR)/.git" ]]; then \
		mkdir -p "$(dir $(CVAT_DIR))"; \
		git clone --depth 1 --single-branch "$(CVAT_REPO)" "$(CVAT_DIR)"; \
		echo "Cloned CVAT to $(CVAT_DIR)"; \
	else \
		echo "CVAT already cloned at $(CVAT_DIR)"; \
	fi

cvat-init-env:
	@if [[ ! -f "$(CVAT_ENV)" ]]; then \
		cp "$(CVAT_ENV).example" "$(CVAT_ENV)"; \
		echo "Created $(CVAT_ENV) from template"; \
	else \
		echo "$(CVAT_ENV) already exists"; \
	fi

cvat-up: cvat-clone cvat-init-env
	@if ! grep -q 'CVAT_PORT' "$(CVAT_ENV)" 2>/dev/null; then \
		echo "CVAT_PORT not set in $(CVAT_ENV)"; \
		exit 1; \
	fi
	cd "$(CVAT_DIR)" && \
		docker compose --env-file "$(CURDIR)/$(CVAT_ENV)" up -d
	@echo "CVAT starting... Run 'make cvat-health' to wait for readiness."

cvat-up-serverless: cvat-clone cvat-init-env
	@if ! grep -q 'CVAT_PORT' "$(CVAT_ENV)" 2>/dev/null; then \
		echo "CVAT_PORT not set in $(CVAT_ENV)"; \
		exit 1; \
	fi
	cd "$(CVAT_DIR)" && \
		docker compose \
			--env-file "$(CURDIR)/$(CVAT_ENV)" \
			-f docker-compose.yml \
			-f components/serverless/docker-compose.serverless.yml \
			up -d
	@echo "CVAT + serverless starting... Run 'make cvat-health' to wait for readiness."

cvat-down:
	@if [[ -d "$(CVAT_DIR)" ]]; then \
		cd "$(CVAT_DIR)" && \
		docker compose --env-file "$(CURDIR)/$(CVAT_ENV)" down 2>/dev/null || true; \
	fi

cvat-down-volumes:
	@if [[ -d "$(CVAT_DIR)" ]]; then \
		cd "$(CVAT_DIR)" && \
		docker compose --env-file "$(CURDIR)/$(CVAT_ENV)" down -v 2>/dev/null || true; \
		echo "CVAT volumes removed (destructive)"; \
	fi

cvat-health: cvat-clone cvat-init-env
	@cvat_port="$$(grep -m1 'CVAT_PORT' "$(CVAT_ENV)" 2>/dev/null | cut -d= -f2 || echo 9090)"; \
	echo "Waiting for CVAT to be ready on port $$cvat_port ..."; \
	for i in $$(seq 1 60); do \
		if curl -sf "http://localhost:$$cvat_port" >/dev/null 2>&1; then \
			echo "CVAT is ready"; \
			exit 0; \
		fi; \
		echo "  waiting... ($$i/60)"; \
		sleep 2; \
	done; \
	echo "CVAT did not become ready in 2 minutes. Run 'make cvat-logs' to diagnose." >&2; \
	exit 1

cvat-ps:
	@if [[ -d "$(CVAT_DIR)" ]]; then \
		cd "$(CVAT_DIR)" && \
		docker compose --env-file "$(CURDIR)/$(CVAT_ENV)" ps; \
	fi

cvat-logs:
	@if [[ -d "$(CVAT_DIR)" ]]; then \
		cd "$(CVAT_DIR)" && \
		docker compose --env-file "$(CURDIR)/$(CVAT_ENV)" logs -f --tail=200; \
	fi

cvat-superuser: cvat-clone
	@if [[ ! -d "$(CVAT_DIR)" ]]; then \
		echo "CVAT not cloned. Run 'make cvat-clone' first."; \
		exit 1; \
	fi
	@echo "Creating superuser in CVAT..."
	cd "$(CVAT_DIR)" && \
		docker exec cvat_server bash -ic 'python3 ~/manage.py createsuperuser --noinput' 2>/dev/null || \
		docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

cvat-reset:
	@echo "WARNING: This will remove all CVAT data (annotations, tasks, database)." >&2
	@read -p "Type 'RESET' to confirm: " confirm; \
	if [[ "$$confirm" != "RESET" ]]; then \
		echo "Aborted."; \
		exit 1; \
	fi
	cd "$(CVAT_DIR)" && \
		docker compose --env-file "$(CURDIR)/$(CVAT_ENV)" down -v 2>/dev/null || true
	@echo "CVAT reset complete. Run 'make cvat-up-serverless' to restart."

cvat-open: cvat-init-env
	@port="$$(grep -m1 'CVAT_PORT' "$(CVAT_ENV)" 2>/dev/null | cut -d= -f2 || echo 9090)"; \
	echo "Open CVAT at: http://localhost:$$port"

# ---- CVAT Serverless / SAM2 ----

cvat-install-nuctl: cvat-clone
	@nuctl_version="$(NUCTL_VERSION)"; \
	if [[ -d "$(CVAT_DIR)/components/serverless" ]] && \
	   grep -q 'quay.io/nuclio/dashboard' "$(CVAT_DIR)/components/serverless/docker-compose.serverless.yml" 2>/dev/null; then \
		nuctl_version=$$(grep -m1 'quay.io/nuclio/dashboard' "$(CVAT_DIR)/components/serverless/docker-compose.serverless.yml" | sed -E 's/.*:(1\.[0-9]+\.[0-9]+).*/\1/'); \
	fi; \
	nuctl_bin="$(NUCTL_DIR)/nuctl-$${nuctl_version}-linux-amd64"; \
	if [[ ! -f "$${nuctl_bin}" ]]; then \
		mkdir -p "$(NUCTL_DIR)"; \
		curl -sL "https://github.com/nuclio/nuclio/releases/download/$${nuctl_version}/nuctl-$${nuctl_version}-linux-amd64" \
			-o "$${nuctl_bin}"; \
		chmod +x "$${nuctl_bin}"; \
		echo "Installed nuctl $${nuctl_version} to $${nuctl_bin}"; \
		echo "To make it permanent: sudo cp $${nuctl_bin} /usr/local/bin/nuctl"; \
	else \
		echo "nuctl already installed at $${nuctl_bin}"; \
	fi

cvat-deploy-sam2-cpu: cvat-clone cvat-init-env cvat-install-nuctl cvat-up-serverless
	@nuctl_bin=$$(ls "$(NUCTL_DIR)"/nuctl-*-linux-amd64 2>/dev/null | head -1); \
	if [[ -z "$${nuctl_bin}" ]]; then \
		echo "nuctl not found. Run 'make cvat-install-nuctl' first."; \
		exit 1; \
	fi; \
	echo "Deploying SAM2 CPU function..."; \
	cd "$(CVAT_DIR)" && \
		./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam/nuclio; \
	echo "SAM2 deployed. Check with 'make cvat-functions'"

cvat-deploy-sam2-gpu: cvat-clone cvat-init-env cvat-install-nuctl cvat-up-serverless
	@nuctl_bin=$$(ls "$(NUCTL_DIR)"/nuctl-*-linux-amd64 2>/dev/null | head -1); \
	if [[ -z "$${nuctl_bin}" ]]; then \
		echo "nuctl not found. Run 'make cvat-install-nuctl' first."; \
		exit 1; \
	fi; \
	echo "Deploying SAM2 GPU function..."; \
	if [[ ! -f "$(CVAT_ENV)" ]] || ! grep -q 'CVAT_GPU_ENABLED=1' "$(CVAT_ENV)" 2>/dev/null; then \
		echo "Set CVAT_GPU_ENABLED=1 in $(CVAT_ENV) before deploying GPU functions."; \
		exit 1; \
	fi; \
	cd "$(CVAT_DIR)" && \
		"$$nuctl_bin" deploy --project-name cvat \
			--path serverless/pytorch/facebookresearch/sam/nuclio \
			--file serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml \
			--platform local \
			--resource-limit nvidia.com/gpu=1 \
			--env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
			--env CVAT_FUNCTIONS_REDIS_PORT=6666 \
			--platform-config '{"attributes": {"network": "cvat_cvat"}}'; \
	echo "SAM2 GPU deployed. Check with 'make cvat-functions'"

cvat-undeploy-sam2: cvat-install-nuctl
	@nuctl_bin=$$(ls "$(NUCTL_DIR)"/nuctl-*-linux-amd64 2>/dev/null | head -1); \
	if [[ -z "$${nuctl_bin}" ]]; then \
		echo "nuctl not found. Run 'make cvat-install-nuctl' first."; \
		exit 1; \
	fi; \
	echo "Undeploying SAM2 function..."; \
	cd "$(CVAT_DIR)" && \
		"$$nuctl_bin" delete function sam2 --project-name cvat --platform local 2>/dev/null || true; \
	echo "SAM2 function removed."

cvat-functions: cvat-install-nuctl
	@nuctl_bin=$$(ls "$(NUCTL_DIR)"/nuctl-*-linux-amd64 2>/dev/null | head -1); \
	if [[ -z "$${nuctl_bin}" ]]; then \
		echo "nuctl not found. Run 'make cvat-install-nuctl' first."; \
		exit 1; \
	fi; \
	cd "$(CVAT_DIR)" && \
		"$$nuctl_bin" get functions --platform local

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
	$(VENV_PATH)/bin/ruff check src/cvat_sam2_mcp src/services tests
	$(VENV_PATH)/bin/ruff format --check src/cvat_sam2_mcp src/services tests
	$(VENV_PATH)/bin/ruff check src/services/sam2_backend
	$(VENV_PATH)/bin/ruff check src/services/annotation_pipeline

format-check:
	$(VENV_PATH)/bin/ruff format src/cvat_sam2_mcp src/services tests src/services/sam2_backend

test:
	$(VENV_PATH)/bin/pytest tests/ -v

compile:
	$(VENV_PATH)/bin/python -m compileall src/cvat_sam2_mcp src/services src/services/sam2_backend

cvat-mcp-run:
	@echo "Starting CVAT SAM2 MCP server (stdio)..."
	@echo "Connect opencode via the cvat-sam2 MCP config entry."
	$(PYTHON) -m cvat_sam2_mcp.server

cvat-mcp-health:
	@echo "Running MCP smoke health check..."
	@$(PYTHON) -c "\
	import json, sys; \
	from cvat_sam2_mcp.cvat_client import CvatClient; \
	from cvat_sam2_mcp.settings import get_settings; \
	s = get_settings(); \
	c = CvatClient(s.cvat_base_url, s.cvat_username, s.cvat_password); \
	r = c.health(); \
	print(json.dumps(r, indent=2)); \
	ok = r.get('reachable') or r.get('serverless_reachable'); \
	sys.exit(0 if ok else 1); \
	" 2>/dev/null || echo "CVAT/SAM2 not reachable (expected if not running)"

cvat-mcp-smoke:
	@echo "Running MCP pre-flight smoke tests..."
	$(VENV_PATH)/bin/pytest tests/test_cvat_sam2_mcp.py -v --tb=short

annotation-dry-run:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Usage: make annotation-dry-run INPUT_DIR=/path/to/images PROTOCOL=annotation_protocols/archaeology_symbols_v1.yaml"; \
		exit 1; \
	fi
	@set -euxo pipefail; \
	protocol="$(PROTOCOL)"; \
	if [ -z "$$protocol" ]; then protocol="annotation_protocols/archaeology_symbols_v1.yaml"; fi; \
	$(PYTHON) -c "\
	import json, sys; \
	from cvat_sam2_mcp.annotation_runner import start_run; \
	r = start_run('$$protocol', '$(INPUT_DIR)', dry_run=True); \
	print(json.dumps(r, indent=2)); \
	"

annotation-run:
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "Usage: make annotation-run INPUT_DIR=/path/to/images PROTOCOL=annotation_protocols/archaeology_symbols_v1.yaml"; \
		exit 1; \
	fi
	@set -euxo pipefail; \
	protocol="$(PROTOCOL)"; \
	if [ -z "$$protocol" ]; then protocol="annotation_protocols/archaeology_symbols_v1.yaml"; fi; \
	$(PYTHON) -c "\
	import json, sys; \
	from cvat_sam2_mcp.annotation_runner import start_run; \
	r = start_run('$$protocol', '$(INPUT_DIR)', dry_run=False); \
	print(json.dumps(r, indent=2)); \
	"

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