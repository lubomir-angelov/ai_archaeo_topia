SHELL := /usr/bin/env bash

.PHONY: help venv-dir venv activate install system-deps check-gdal setup
.PHONY: cvat-clone cvat-init-env cvat-up cvat-up-serverless cvat-down cvat-down-volumes
.PHONY: cvat-health cvat-ps cvat-logs cvat-superuser cvat-reset
.PHONY: cvat-install-nuctl cvat-deploy-sam2-cpu cvat-deploy-sam2-gpu
.PHONY: cvat-undeploy-sam2 cvat-functions cvat-open

VENV_DIR := $(HOME)/venvs
VENV_NAME := ai_archaeotopia
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

cvat-deploy-sam2-cpu: cvat-clone cvat-install-nuctl
	@nuctl_bin=$$(ls "$(NUCTL_DIR)"/nuctl-*-linux-amd64 2>/dev/null | head -1); \
	if [[ -z "$${nuctl_bin}" ]]; then \
		echo "nuctl not found. Run 'make cvat-install-nuctl' first."; \
		exit 1; \
	fi
	@echo "Deploying SAM2 CPU function..."
	cd "$(CVAT_DIR)" && \
		./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam/nuclio
	@echo "SAM2 deployed. Check with 'make cvat-functions'"

cvat-deploy-sam2-gpu: cvat-clone cvat-install-nuctl
	@nuctl_bin=$$(ls "$(NUCTL_DIR)"/nuctl-*-linux-amd64 2>/dev/null | head -1); \
	if [[ -z "$${nuctl_bin}" ]]; then \
		echo "nuctl not found. Run 'make cvat-install-nuctl' first."; \
		exit 1; \
	fi
	@echo "Deploying SAM2 GPU function..."
	@if [[ ! -f "$(CVAT_ENV)" ]] || ! grep -q 'CVAT_GPU_ENABLED=1' "$(CVAT_ENV)" 2>/dev/null; then \
		echo "Set CVAT_GPU_ENABLED=1 in $(CVAT_ENV) before deploying GPU functions."; \
		exit 1; \
	fi
	cd "$(CVAT_DIR)" && \
		"$${nuctl_bin}" deploy --project-name cvat \
			--path serverless/pytorch/facebookresearch/sam/nuclio \
			--file serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml \
			--platform local \
			--resource-limit nvidia.com/gpu=1 \
			--env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
			--env CVAT_FUNCTIONS_REDIS_PORT=6666 \
			--trigger-http \
			--attributes '{"network": "cvat_cvat"}' \
			--run-arguments "--privileged"
	@echo "SAM2 GPU deployed. Check with 'make cvat-functions'"

cvat-undeploy-sam2: cvat-install-nuctl
	@nuctl_bin=$$(ls "$(NUCTL_DIR)"/nuctl-*-linux-amd64 2>/dev/null | head -1); \
	if [[ -z "$${nuctl_bin}" ]]; then \
		echo "nuctl not found. Run 'make cvat-install-nuctl' first."; \
		exit 1; \
	fi
	@echo "Undeploying SAM2 function..."
	cd "$(CVAT_DIR)" && \
		"$${nuctl_bin}" delete function sam2 --project-name cvat --platform local 2>/dev/null || true
	@echo "SAM2 function removed."

cvat-functions: cvat-install-nuctl
	@nuctl_bin=$$(ls "$(NUCTL_DIR)"/nuctl-*-linux-amd64 2>/dev/null | head -1); \
	if [[ -z "$${nuctl_bin}" ]]; then \
		echo "nuctl not found. Run 'make cvat-install-nuctl' first."; \
		exit 1; \
	fi
	cd "$(CVAT_DIR)" && \
		"$${nuctl_bin}" get functions --platform local