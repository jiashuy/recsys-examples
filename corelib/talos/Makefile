# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Talos — repo task runner (development only).
#
# End users do not need this file: they `pip install talos`. This is for
# building wheels and installing them into a local environment while working
# on the package.
#
#   make build                          wheel for the current host platform
#   make build ALL=1                    wheels for every supported platform
#   make build PLATFORM=aarch64-linux   wheel for one specific platform
#   make install-wheel                  build, then pip install into this env
#   make install-skills DIR=<project>   the above, then link the skills into
#                                       a project (DIR defaults to .)
#   make clean                          remove build artifacts
#
# Supported PLATFORM keys (see setup.py:PLATFORMS):
#   x86_64-linux  aarch64-linux  x86_64-macos  aarch64-macos
#
# Each build downloads the matching veloq binary + skill tarball from GitHub
# Releases; the pinned version lives in pyproject.toml under [tool.talos].

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
DIST_DIR  := $(REPO_ROOT)/dist
PYTHON    ?= python3
AGENTS    ?= claude codex
DIR       ?= .

ALL_PLATFORMS := x86_64-linux aarch64-linux x86_64-macos aarch64-macos

.DEFAULT_GOAL := help
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
SHELL := bash
.PHONY: help build install-wheel install-skills clean

help:
	@sed -n '/^# Talos — repo task runner/,/^$$/p' $(MAKEFILE_LIST) | sed 's/^# \?//'

# Two environment facts decide the pip flags, both probed up front so the
# install runs once instead of failing into a retry:
#   --user                    install into the user site, leaving the system
#                             Python alone. Invalid inside a venv/conda (and
#                             pointless there — the env is already private).
#   --break-system-packages   PEP 668 distros (Debian/Ubuntu) refuse to install
#                             into the system Python without it.
define PIP_FLAGS
	pip_flags=""
	if $(PYTHON) -c 'import os,sys; sys.exit(1 if sys.prefix != sys.base_prefix or os.environ.get("CONDA_PREFIX") else 0)'; then
	  pip_flags="--user"
	  if $(PYTHON) -c 'import os,sys,sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")) else 1)'; then
	    pip_flags="$$pip_flags --break-system-packages"
	  fi
	fi
endef

define PIP_INSTALL
	$(PIP_FLAGS)
	echo ">> pip install $$pip_flags $(1)"
	$(PYTHON) -m pip install $$pip_flags $(1)
endef

# Remove veloq artifacts fetched into the source tree so each platform build
# starts from a known-empty state. setup.py refills them.
define CLEAN_FETCHED
	rm -f  $(REPO_ROOT)/skills/bin/veloq $(REPO_ROOT)/skills/bin/veloq.exe
	rm -f  $(REPO_ROOT)/skills/nsys_profile_analysis/SKILL.md
	rm -rf $(REPO_ROOT)/skills/nsys_profile_analysis/references
	rm -f  $(REPO_ROOT)/skills/ncu_profile_analysis/SKILL.md
	rm -rf $(REPO_ROOT)/skills/ncu_profile_analysis/references
	rm -rf $(REPO_ROOT)/skills/pytorch_profile_analysis
	rm -rf $(REPO_ROOT)/build
endef

clean:
	@rm -rf $(DIST_DIR) $(REPO_ROOT)/build
	find $(REPO_ROOT) -type d -name '*.egg-info' -exec rm -rf {} +
	find $(REPO_ROOT) -type d -name '__pycache__' -exec rm -rf {} +
	$(CLEAN_FETCHED)
	echo ">> cleaned"

build:
	@# Probe with `-m build --version`, not `import build`: a stale build/
	@# directory resolves as a namespace package and fools the import.
	if ! $(PYTHON) -m build --version >/dev/null 2>&1; then
	  echo ">> installing PyPA build frontend"
	  $(call PIP_INSTALL,--upgrade build)
	fi
	echo ">> cleaning previous artifacts"
	rm -rf $(DIST_DIR) $(REPO_ROOT)/build
	find $(REPO_ROOT) -type d -name '*.egg-info' -exec rm -rf {} +
	find $(REPO_ROOT) -type d -name '__pycache__' -exec rm -rf {} +
	mkdir -p $(DIST_DIR)
	if [ -n "$(ALL)" ]; then
	  for plat in $(ALL_PLATFORMS); do
	    echo ""
	    echo ">> ── building for VELOQ_PLATFORM=$$plat ──"
	    $(CLEAN_FETCHED)
	    VELOQ_PLATFORM=$$plat $(PYTHON) -m build --wheel --outdir $(DIST_DIR) $(REPO_ROOT)
	  done
	elif [ -n "$(PLATFORM)" ]; then
	  echo ">> ── building for VELOQ_PLATFORM=$(PLATFORM) ──"
	  $(CLEAN_FETCHED)
	  VELOQ_PLATFORM=$(PLATFORM) $(PYTHON) -m build --wheel --outdir $(DIST_DIR) $(REPO_ROOT)
	else
	  echo ">> building for current host (auto-detected platform)"
	  $(CLEAN_FETCHED)
	  $(PYTHON) -m build --wheel --outdir $(DIST_DIR) $(REPO_ROOT)
	fi
	echo ""
	echo ">> artifacts:"
	ls -1 $(DIST_DIR)

install-wheel: build
	@# A host build leaves exactly one wheel in a freshly-cleaned dist/.
	shopt -s nullglob
	wheels=($(DIST_DIR)/talos-*.whl)
	if [ $${#wheels[@]} -ne 1 ]; then
	  echo "expected exactly one wheel in $(DIST_DIR), found $${#wheels[@]}" >&2
	  printf '  %s\n' "$${wheels[@]}" >&2
	  exit 2
	fi
	wheel="$${wheels[0]}"
	echo ">> installing $$wheel"
	$(call PIP_INSTALL,--force-reinstall --no-deps "$$wheel")
	echo ""
	echo "installed. Link the skills into a project with:"
	echo "  make install-skills DIR=/path/to/your/project"

install-skills: install-wheel
	@cd $(DIR) && $(PYTHON) -m talos.install_skills $(AGENTS)
