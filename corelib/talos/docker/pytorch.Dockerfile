# Talos development image: PyTorch + the NVIDIA profilers the skills drive +
# the coding agents that run them.
#
#   docker build -f pytorch.Dockerfile -t talos .
#
# Everything is fetched from public sources at build time, so this builds
# anywhere with network access — no pre-downloaded installers required.

ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg2 git vim gdb sudo sqlite3 \
        ripgrep bubblewrap socat xz-utils jq \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Nsight Systems (nsys) and Nsight Compute (ncu), from NVIDIA's public
# devtools apt repository.
#
# Pin a specific release by overriding the build args, e.g.
#   --build-arg NSYS_PACKAGE=nsight-systems-cli-2026.2.1
# Leave them unset to track the repo's current metapackage.
# ---------------------------------------------------------------------------
ARG NSYS_PACKAGE=nsight-systems-cli
ARG NCU_PACKAGE=nsight-compute

RUN set -eux; \
    . /etc/os-release; \
    ubuntu_ver="$(echo "${VERSION_ID}" | tr -d '.')"; \
    arch="$(dpkg --print-architecture)"; \
    base="https://developer.download.nvidia.com/devtools/repos"; \
    # Fall back to the newest repo we know exists if this exact release has
    # no directory published yet.
    repo="${base}/ubuntu${ubuntu_ver}/${arch}"; \
    if ! curl -fsI "${repo}/nvidia.pub" >/dev/null 2>&1; then \
        for fallback in 2404 2204; do \
            if curl -fsI "${base}/ubuntu${fallback}/${arch}/nvidia.pub" >/dev/null 2>&1; then \
                repo="${base}/ubuntu${fallback}/${arch}"; break; \
            fi; \
        done; \
    fi; \
    echo ">>> nsight repo: ${repo}"; \
    curl -fsSL "${repo}/nvidia.pub" | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] ${repo} /" \
        > /etc/apt/sources.list.d/nvidia-devtools.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends "${NSYS_PACKAGE}" "${NCU_PACKAGE}"; \
    apt-get clean; rm -rf /var/lib/apt/lists/*; \
    chmod -R a+rX /opt/nvidia; \
    # The packages ship target binaries for several architectures under the
    # same name, so pick the candidate that actually runs on this host.
    for c in /opt/nvidia/nsight-systems-cli/*/target-linux-*/nsys \
             /opt/nvidia/nsight-systems/*/target-linux-*/nsys; do \
        if [ -x "$c" ] && "$c" --version >/dev/null 2>&1; then \
            ln -sf "$c" /usr/local/bin/nsys; break; \
        fi; \
    done; \
    for c in /opt/nvidia/nsight-compute/*/ncu \
             /opt/nvidia/nsight-compute/*/target/*/ncu; do \
        if [ -x "$c" ] && "$c" --version >/dev/null 2>&1; then \
            ln -sf "$c" /usr/local/bin/ncu; break; \
        fi; \
    done; \
    nsys --version; \
    ncu --version

# ---------------------------------------------------------------------------
# Node from the official tarball: distro packages are far behind what the
# agents require (Ubuntu 22.04 ships Node 12; they need 18+).
# ---------------------------------------------------------------------------
ARG NODE_VERSION=22.20.0
RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
        amd64) node_arch=x64 ;; \
        arm64) node_arch=arm64 ;; \
        *) echo "unsupported architecture for the Node tarball" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" \
        | tar -xJ -C /usr/local --strip-components=1; \
    node --version; npm --version

# ---------------------------------------------------------------------------
# Coding agents. Credentials and configuration are yours to mount at run time.
# ---------------------------------------------------------------------------
RUN npm install -g \
        @anthropic-ai/claude-code \
        @anthropic-ai/sandbox-runtime \
        @openai/codex \
    && npm cache clean --force

# triton is deliberately absent: the PyTorch base image pins its own build,
# and asking pip for it invites a resolver fight with torch.
#
# --break-system-packages is a no-op on the conda-based PyTorch images but is
# required if BASE_IMAGE is swapped for a system-Python one (PEP 668).
RUN pip install --no-cache-dir \
        nvtx \
        pyarrow \
        pytest \
        ruff \
 || pip install --no-cache-dir --break-system-packages \
        nvtx \
        pyarrow \
        pytest \
        ruff
