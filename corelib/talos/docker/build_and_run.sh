#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the Talos development image and open a shell in it.
#
#   ./build_and_run.sh                       build, then run an interactive shell
#   ./build_and_run.sh --skip-build          reuse the existing image
#   ./build_and_run.sh --tag myimage         image name (default: talos)
#   ./build_and_run.sh --daemon              run detached instead of interactive
#   ./build_and_run.sh --cmd "pytest -q"     run one command instead of a shell
#   ./build_and_run.sh --dockerfile f.Docker use a different Dockerfile
#
# The repo is mounted at /workspace. Agent credentials are not handled here:
# mount whatever the agent expects yourself, e.g.
#   EXTRA_ARGS="-v $HOME/.claude:/root/.claude -v $HOME/.claude.json:/root/.claude.json"
#
# Pick GPUs with NV_GPU (default: all):
#   NV_GPU=0,1 ./build_and_run.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

DOCKERFILE="pytorch.Dockerfile"
IMAGE_NAME="talos_example"
CONTAINER_NAME=""
RUN_TYPE="-it"
DOCKER_CMD="bash"
SKIP_BUILD=false
WORKDIR="/workspace"

usage() {
    sed -n '/^# Build the Talos/,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
    --dockerfile)     DOCKERFILE="$2"; shift 2 ;;
    --tag)            IMAGE_NAME="$2"; shift 2 ;;
    --container-name) CONTAINER_NAME="$2"; shift 2 ;;
    --cmd)            DOCKER_CMD="$2"; shift 2 ;;
    --daemon)         RUN_TYPE="-d"; DOCKER_CMD="sleep infinity"; shift ;;
    --skip-build)     SKIP_BUILD=true; shift ;;
    -h|--help)        usage ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

: "${CONTAINER_NAME:=${IMAGE_NAME}}"

cd "${SCRIPT_DIR}"

if [ "${SKIP_BUILD}" = false ]; then
    echo ">> building ${IMAGE_NAME}:latest from ${DOCKERFILE}"
    docker build -f "${DOCKERFILE}" --tag "${IMAGE_NAME}:latest" "${SCRIPT_DIR}"
fi

# Array, not a string: an unquoted string would word-split the quotes into
# docker's argv.
GPU_ARGS=(--gpus all)
if [ -n "${NV_GPU:-}" ]; then
    GPU_ARGS=(--gpus "device=${NV_GPU}")
fi

# Split EXTRA_ARGS on whitespace so callers can pass their own mounts/env.
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

echo ">> running ${CONTAINER_NAME}  (${PROJECT_ROOT} -> ${WORKDIR})"
exec docker run \
    "${RUN_TYPE}" \
    --rm \
    "${GPU_ARGS[@]}" \
    --name "${CONTAINER_NAME}" \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --workdir "${WORKDIR}" \
    -v "${PROJECT_ROOT}:${WORKDIR}" \
    "${EXTRA[@]}" \
    "${IMAGE_NAME}:latest" \
    bash -c "${DOCKER_CMD}"
