#!/bin/bash
# Run on the GB200 node to launch the setup container via SLURM/Pyxis.
# /lustre is mounted inside the container at /lustre.
# All build artifacts are saved to ${REPO_ROOT}/deps/.
#
# Usage:
#   bash run_gb200_setup.sh
#   JOBID=<your_job_id> bash run_gb200_setup.sh   # override job id

set -eo pipefail

REPO_ROOT="/lustre/fsw/coreai_devtech_all/jiashuy/GR-mon5-11"
DEPS_DIR="${REPO_ROOT}/deps"
JOBID="${JOBID:-1736558}"

mkdir -p "${DEPS_DIR}"

echo "=== Launching GB200 setup container ==="
echo "Repo root : ${REPO_ROOT}"
echo "Deps dir  : ${DEPS_DIR}"
echo "Job ID    : ${JOBID}"

export REPO_ROOT DEPS_DIR

srun \
  --overlap \
  --jobid="${JOBID}" \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --container-mounts=/lustre:/lustre \
  --container-name="recsys-docker" \
  --container-image="nvcr.io/nvidia/pytorch:26.02-py3" \
  bash "${REPO_ROOT}/docker/install_gb200.sh" 2>&1 | tee "${REPO_ROOT}/setup_gb200.log"

echo "=== Setup complete. Artifacts saved to: ${DEPS_DIR} ==="
