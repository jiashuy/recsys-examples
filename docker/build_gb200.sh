#!/bin/bash
set -e
trap 'echo "Build interrupted."; docker buildx stop; exit 1' INT TERM

REGISTRY="gitlab-master.nvidia.com:5005/devtech-compute/distributed-recommender:arm_gb200"

docker buildx build \
  --platform linux/arm64 \
  --progress=plain \
  -f docker/Dockerfile.gb200 \
  -t "${REGISTRY}" \
  --push \
  . 2>&1 | tee ./build_gb200.log

echo "Done: ${REGISTRY}"
