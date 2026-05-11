#!/bin/bash
set -e
trap 'echo "Build interrupted."; docker buildx stop; exit 1' INT TERM

# Step 1: build image and load into local Docker image store
docker buildx build \
  --platform linux/arm64 \
  --progress=plain \
  -f docker/Dockerfile.gb200 \
  -t recsys-examples:gb200 \
  --load \
  . 2>&1 | tee ./build_gb200.log

# Step 2: export image to file
echo "Saving image to gb200.tar.gz ..."
docker save recsys-examples:gb200 | gzip > ./gb200.tar.gz
echo "Done: gb200.tar.gz"
