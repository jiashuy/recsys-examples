#!/bin/bash
# Run INSIDE the container by run_gb200_setup.sh.
# Installs all GB200 dependencies into ${DEPS_DIR}.
# Idempotent: skips steps whose output directories already exist.

set -e

REPO_ROOT="${REPO_ROOT:-/lustre/fsw/coreai_devtech_all/jiashuy/GR-mon5-11}"
DEPS_DIR="${DEPS_DIR:-${REPO_ROOT}/deps}"
INSTALL_PREFIX="${DEPS_DIR}/python_pkgs"
SRC_DIR="${DEPS_DIR}/src"

mkdir -p "${INSTALL_PREFIX}" "${SRC_DIR}"

# All pip installs go to INSTALL_PREFIX so they persist on the host
PIP_INSTALL="pip install --prefix=${INSTALL_PREFIX} --no-cache-dir"
export PYTHONPATH="${INSTALL_PREFIX}/lib/$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:${PYTHONPATH}"

echo ">>> [1/8] libnvidia-ml / CCCL symlinks"
rm -rf /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1
ln -sf /usr/local/cuda-13/targets/sbsa-linux/lib/stubs/libnvidia-ml.so \
       /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1
ln -sf /usr/local/cuda-13/targets/sbsa-linux/include/cccl/cuda \
       /usr/local/cuda/include/cuda
apt update -y --fix-missing && apt install -y gdb && apt clean && rm -rf /var/lib/apt/lists/*

echo ">>> [2/8] Megatron-LM + pip deps"
[ -d "${SRC_DIR}/megatron-lm" ] || \
  git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git "${SRC_DIR}/megatron-lm"
pip install --no-deps -e "${SRC_DIR}/megatron-lm"
${PIP_INSTALL} torchx gin-config torchmetrics==1.0.3 typing-extensions iopath pyvers \
  cloudpickle triton==3.6.0 nvidia-cutlass-dsl==4.3.0 pre-commit

echo ">>> [3/8] FBGEMM (sm_100 only)"
pip install --no-cache-dir setuptools-git-versioning scikit-build
[ -d "${SRC_DIR}/fbgemm" ] || \
  git clone --recursive -b v1.5.0 https://github.com/pytorch/FBGEMM.git "${SRC_DIR}/fbgemm"
cd "${SRC_DIR}/fbgemm/fbgemm_gpu"
python setup.py install --prefix="${INSTALL_PREFIX}" \
  --build-target=default --build-variant=cuda \
  -DTORCH_CUDA_ARCH_LIST="10.0"

echo ">>> [4/8] TorchRec"
${PIP_INSTALL} tensordict orjson
[ -d "${SRC_DIR}/torchrec" ] || \
  git clone --recursive -b release/V1.5.0 https://github.com/pytorch/torchrec.git "${SRC_DIR}/torchrec"
cd "${SRC_DIR}/torchrec"
pip install --no-deps --prefix="${INSTALL_PREFIX}" .

echo ">>> [5/8] flash-attention (arbitrary_mask)"
[ -d "${SRC_DIR}/flash-attention" ] || \
  git clone -b arbitrary_mask https://github.com/jiayus-nvidia/flash-attention.git "${SRC_DIR}/flash-attention"
cd "${SRC_DIR}/flash-attention"
# Patch get_package_version to return a fixed version string,
# bypassing NGC torch version parsing which fails on non-standard version strings
# like "2.11.0a0+eb65b36914.nv26.02"
python3 - << 'PYEOF'
import re, pathlib
f = pathlib.Path("setup.py")
c = f.read_text()
patched = re.sub(
    r"def get_package_version\(\):.*?(?=\ndef |\nclass |\Z)",
    'def get_package_version():\n    return "2.7.4+local"\n\n',
    c, flags=re.DOTALL
)
if patched != c:
    f.write_text(patched)
    print("Patched get_package_version for NGC torch compatibility")
else:
    print("WARNING: get_package_version pattern not found, proceeding without patch")
PYEOF
pip install --no-deps --no-build-isolation -e .

echo ">>> [6/8] fbgemm_gpu_hstu (Blackwell only)"
FBGEMM_HSTU_DIR="${SRC_DIR}/fbgemm_hstu"
[ -d "${FBGEMM_HSTU_DIR}" ] || cp -r "${REPO_ROOT}/third_party/FBGEMM" "${FBGEMM_HSTU_DIR}"
cd "${FBGEMM_HSTU_DIR}/fbgemm_gpu/experimental/hstu"
HSTU_DISABLE_86OR89=TRUE \
HSTU_DISABLE_ARBITRARY=TRUE \
HSTU_DISABLE_LOCAL=TRUE \
HSTU_DISABLE_RAB=TRUE \
HSTU_DISABLE_DRAB=TRUE \
HSTU_DISABLE_FP16=TRUE \
HSTU_ARCH_LIST="10.0" \
pip install --no-build-isolation --prefix="${INSTALL_PREFIX}" .

echo ">>> [7/8] nvcomp (aarch64)"
if [ ! -d "${DEPS_DIR}/nvcomp" ]; then
  cd "${DEPS_DIR}"
  rm -f nvcomp-linux-aarch64-5.1.0.21_cuda12-archive.tar.xz
  wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-aarch64/nvcomp-linux-aarch64-5.1.0.21_cuda12-archive.tar.xz
  tar -xf nvcomp-linux-aarch64-5.1.0.21_cuda12-archive.tar.xz
  mv nvcomp-linux-aarch64-5.1.0.21_cuda12-archive nvcomp
  rm nvcomp-linux-aarch64-5.1.0.21_cuda12-archive.tar.xz
else
  echo "    nvcomp already exists, skipping download."
fi

echo ">>> [8/8] dynamicemb + commons"
cd "${REPO_ROOT}/corelib/dynamicemb"
python setup.py install --prefix="${INSTALL_PREFIX}"

cd "${REPO_ROOT}/examples/commons"
TORCH_CUDA_ARCH_LIST="10.0" python3 setup.py install --prefix="${INSTALL_PREFIX}"

echo ">>> All dependencies installed to: ${DEPS_DIR}"
PY_VER=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
echo ">>> To use, set: export PYTHONPATH=${INSTALL_PREFIX}/lib/${PY_VER}/site-packages:\$PYTHONPATH"
