#!/usr/bin/env bash
# Fetch the CUTLASS headers PyTorch v2.6.0 pins (third_party/cutlass @ bbe579a9) into third_party/cutlass.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -d third_party/cutlass/include/cute ] && exit 0
mkdir -p third_party && git init -q third_party/cutlass && cd third_party/cutlass
git remote add origin https://github.com/NVIDIA/cutlass.git
git fetch -q --depth 1 --filter=blob:none origin bbe579a9e3beb6ea6626d9227ec32d0dae119a49
git sparse-checkout set include && git checkout -q FETCH_HEAD
