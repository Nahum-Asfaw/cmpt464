#!/usr/bin/env bash
# Create the assignment's Python environment and download the meshes.
# Requires Linux x86_64, Python 3.12, and an NVIDIA driver supporting CUDA 13.0.
#
# Install:   bash ./install_env.sh
# Activate:  source .venv/bin/activate
# Remove:    bash ./install_env.sh --remove
# --remove deletes only .venv; downloaded meshes stay in data/.
#
# Off campus, the package server may require the SFU VPN.

set -e

# Keep .venv and data beside this script, regardless of the calling directory.
cd -- "$(dirname -- "$0")"

case "$*" in
"") ;;
--remove)
    rm -rf -- .venv
    echo "Removed the repository's .venv directory."
    exit 0
    ;;
*)
    echo "Usage: bash ./install_env.sh [--remove]" >&2
    exit 1
    ;;
esac

python3.12 -m venv .venv

# Kaolin's imports require torchvision.
# Use the venv's Python so packages are installed into this repository's environment.
.venv/bin/python -m pip install \
    --index-url http://cs-gruvi-84.cmpt.sfu.ca:8001/simple/ \
    --trusted-host cs-gruvi-84.cmpt.sfu.ca \
    --no-cache-dir \
    torch==2.9.0+cu130 \
    torchvision==0.24.0+cu130 \
    kaolin==0.18.0

# Download the training and evaluation meshes into data/train/ and data/eval/.
mkdir -p data
.venv/bin/python - <<'PYTHON'
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

archive = Path("data/assignment2_data.zip")
urlretrieve("ftp://cs-gruvi-84.cmpt.sfu.ca:2122/assignment2_data.zip", archive)
with ZipFile(archive) as meshes:
    meshes.extractall("data")
archive.unlink()
PYTHON

echo "Environment ready. Activate it with: source .venv/bin/activate"
