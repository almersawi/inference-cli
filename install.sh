#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${HOME}/.local/bin"
LINK_PATH="${TARGET_DIR}/inference"

mkdir -p "${TARGET_DIR}"
ln -sf "${SCRIPT_DIR}/inference.py" "${LINK_PATH}"

echo "Installed: ${LINK_PATH} -> ${SCRIPT_DIR}/inference.py"

case ":${PATH}:" in
    *":${TARGET_DIR}:"*) echo "PATH already includes ${TARGET_DIR}." ;;
    *) echo "WARNING: ${TARGET_DIR} is NOT on PATH. Add this to your ~/.zshrc:"
       echo "    export PATH=\"${TARGET_DIR}:\$PATH\"" ;;
esac
