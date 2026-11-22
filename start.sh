#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export BUILDX_BUILDER="${BUILDX_BUILDER:-default}"
docker compose up --build --detach --wait
port=${VITRAIL_PORT:-19961}
printf 'Vitrail Bench at http://127.0.0.1:%s\n' "$port"
