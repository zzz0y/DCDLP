#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${DCDLP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${DCDLP_IMAGE:-dcdlp:selective-2.3.1-cu121}"
ARCHIVE="${DCDLP_IMAGE_ARCHIVE:-$ROOT/offline/dcdlp-selective-image.tar}"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Image archive not found: $ARCHIVE" >&2
  exit 1
fi
docker load --input "$ARCHIVE"
docker image inspect "$IMAGE" --format 'Loaded {{.RepoTags}} ({{.Architecture}}/{{.Os}})'
