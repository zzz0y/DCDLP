#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${DCDLP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARCHIVE_GZ="$ROOT/offline/dcdlp-image.tar.gz"
ARCHIVE_TAR="$ROOT/offline/dcdlp-image.tar"

if [[ -f "$ARCHIVE_GZ" ]]; then
  gzip -dc "$ARCHIVE_GZ" | docker load
elif [[ -f "$ARCHIVE_TAR" ]]; then
  docker load --input "$ARCHIVE_TAR"
else
  echo "Image archive not found under $ROOT/offline" >&2
  exit 1
fi

docker image inspect dcdlp:2.3.1-cu121 --format 'Loaded {{.RepoTags}} ({{.Architecture}}/{{.Os}})'
