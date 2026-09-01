#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFECT_DIR="$REPO_ROOT"
PREFECT_GCP_DIR="$REPO_ROOT/src/integrations/prefect-gcp"
PUBLISH_URL="https://us-central1-python.pkg.dev/cds-artifacts/public-python/"

echo "==> Ensuring Artifact Registry keyring auth is available"
uv tool install --quiet --with keyrings.google-artifactregistry-auth keyring

export UV_KEYRING_PROVIDER=subprocess
unset UV_PUBLISH_TOKEN

echo "==> Building prefect"
rm -rf "$PREFECT_DIR/dist"
uv build --project "$PREFECT_DIR" --out-dir "$PREFECT_DIR/dist"

echo "==> Building prefect-gcp"
rm -rf "$PREFECT_GCP_DIR/dist"
uv build --project "$PREFECT_GCP_DIR" --out-dir "$PREFECT_GCP_DIR/dist"

echo "==> Publishing prefect to $PUBLISH_URL"
uv publish --publish-url "$PUBLISH_URL" --username oauth2accesstoken "$PREFECT_DIR"/dist/*

echo "==> Publishing prefect-gcp to $PUBLISH_URL"
uv publish --publish-url "$PUBLISH_URL" --username oauth2accesstoken "$PREFECT_GCP_DIR"/dist/*

echo "==> Done"
