#!/usr/bin/env bash
# Run the AutoTTS baseline discovery (no profile preferences).
# Usage: BETA=0.5 bash scripts/run_autotts_baseline.sh
#        BETA=0.5 CONFIG=config/experiment_aime.json bash scripts/run_autotts_baseline.sh
set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BETA="${BETA:?set BETA (e.g. 0.5 or 1.0)}"
CONFIG="${CONFIG:-config/experiment_aime.json}"
RUN_ID="${RUN_ID:-autotts_beta${BETA//./}}"

# --- Token loading ---
TOKEN_CACHE="$REPO_ROOT/.anthropic_env"
if [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ] && [ -r "$TOKEN_CACHE" ]; then
  echo "loading token from environment file: $TOKEN_CACHE"
  # shellcheck disable=SC1090
  . "$TOKEN_CACHE"
fi
if [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
  echo "FATAL: no ANTHROPIC_AUTH_TOKEN available"
  exit 1
fi

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.anthropic.com}"
export ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_CUSTOM_HEADERS="${ANTHROPIC_CUSTOM_HEADERS:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-any}"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="1"
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK="1"
export CLAUDE_CODE_MAX_RETRIES="0"
export MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-4000}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-32000}"
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3900000}"
export CLAUDE_STREAM_IDLE_TIMEOUT_MS="${CLAUDE_STREAM_IDLE_TIMEOUT_MS:-3900000}"

echo "run_id=$RUN_ID"
echo "beta=$BETA"
echo "config=$CONFIG"
echo "endpoint=$ANTHROPIC_BASE_URL"

exec "$REPO_ROOT/.venv/bin/python" scripts/run_discovery.py \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --round-limit 5 \
    --seed-count 1000 \
    --betas "$BETA" \
    --evaluation-workers 60 \
    --prompt discovery/prompts/discovery_autotts.md
