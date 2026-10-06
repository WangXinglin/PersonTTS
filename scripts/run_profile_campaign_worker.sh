#!/usr/bin/env bash
# Canonical AIME profile-campaign worker launcher.
# Keep experiment semantics in the campaign manifest; this script only supplies
# the verified runtime environment and resumable worker flags.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CAMPAIGN_ID="${CAMPAIGN_ID:-aime_profile_campaign}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$REPO_ROOT/artifacts/profile_discovery/$CAMPAIGN_ID}"
WORKER_ID="${WORKER_ID:-${TRIAL_ID:-$(hostname).$$}}"
LOG_PATH="${LOG_PATH:-/tmp/${CAMPAIGN_ID}_${WORKER_ID}.log}"

# --- Token loading ---
# Load credentials from an optional local environment file or from the
# environment. A token file may use either the OAuth-style
# ANTHROPIC_AUTH_TOKEN or a plain ANTHROPIC_API_KEY with its own endpoint.
TOKEN_CACHE="${TOKEN_CACHE:-$REPO_ROOT/.anthropic_env}"
if [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -r "$TOKEN_CACHE" ]; then
  echo "loading token from environment file: $TOKEN_CACHE"
  # shellcheck disable=SC1090
  . "$TOKEN_CACHE"
fi
if [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "FATAL: no ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY available"
  echo "  Set a token in the environment or populate $TOKEN_CACHE"
  exit 1
fi

# --- Upstream endpoint ---
# Set ANTHROPIC_BASE_URL to point at a gateway; it defaults to the public API.
# An API-key token file may set its own ANTHROPIC_BASE_URL, which is not
# overridden here.
if [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
  export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.anthropic.com}"
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-any}"
else
  export ANTHROPIC_BASE_URL
  export ANTHROPIC_API_KEY
fi
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-}"
export ANTHROPIC_CUSTOM_HEADERS="${ANTHROPIC_CUSTOM_HEADERS:-}"

export CAMPAIGN_ID CAMPAIGN_ROOT WORKER_ID LOG_PATH
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="1"
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK="1"
export CLAUDE_CODE_MAX_RETRIES="0"
export MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-4000}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-32000}"
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3900000}"
export CLAUDE_STREAM_IDLE_TIMEOUT_MS="${CLAUDE_STREAM_IDLE_TIMEOUT_MS:-3900000}"

echo "campaign_id=$CAMPAIGN_ID"
echo "campaign_root=$CAMPAIGN_ROOT"
echo "worker_id=$WORKER_ID"
echo "model=claude-sonnet-4-6"
echo "max_thinking_tokens=$MAX_THINKING_TOKENS"
echo "max_output_tokens=$CLAUDE_CODE_MAX_OUTPUT_TOKENS"
echo "api_timeout_ms=$API_TIMEOUT_MS"
echo "stream_idle_timeout_ms=$CLAUDE_STREAM_IDLE_TIMEOUT_MS"
echo "agent_retries=$CLAUDE_CODE_MAX_RETRIES"
echo "endpoint=$ANTHROPIC_BASE_URL"

RC=0
if [ ! -f "$CAMPAIGN_ROOT/campaign.json" ]; then
  echo "campaign not initialized: $CAMPAIGN_ROOT" >"$LOG_PATH"
  RC=2
else
  "$REPO_ROOT/.venv/bin/python" -m py_compile \
    scripts/profile_campaign.py \
    discovery/profile_campaign.py \
    discovery/claude_proposer.py \
    evaluation/cache_inventory.py \
    evaluation/runner.py
  PREFLIGHT_RC=$?
  if [ "$PREFLIGHT_RC" -ne 0 ]; then
    echo "worker preflight failed: $PREFLIGHT_RC" >"$LOG_PATH"
    RC="$PREFLIGHT_RC"
  else
    WORKER_ARGS=(
      worker
      --campaign-root "$CAMPAIGN_ROOT"
      --worker-id "$WORKER_ID"
      --idle-exit-seconds 120
      --heartbeat-interval-seconds 30
      --lease-timeout-seconds 300
      --max-bucket-probes 32
      --max-task-probes 16
      --reap-bucket-probes 4
      --reap-task-probes 8
      --retry-backoff-base-seconds 60
      --retry-backoff-max-seconds 900
    )
    if [ "${WORKER_ONCE:-0}" = "1" ]; then
      WORKER_ARGS+=(--once)
    fi
    "$REPO_ROOT/.venv/bin/python" scripts/profile_campaign.py "${WORKER_ARGS[@]}" >"$LOG_PATH" 2>&1
    RC=$?
  fi
fi

echo "worker_exit_code=$RC"
echo "worker_log=$LOG_PATH"
tail -n 200 "$LOG_PATH" 2>/dev/null || true
exit "$RC"
