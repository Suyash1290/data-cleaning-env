#!/usr/bin/env bash

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper functions
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
hint() { echo -e "${YELLOW}[HINT]${NC} $1"; }
stop_at() { echo -e "\n${RED}${BOLD}Validation stopped at $1 due to errors.${NC}"; exit 1; }

# Timeout helper for commands that might hang
run_with_timeout() {
  local timeout=$1
  shift
  # Use perl or python for cross-platform timeout if 'timeout' cmd missing
  if command -v timeout &>/dev/null; then
    timeout "$timeout" "$@"
  else
    # Mac fallback using perl
    perl -e 'eval { local $SIG{ALRM} = sub { die "alarm\n" }; alarm shift; system(@ARGV); alarm 0 }; if ($@) { exit 124 }' "$timeout" "$@"
  fi
}

DOCKER_BUILD_TIMEOUT=180
REPO_DIR="."

printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Pre-Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n\n"

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All checks passed!${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"
exit 0
