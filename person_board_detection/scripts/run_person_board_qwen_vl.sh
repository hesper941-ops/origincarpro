#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${PERSON_BOARD_WORKSPACE:-/root/intelligent_car_ws}"
ENV_FILE="${PERSON_BOARD_LLM_ENV_FILE:-/root/.config/person_board_llm/env}"

source_safely() {
  set +u
  source "$1"
  set -u
}

source_safely /opt/tros/humble/setup.bash
source_safely "$WORKSPACE/install/setup.bash"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: private environment file missing: $ENV_FILE" >&2
  exit 2
fi
source_safely "$ENV_FILE"
: "${DASHSCOPE_API_KEY:?DASHSCOPE_API_KEY is not configured}"
: "${PERSON_BOARD_LLM_BASE_URL:?PERSON_BOARD_LLM_BASE_URL is not configured}"
: "${PERSON_BOARD_LLM_MODEL:?PERSON_BOARD_LLM_MODEL is not configured}"

HOST="$(python3 -c 'import os,urllib.parse; print(urllib.parse.urlparse(os.environ["PERSON_BOARD_LLM_BASE_URL"]).netloc)')"
echo "API key configured=true"
echo "Base URL host=$HOST"
echo "model=$PERSON_BOARD_LLM_MODEL"
exec ros2 launch person_board_detection person_board_qwen_vl.launch.py "$@"
