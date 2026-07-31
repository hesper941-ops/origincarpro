#!/usr/bin/env bash
set -euo pipefail

DESKTOP_USER="${PERSON_BOARD_DESKTOP_USER:-sunrise}"

if [[ -f /opt/tros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/tros/humble/setup.bash
else
  echo "ERROR: /opt/tros/humble/setup.bash not found" >&2
  exit 1
fi
if [[ -f /root/intelligent_car_ws/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /root/intelligent_car_ws/install/setup.bash
else
  echo "ERROR: workspace install/setup.bash not found; build the package first" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  DISPLAY="$(who | awk -v user="$DESKTOP_USER" '
    $1 == user && match($0, /\(:[0-9]+(\.[0-9]+)?\)/) {
      value = substr($0, RSTART + 1, RLENGTH - 2); print value; exit
    }')"
fi
DISPLAY="${DISPLAY:-:0}"
export DISPLAY

desktop_uid="$(id -u "$DESKTOP_USER" 2>/dev/null || true)"
if [[ -z "$desktop_uid" ]]; then
  echo "ERROR: desktop user '$DESKTOP_USER' does not exist" >&2
  exit 1
fi

declare -a authority_candidates=()
if [[ -n "${XAUTHORITY:-}" ]]; then
  authority_candidates+=("$XAUTHORITY")
fi
authority_candidates+=(
  "/home/$DESKTOP_USER/.Xauthority"
  "/run/user/$desktop_uid/gdm/Xauthority"
)

for proc_env in /proc/[0-9]*/environ; do
  [[ -r "$proc_env" ]] || continue
  proc_uid="$(stat -c %u "${proc_env%/environ}" 2>/dev/null || true)"
  [[ "$proc_uid" == "$desktop_uid" ]] || continue
  discovered="$(tr '\0' '\n' < "$proc_env" 2>/dev/null | sed -n 's/^XAUTHORITY=//p' | head -n 1)"
  [[ -n "$discovered" ]] && authority_candidates+=("$discovered")
done

resolved_authority=""
for candidate in "${authority_candidates[@]}"; do
  if [[ -f "$candidate" && -r "$candidate" ]]; then
    resolved_authority="$candidate"
    break
  fi
done
if [[ -z "$resolved_authority" ]]; then
  echo "ERROR: no readable XAUTHORITY found for $DESKTOP_USER" >&2
  echo "Checked: ${authority_candidates[*]}" >&2
  echo "DISPLAY=$DISPLAY" >&2
  exit 1
fi
export XAUTHORITY="$resolved_authority"

echo "desktop_user=$DESKTOP_USER"
echo "DISPLAY=$DISPLAY"
echo "XAUTHORITY=$XAUTHORITY"

if command -v xdpyinfo >/dev/null 2>&1; then
  if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "ERROR: cannot access DISPLAY=$DISPLAY with XAUTHORITY=$XAUTHORITY" >&2
    exit 1
  fi
fi

exec ros2 launch person_board_detection person_board_display.launch.py
