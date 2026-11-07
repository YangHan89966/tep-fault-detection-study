#!/usr/bin/env bash
# Use bash explicitly; no dependency on archive execute bits.
set -u
kit="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || exit 1
action=run
case "${1:-}" in
  --setup-runtime) action=setup; shift ;;
  --probe-runtime) action=probe; shift ;;
esac
bin_venv="$kit/.f9-venv-wsl/bin/python"
candidates=()
[ -z "${F9_PYTHON:-}" ] || candidates+=("$F9_PYTHON")
candidates+=("$bin_venv")
[ -z "${VIRTUAL_ENV:-}" ] || candidates+=("$VIRTUAL_ENV/bin/python")
candidates+=(python3.12 python3 python)
selected=""; base=""
for pybin in "${candidates[@]}"; do
  command -v -- "$pybin" >/dev/null 2>&1 || continue
  "$pybin" -c 'import sys; sys.exit(not(sys.implementation.name=="cpython" and sys.version_info[:2]==(3,12)))' >/dev/null 2>&1 || continue
  [ -n "$base" ] || base="$pybin"
  if "$pybin" -c 'import numpy as n,sys;sys.exit(n.__version__!="2.3.5")' >/dev/null 2>&1; then selected="$pybin"; break; fi
done
if [ -n "$selected" ]; then
  if [ "$action" = run ]; then exec "$selected" "$kit/runtime/launch.py" "$@"; fi
  [ "$action" = probe ] || echo "Pinned runtime already available: $selected"
  exit 0
fi
[ "$action" != probe ] || exit 12
if [ "$action" != setup ]; then
  echo "Needs CPython 3.12 and NumPy 2.3.5. With Python 3.12 installed, use: bash RUN_WSL.sh --setup-runtime" >&2
  exit 12
fi
if [ -z "$base" ]; then echo "No existing CPython 3.12. Please install CPython 3.12 with venv support, or set F9_PYTHON to its path." >&2; exit 12; fi
# Never overwrite another project or modify the selected system interpreter.
envdir="$kit/.f9-venv-wsl"
if [ ! -f "$envdir/pyvenv.cfg" ]; then
  if [ -e "$envdir" ]; then echo "Cannot create venv: directory already exists without pyvenv.cfg: $envdir" >&2; exit 13; fi
  "$base" -m venv --copies "$envdir" || exit 13
fi
"$bin_venv" -c 'import sys;sys.exit(sys.prefix==sys.base_prefix or sys.version_info[:2]!=(3,12))' || { echo 'Invalid private Python 3.12 venv.' >&2; exit 13; }
"$bin_venv" -m pip --require-virtualenv --disable-pip-version-check install --only-binary=:all: 'numpy==2.3.5' || exit 13
exec "$bin_venv" -c 'import numpy as n;assert n.__version__=="2.3.5";print("Local runtime ready. Start RUN_FAULT9.cmd or bash RUN_WSL.sh.")'
