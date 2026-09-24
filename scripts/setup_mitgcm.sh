#!/usr/bin/env bash
# MITgcm checkpoint69k + gfortran + paquets Python (VM ou nuage Claude Code).
set -eu
ROOT="${MITGCM_ROOT:-$HOME/MITgcm}"
if ! command -v gfortran >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then apt-get update -qq && apt-get install -y -qq gfortran make
  elif command -v sudo >/dev/null 2>&1; then sudo apt-get update -qq && sudo apt-get install -y -qq gfortran make
  else echo "Installer gfortran : apt-get install gfortran" >&2; exit 1; fi
fi
[ -d "$ROOT/model" ] || git clone -q --depth 1 --branch checkpoint69k https://github.com/MITgcm/MITgcm.git "$ROOT"
python3 -c "import numpy, scipy, matplotlib" 2>/dev/null || python3 -m pip install -q numpy scipy matplotlib \
  || python3 -m pip install -q --break-system-packages numpy scipy matplotlib
echo "OK : gfortran $(gfortran -dumpversion), MITgcm dans $ROOT"
