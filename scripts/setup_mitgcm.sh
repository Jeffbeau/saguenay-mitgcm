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
# Paquets Python : jamais de pip dans le Python systeme (Ubuntu 24.04 le refuse, et un NumPy 2
# installe a cote de rasterio d'apt le casse). Dans un venv actif : pip ; sinon : apt.
if ! python3 -c "import numpy, scipy, matplotlib" 2>/dev/null; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then python3 -m pip install -q numpy scipy matplotlib
  elif [ "$(id -u)" = 0 ]; then apt-get install -y -qq python3-numpy python3-scipy python3-matplotlib
  else echo "Activer un venv (source ~/venv/bin/activate) ou : sudo apt install python3-numpy python3-scipy python3-matplotlib" >&2; exit 1; fi
fi
echo "OK : gfortran $(gfortran -dumpversion), MITgcm dans $ROOT"
