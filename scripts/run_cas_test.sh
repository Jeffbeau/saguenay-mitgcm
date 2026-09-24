#!/usr/bin/env bash
# Cas test synthetique de bout en bout (serie, 1 coeur, ~10 min) : generation, compilation, run, diagnostics.
set -eu
ROOT="${MITGCM_ROOT:-$HOME/MITgcm}"
cd "$(dirname "$0")/../cas_test"
python3 gen_testcase.py --mitgcm "$ROOT"
mkdir -p build run
if [ ! -x build/mitgcmuv ]; then
  (cd build && "$ROOT/tools/genmake2" -rootdir="$ROOT" -mods=../code > genmake.log 2>&1 \
     && make depend > depend.log 2>&1 && make -j"$(nproc)" > make.log 2>&1)
fi
cd run && rm -f ./*.data ./*.meta output.txt && ln -sf ../input/* . && ln -sf ../build/mitgcmuv .
./mitgcmuv > output.txt 2>&1
grep -q "Execution ended Normally" output.txt || { echo "Echec du run : voir cas_test/run/output.txt" >&2; exit 1; }
python3 ../../sagdiag/sagdiag_run.py . --config ../../sagdiag/coupes_test.json --min-diam 3
echo "Comparer cas_test/run/diag/resume.txt a resultats_cas_test/resume.txt"
