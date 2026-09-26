#!/usr/bin/env bash
# Imbrication du cas test : enfant A (hydrostatique, z*) et/ou B (non hydrostatique, SL lineaire)
# forces hors ligne par le parent (cas_test/run, produit par run_cas_test.sh).
# Usage : bash scripts/run_imbrication.sh [A] [B]     (defaut : A B, en parallele, 1 coeur chacun)
set -eu
ROOT="${MITGCM_ROOT:-$HOME/MITgcm}"
cd "$(dirname "$0")/.."
REPO=$PWD
[ -f cas_test/run/state3D.0000000000.001.001.meta ] || [ -f cas_test/run/state3D.0000000000.meta ] \
  || { echo "Parent absent : lancer d'abord scripts/run_cas_test.sh" >&2; exit 1; }
CFGS="${*:-A B}"

run_enfant() {
  c=$1; d=cas_test/enfant_$c; opt=""; [ "$c" = B ] && opt="--nh"
  python3 cas_test/gen_enfant.py $opt --mitgcm "$ROOT" > /dev/null
  python3 imbrication/extraire_obcs.py cas_test/run "$d" > "$d/extraction.log"
  if [ ! -x "$d/build/mitgcmuv" ] || [ -n "$(find "$d/code" -newer "$d/build/mitgcmuv")" ]; then
    rm -rf "$d/build"; mkdir -p "$d/build"
    (cd "$d/build" && "$ROOT/tools/genmake2" -rootdir="$ROOT" -mods=../code > genmake.log 2>&1 \
       && make depend > depend.log 2>&1 && make -j2 > make.log 2>&1)
  fi
  mkdir -p "$d/run"
  (cd "$d/run" && rm -rf ./*.data ./*.meta output.txt diag && ln -sf ../input/* . && ln -sf ../build/mitgcmuv . \
     && ./mitgcmuv > output.txt 2>&1)
  grep -q "Execution ended Normally" "$d/run/output.txt" || { echo "Echec du run $c : $d/run/output.txt" >&2; return 1; }
  python3 imbrication/comparer.py cas_test/run "$d" --config sagdiag/coupes_test.json > "$d/comparaison.log"
  echo "== Enfant $c"; cat "$d/extraction.log"; echo; cat "$d/run/diag/imbrication.txt"
}

pids=""
for c in $CFGS; do run_enfant "$c" > "/tmp/imbrication_$c.log" 2>&1 & pids="$pids $!"; done
st=0; for p in $pids; do wait "$p" || st=1; done
for c in $CFGS; do cat "/tmp/imbrication_$c.log"; echo; done
exit $st
