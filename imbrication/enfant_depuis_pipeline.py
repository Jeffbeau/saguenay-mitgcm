#!/usr/bin/env python3
"""
enfant.json pour le vrai enfant (25 m) a partir des sorties du pipeline grille (s02/s03).

Le repere du parent MITgcm commence a sa face sud-ouest (usingCartesianGrid, XG = 0) :
x0, y0 de l'enfant = faces de l'enfant - faces du parent (UTM 19N).
Les OB sont les bords mouilles de l'enfant ; les coins vont aux frontieres E/O.

Usage (sur la VM, apres run_all.py) :
  python3 imbrication/enfant_depuis_pipeline.py pipeline_grille/output ~/runs/enfantA \
      --t0 89280 --cycles 2 [--nh] [--periode 930]
puis
  python3 imbrication/extraire_obcs.py ~/runs/parent ~/runs/enfantA
"""
import argparse, json, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("sortie", help="dossier de sortie du pipeline (contient parent/ et child/)")
ap.add_argument("enfant", help="dossier de l'enfant (input/ y est cree)")
ap.add_argument("--t0", type=float, required=True, help="depart dans le temps du parent (s), multiple de 44640")
ap.add_argument("--cycles", type=float, default=2.0)
ap.add_argument("--periode", type=float, default=930.0, help="espacement des enregistrements OBCS (s)")
ap.add_argument("--nh", action="store_true", help="config B : non hydrostatique, surface libre lineaire")
ap.add_argument("--hfacmin", type=float, default=0.2)
ap.add_argument("--hfacmindr", type=float, default=0.5)
ap.add_argument("--eponge", type=int, default=40, help="epaisseur de l'eponge (cellules de l'enfant)")
a = ap.parse_args()

T_M2 = 44640.0
assert abs(a.t0 / T_M2 - round(a.t0 / T_M2)) < 1e-9, "t0 doit etre un multiple de 44640 s"
gp = np.load(os.path.join(a.sortie, "parent", "grid.npz"))
gc = np.load(os.path.join(a.sortie, "child", "grid.npz"))
nx, ny = gc["xc"].size, gc["yc"].size
dx = float(gc["xg"][1] - gc["xg"][0])
wet = gc["wet"].astype(bool)
ob = {"W": np.nonzero(wet[:, 0])[0], "E": np.nonzero(wet[:, -1])[0],
      "S": np.nonzero(wet[0, :])[0], "N": np.nonzero(wet[-1, :])[0]}
for side, jrow in (("S", 0), ("N", ny - 1)):
    ob[side] = np.array([i for i in ob[side]
                         if not ((i == 0 and jrow in ob["W"]) or (i == nx - 1 and jrow in ob["E"]))], int)
ob = {k: [int(v) for v in q] for k, q in ob.items() if len(q)}

os.makedirs(os.path.join(a.enfant, "input"), exist_ok=True)
shutil.copy(os.path.join(a.sortie, "child", "bathy.bin"), os.path.join(a.enfant, "input", "bathy.bin"))
desc = dict(nx=nx, ny=ny, dx=dx, dy=dx,
            x0=float(gc["xg"][0] - gp["xg"][0]), y0=float(gc["yg"][0] - gp["yg"][0]),
            dz=[float(v) for v in gc["dz"]], hFacMin=a.hfacmin, hFacMinDr=a.hfacmindr,
            bathy="input/bathy.bin", ob=ob, t0=a.t0, duree=a.cycles * T_M2, periode=a.periode,
            zstar=not a.nh, nonhydrostatique=a.nh, sponge=a.eponge)
json.dump(desc, open(os.path.join(a.enfant, "enfant.json"), "w"), indent=1)
nrec = int(round(desc["duree"] / a.periode)) + 2
print(f"enfant {nx}x{ny} dx={dx:g} m, x0={desc['x0']:.0f} y0={desc['y0']:.0f} m dans le repere du parent")
print("OB :", {k: len(v) for k, v in ob.items()})
print(f"data : externForcingPeriod = {a.periode:.1f}, externForcingCycle = {nrec * a.periode:.1f}")
print("-> " + os.path.join(a.enfant, "enfant.json"))
