#!/usr/bin/env python3
"""
Configuration MITgcm complete du vrai enfant a partir des sorties du pipeline grille (s02/s03).

Ecrit dans ENFANT_DIR : input/ (namelists, bathy.bin), code/ (SIZE.h, options, correctif z*)
et enfant.json. Les parametres physiques (f0, hFac*) et tRef/sRef sont lus dans le fichier
data du run parent, pour que l'enfant ait le meme etat de reference.

Le repere du parent MITgcm commence a sa face sud-ouest (usingCartesianGrid, XG = 0) :
x0, y0 de l'enfant = faces de l'enfant - faces du parent (UTM 19N).

Usage (sur la VM, apres SAG_PROFILE=mini python run_all.py et le run du parent) :
  python3 imbrication/enfant_depuis_pipeline.py pipeline_grille/output_mini ~/runs/enfantB \\
      --parent ~/runs/mini --t0 89280 --cycles 2 --nh --npx 2
  python3 imbrication/extraire_obcs.py ~/runs/mini ~/runs/enfantB
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sagdiag"))
import config_enfant as cfg
import sagdiag as sd

ap = argparse.ArgumentParser()
ap.add_argument("sortie", help="dossier de sortie du pipeline (contient parent/ et child/)")
ap.add_argument("enfant", help="dossier de l'enfant (cree)")
ap.add_argument("--parent", required=True, help="dossier du run parent (son fichier data)")
ap.add_argument("--t0", type=float, required=True, help="depart dans le temps du parent (s), multiple de 44640")
ap.add_argument("--cycles", type=float, default=2.0)
ap.add_argument("--dt", type=float, default=None, help="pas de temps (defaut : 5 s a 50 m ou moins, sinon 15 s)")
ap.add_argument("--periode", type=float, default=930.0, help="espacement des enregistrements OBCS (s)")
ap.add_argument("--nh", action="store_true", help="config B : non hydrostatique, surface libre lineaire")
ap.add_argument("--npx", type=int, default=1, help="processus MPI en x (mpirun -np NPX)")
ap.add_argument("--nsx", type=int, default=1, help="tuiles par processus en x")
ap.add_argument("--eponge", type=float, default=1000.0, help="epaisseur de l'eponge (m)")
ap.add_argument("--mitgcm", default=os.path.expanduser("~/MITgcm"))
a = ap.parse_args()

gp = np.load(os.path.join(a.sortie, "parent", "grid.npz"))
gc = np.load(os.path.join(a.sortie, "child", "grid.npz"))
dx = float(gc["xg"][1] - gc["xg"][0])
dxp = float(gp["xg"][1] - gp["xg"][0])
x0 = float(gc["xg"][0] - gp["xg"][0]); y0 = float(gc["yg"][0] - gp["yg"][0])
for v in (x0, y0):
    assert abs(v / dxp - round(v / dxp)) < 1e-6, "faces de l'enfant non alignees sur celles du parent"
H = np.where(gc["wet"], gc["depth"], 0.0)
Hb = -np.fromfile(os.path.join(a.sortie, "child", "bathy.bin"), ">f8").reshape(H.shape)
assert np.allclose(H, Hb), "grid.npz et bathy.bin de l'enfant different"

data_p = os.path.join(a.parent, "data")
nv = lambda n, d: sd.nml_value(data_p, n, d)
phys = dict(f0=nv("f0", 1.0e-4), hFacMin=nv("hFacMin", 0.2), hFacMinDr=nv("hFacMinDr", 0.5),
            hFacInf=nv("hFacInf", 0.2), hFacSup=nv("hFacSup", 2.0))
dt = a.dt or (5.0 if dx <= 50 else 15.0)
cfg.ecrire_config(a.enfant, H=H, dx=dx, dz=np.asarray(gc["dz"], float), x0=x0, y0=y0,
                  ob=cfg.frontieres(H > 0), nh=a.nh, dt=dt, t0=a.t0, duree=a.cycles * cfg.T_M2,
                  periode=a.periode, tsref=cfg.tsref_du_parent(data_p), mitgcm=a.mitgcm,
                  titre=f"Enfant Saguenay {dx:g} m", phys=phys, nsx=a.nsx, npx=a.npx, eponge_m=a.eponge)
print(f"x0={x0:.0f} y0={y0:.0f} m dans le repere du parent (rapport {dxp / dx:g})")
print(f"Suite : python3 imbrication/extraire_obcs.py {a.parent} {a.enfant}")
