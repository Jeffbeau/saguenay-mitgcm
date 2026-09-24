#!/usr/bin/env python3
"""
Enfant du cas test synthetique : sous-domaine raffine, force par le parent via OBCS hors ligne.

Config A : hydrostatique, z* (memes choix que le parent).
Config B (--nh) : non hydrostatique, surface libre lineaire (le solveur 3D refuse
nonlinFreeSurf != 0 dans checkpoint69k), GGL90viscMax plafonne car la viscosite
verticale est explicite sur w (calc_gw.F).

Ce script ecrit la grille et la bathymetrie ; imbrication/config_enfant.py ecrit les
namelists, code/ et enfant.json.
Les fichiers OBCS et les conditions initiales viennent ensuite de
imbrication/extraire_obcs.py (a partir des sorties du parent).

Usage : python3 gen_enfant.py [--ratio 2] [--nh] [--out enfant_A]
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "imbrication"))
import geometrie as geo
import config_enfant as cfg

p = argparse.ArgumentParser()
p.add_argument("--ratio", type=int, default=2, help="dx parent / dx enfant")
p.add_argument("--box", type=float, nargs=4, default=[10000, 33600, 4000, 19200],
               metavar=("X0", "X1", "Y0", "Y1"), help="emprise (m, multiples de dx parent)")
p.add_argument("--dxp", type=float, default=400.0, help="dx du parent")
p.add_argument("--nr", type=int, default=32)
p.add_argument("--dt", type=float, default=15.0)
p.add_argument("--ncyc", type=float, default=2.0, help="duree (cycles M2)")
p.add_argument("--t0", type=float, default=44640.0, help="instant de depart dans le temps du parent (s)")
p.add_argument("--period", type=float, default=930.0, help="espacement des enregistrements OBCS (s)")
p.add_argument("--nh", action="store_true", help="config B : non hydrostatique, surface libre lineaire")
p.add_argument("--nsx", type=int, default=2)
p.add_argument("--parent-input", default=os.path.join(HERE, "input"))
p.add_argument("--mitgcm", default=os.path.expanduser("~/MITgcm"))
p.add_argument("--out", default=None, help="dossier de sortie (defaut enfant_A ou enfant_B)")
args = p.parse_args()

OUT = os.path.join(HERE, args.out or ("enfant_B" if args.nh else "enfant_A"))

T_M2 = 44640.0
F0 = 1.087e-4
HFACMIN, HFACMINDR = 0.3, 2.0
NEST_COPY, NEST_BLEND = 2, 2          # bandes (cellules parent) : copie exacte, puis melange
R, DXP, DT, P = args.ratio, args.dxp, args.dt, args.period
DX = DXP / R
X0, X1, Y0, Y1 = args.box
for v in args.box:
    assert abs(v / DXP - round(v / DXP)) < 1e-9, "emprise non alignee sur les faces du parent"
NX, NY, NR = int(round((X1 - X0) / DX)), int(round((Y1 - Y0) / DX)), args.nr
DUR = args.ncyc * T_M2

# --------------------------------------------------------------- grille, bathy
dz, zf, zc = geo.niveaux(NR)
x = X0 + (np.arange(NX) + 0.5) * DX
y = Y0 + (np.arange(NY) + 0.5) * DX
X, Y = np.meshgrid(x, y)
H = geo.profondeur(X, Y)

# bathymetrie du parent (meme fichier que le run parent) copiee pres des bords
NXP, NYP = int(round(geo.LX / DXP)), int(round(geo.LY / DXP))
Hp = -np.fromfile(os.path.join(args.parent_input, "bathy.bin"), ">f8").reshape(NYP, NXP)
Hp_up = Hp[(Y // DXP).astype(int), (X // DXP).astype(int)]
ii, jj = np.meshgrid(np.arange(NX), np.arange(NY))
dedge = np.minimum.reduce([ii + 0.5, NX - ii - 0.5, jj + 0.5, NY - jj - 0.5]) / R   # cellules parent
w = np.clip((dedge - NEST_COPY) / NEST_BLEND, 0.0, 1.0)
H = np.where(w == 0.0, Hp_up, (1 - w) * Hp_up + w * H)
H = np.where(H < 10.0, 0.0, H)
hfac = lambda h: geo.hfac_c(h, dz, HFACMIN, HFACMINDR)
wet = hfac(H)[0] > 0
for _ in range(3):
    nb = np.zeros_like(wet, dtype=int)
    nb[1:, :] += wet[:-1, :]; nb[:-1, :] += wet[1:, :]
    nb[:, 1:] += wet[:, :-1]; nb[:, :-1] += wet[:, 1:]
    bad = wet & (nb < 2) & (w > 0)
    H[bad] = 0.0
    wet = hfac(H)[0] > 0

# ------------------------------------------- frontieres, namelists, code/, enfant.json
tsref = cfg.tsref_du_parent(os.path.join(args.parent_input, "data"))
cfg.ecrire_config(OUT, H=H, dx=DX, dz=dz, x0=X0, y0=Y0, ob=cfg.frontieres(wet), nh=args.nh,
                  dt=DT, t0=args.t0, duree=DUR, periode=P, tsref=tsref, mitgcm=args.mitgcm,
                  titre="Enfant du cas test", phys=dict(f0=F0, hFacMin=HFACMIN, hFacMinDr=HFACMINDR),
                  nsx=args.nsx)
