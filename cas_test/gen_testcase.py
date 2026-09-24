#!/usr/bin/env python3
"""
Cas test synthetique "mini" du Saguenay pour MITgcm checkpoint69k.

Meme emprise que le profil mini (37,6 x 21,6 km), memes choix numeriques
(z*, staggerTimeStep, GGL90, OBCS par pompage + riviere, eponge, adv. 33,
Smagorinsky 2.2, parois glissantes, Cd 2.5e-3, dt 30 s), mais bathymetrie
idealisee (fjord a 3 seuils 124 / 68 / 23 m + estuaire) et resolution
parametrable (400 m par defaut pour tourner sur 1 coeur).

Usage :  python3 gen_testcase.py [--dx 400] [--nr 32] [--ncyc 4]
Ecrit ./code/ et ./input/ a cote du script.
"""
import argparse, os, textwrap
import numpy as np
from scipy.interpolate import PchipInterpolator

p = argparse.ArgumentParser()
p.add_argument("--dx", type=float, default=400.0)
p.add_argument("--nr", type=int, default=32)
p.add_argument("--ncyc", type=int, default=4)
p.add_argument("--dt", type=float, default=30.0)
p.add_argument("--nsx", type=int, default=2, help="tuiles par processus en x")
p.add_argument("--mitgcm", default=os.path.expanduser("~/MITgcm"), help="racine MITgcm (options .h)")
args = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CODE, INP = os.path.join(HERE, "code"), os.path.join(HERE, "input")
os.makedirs(CODE, exist_ok=True); os.makedirs(INP, exist_ok=True)

# ----------------------------------------------------------------- constantes
LX, LY = 37600.0, 21600.0
DX = args.dx
NX, NY, NR = int(round(LX / DX)), int(round(LY / DX)), args.nr
assert NX % args.nsx == 0, "NX doit etre divisible par nsx"
T_M2 = 44640.0            # periode modele (s) : 1488 pas de 30 s
DT = args.dt
NCYC = args.ncyc
PFORC = 1860.0            # espacement des enregistrements OBCS (24 / cycle)
A_TIDE = 1.6              # amplitude M2 (m)
Q_RIV = 1200.0            # debit (m3/s)
A_AMONT = 213e6           # surface du fjord en amont de l'OB ouest (m2)
F0 = 1.087e-4             # 48.2 N
HFACMIN, HFACMINDR = 0.3, 2.0
OMEGA = 2 * np.pi / T_M2
assert abs(T_M2 / DT - round(T_M2 / DT)) < 1e-9

# ------------------------------------------------------------ grille verticale
dz = np.geomspace(1.5, 20.0, NR)
dz *= 280.0 / dz.sum()
dz = np.round(dz, 3)
zf = np.concatenate([[0.0], np.cumsum(dz)])          # faces (positives vers le bas)
zc = 0.5 * (zf[:-1] + zf[1:])

# --------------------------------------------------------- bathymetrie (m > 0)
x = (np.arange(NX) + 0.5) * DX
y = (np.arange(NY) + 0.5) * DX
X, Y = np.meshgrid(x, y)                              # (NY, NX)

# profil du talweg : noeuds (km, m) ; PCHIP conserve exactement les cols
xk = np.array([0, 3, 5.0, 7, 13, 16.0, 19, 26, 29.0, 31, 40])
hk = np.array([250, 250, 124, 200, 190, 68, 120, 110, 23, 60, 150])
h_axis = PchipInterpolator(xk * 1e3, hk)
y_axis = lambda xx: 15e3 - 3e3 * (np.clip(xx, 0, 30e3) / 30e3) ** 2
def half_width(xx):
    w = np.interp(xx, [0, 20e3, 27e3, 30e3, 40e3], [1000, 1000, 800, 600, 600])
    return w
d = np.abs(Y - y_axis(X))
hw = half_width(X)
H_f = np.where((d < hw) & (X <= 31.5e3),
               h_axis(X) * np.clip(1 - (d / hw) ** 2, 0, 1) ** 0.4, 0.0)
# estuaire : au sud-est d'une rive nord orientee a 40 deg passant par l'embouchure
th = np.deg2rad(40.0)
s = -np.sin(th) * (X - 30e3) + np.cos(th) * (Y - 12e3)   # s < 0 : estuaire
doff = np.clip(-s, 0, None)
H_e = np.where(s < 0, 15 + 255 * np.tanh(doff / 3e3) ** 1.5, 0.0)
H = np.maximum(H_f, H_e)
H = np.where(H < 10.0, 0.0, np.minimum(H, 270.0))
H[:, 0] = np.where(np.abs(y - y_axis(0)) < half_width(0), H[:, 0], 0.0)

# ----------------------------------------------- hFac comme MITgcm (z-levels)
def hfac_c(Hc):
    hf = np.zeros((NR,) + Hc.shape)
    for k in range(NR):
        mn = max(HFACMIN, min(HFACMINDR / dz[k], 1.0))
        t = np.clip((Hc - zf[k]) / dz[k], 0.0, 1.0)
        t = np.where(t < mn, np.where(t < 0.5 * mn, 0.0, mn), t)
        hf[k] = t
    return hf
hC = hfac_c(H)
wet = hC[0] > 0
# retirer les points isoles (moins de 2 voisins mouilles) : evite les puits a 1 point
for _ in range(3):
    nb = np.zeros_like(wet, dtype=int)
    nb[1:, :] += wet[:-1, :]; nb[:-1, :] += wet[1:, :]
    nb[:, 1:] += wet[:, :-1]; nb[:, :-1] += wet[:, 1:]
    bad = wet & (nb < 2)
    H[bad] = 0.0
    hC = hfac_c(H); wet = hC[0] > 0
A_mini = wet.sum() * DX * DX

# aires des faces ouvertes (indices Python 0-based)
A_W = (np.minimum(hC[:, :, 0], hC[:, :, 1]) * dz[:, None]).sum() * DX        # u(i=1)
A_E = (np.minimum(hC[:, :, NX - 2], hC[:, :, NX - 1]) * dz[:, None]).sum() * DX  # u(i=NX-1)
hS = np.minimum(hC[:, 0, :], hC[:, 1, :])                                       # v(j=1)
hS[:, NX - 1] = 0.0            # pas d'OB sud dans la colonne du coin SE (deja OB est)
A_S = (hS * dz[:, None]).sum() * DX

# ---------------------------------------------------------------- T / S initiaux
S_f = 30.5 - 22.0 * np.exp(-zc / 4.0)
T_f = 1.5 + 12.0 * np.exp(-zc / 6.0)
S_e = 33.8 - 7.5 * np.exp(-zc / 25.0)
T_e = 4.3 - 3.8 * np.exp(-((zc - 70.0) / 45.0) ** 2) + 2.0 * np.exp(-zc / 10.0)
w_e = 0.5 * (1 - np.tanh(s / 1.5e3))            # 1 dans l'estuaire, 0 dans le fjord
w_e[X < 27e3] = 0.0
T3 = (1 - w_e)[None] * T_f[:, None, None] + w_e[None] * T_e[:, None, None]
S3 = (1 - w_e)[None] * S_f[:, None, None] + w_e[None] * S_e[:, None, None]
vol = hC * dz[:, None, None]
tRef = (T3 * vol).sum((1, 2)) / np.maximum(vol.sum((1, 2)), 1e-9)
sRef = (S3 * vol).sum((1, 2)) / np.maximum(vol.sum((1, 2)), 1e-9)
tRef = np.where(vol.sum((1, 2)) > 0, tRef, T_e); sRef = np.where(vol.sum((1, 2)) > 0, sRef, S_e)

# --------------------------------------------- forcage OBCS (convention MITgcm)
# Enregistrement n (1-based) = instant t_n = (n - 1/2) * PFORC.
# Pour 0 <= t < PFORC/2, MITgcm interpole entre le DERNIER enregistrement et le
# premier (sequence periodique) : on ajoute donc 2 enregistrements a la fin,
# N+1 = suite logique, N+2 = etat avant le depart (t = -PFORC/2, riviere seule).
nrun = int(round(NCYC * T_M2 / PFORC))
tt = (np.arange(1, nrun + 2) - 0.5) * PFORC
tt = np.concatenate([tt, [-0.5 * PFORC]])
NREC = tt.size

def ramp(t):
    t = np.asarray(t, float)
    r = np.where(t <= 0, 0.0, np.where(t < T_M2, 0.5 * (1 - np.cos(np.pi * t / T_M2)), 1.0))
    dr = np.where((t > 0) & (t < T_M2), 0.5 * np.pi / T_M2 * np.sin(np.pi * t / T_M2), 0.0)
    return r, dr
r, dr = ramp(tt)
eta_t = A_TIDE * r * np.sin(OMEGA * tt)
deta = A_TIDE * (dr * np.sin(OMEGA * tt) + r * OMEGA * np.cos(OMEGA * tt))
Q_W = Q_RIV - A_AMONT * deta                  # entrant positif
Q_ES = (A_mini + A_AMONT) * deta - Q_RIV      # entrant positif
uW = Q_W / A_W
cES = Q_ES / (A_E + A_S)

def wr(name, arr):
    np.asarray(arr, dtype=">f8").tofile(os.path.join(INP, name))

wr("bathy.bin", -H)
wr("T_ini.bin", T3); wr("S_ini.bin", S3)
ones_yz = np.ones((NR, NY)); ones_xz = np.ones((NR, NX))
wr("OBWu.bin", uW[:, None, None] * ones_yz[None])
wr("OBEu.bin", -cES[:, None, None] * ones_yz[None])
wr("OBSv.bin", cES[:, None, None] * ones_xz[None])
wr("OBWt.bin", np.repeat((T_f[:, None] * ones_yz)[None], NREC, 0))
wr("OBWs.bin", np.repeat((S_f[:, None] * ones_yz)[None], NREC, 0))
wr("OBEt.bin", np.repeat((T_e[:, None] * ones_yz)[None], NREC, 0))
wr("OBEs.bin", np.repeat((S_e[:, None] * ones_yz)[None], NREC, 0))
wr("OBSt.bin", np.repeat((T_e[:, None] * ones_xz)[None], NREC, 0))
wr("OBSs.bin", np.repeat((S_e[:, None] * ones_xz)[None], NREC, 0))
np.savez(os.path.join(INP, "forcing_check.npz"), t=tt, eta=eta_t, Q_W=Q_W, Q_ES=Q_ES,
         A_W=A_W, A_E=A_E, A_S=A_S, A_mini=A_mini)

# -------------------------------------------------------------- namelists
def nml(txt):
    """MITgcm reconnait le terminateur ' &' seulement avec une espace en tete."""
    txt = textwrap.dedent(txt)
    return "\n".join((" " + l) if (l and not l.startswith(" ")) else l
                     for l in txt.splitlines()) + "\n"

def flist(v, fmt="{:.3f}", per=6):
    items = [fmt.format(a) for a in v]
    return ",\n  ".join(", ".join(items[i:i + per]) for i in range(0, len(items), per)) + ","

# niveaux pour les sorties 2D (~1, 10, 30 m)
lev = [int(np.argmin(np.abs(zc - z))) + 1 for z in (1.0, 10.0, 30.0)]
nsteps = int(round(NCYC * T_M2 / DT))
spong = int(round(3000.0 / DX))

open(os.path.join(INP, "data"), "w").write(nml(f"""\
 # Cas test synthetique mini - Config A (hydrostatique, z*)
 &PARM01
 tRef = {flist(tRef)}
 sRef = {flist(sRef)}
 eosType = 'JMD95Z',
 rhoConst = 1025.,
 rhoNil = 1025.,
 gravity = 9.81,
 f0 = {F0:.4e},
 beta = 0.,
 viscAh = 0.,
 viscC2smag = 2.2,
 viscAr = 1.E-5,
 diffKhT = 0., diffKhS = 0.,
 diffKrT = 1.E-6, diffKrS = 1.E-6,
 no_slip_sides = .FALSE.,
 no_slip_bottom = .FALSE.,
 bottomDragQuadratic = 2.5E-3,
 vectorInvariantMomentum = .TRUE.,
 implicitViscosity = .TRUE.,
 implicitDiffusion = .TRUE.,
 tempAdvScheme = 33,
 saltAdvScheme = 33,
 staggerTimeStep = .TRUE.,
 implicitFreeSurface = .TRUE.,
 exactConserv = .TRUE.,
 nonlinFreeSurf = 4,
 select_rStar = 2,
 hFacInf = 0.1,
 hFacSup = 5.,
 hFacMin = {HFACMIN},
 hFacMinDr = {HFACMINDR},
 readBinaryPrec = 64,
 writeBinaryPrec = 32,
 useSingleCpuIO = .FALSE.,
 debugLevel = 0,
 &
 &PARM02
 cg2dMaxIters = 500,
 cg2dTargetResWunit = 1.E-12,
 &
 &PARM03
 nIter0 = 0,
 nTimeSteps = {nsteps},
 deltaT = {DT:.1f},
 abEps = 0.1,
 pChkptFreq = {T_M2:.1f},
 chkptFreq = 0.,
 dumpFreq = 0.,
 monitorFreq = 1860.,
 monitorSelect = 2,
 periodicExternalForcing = .TRUE.,
 externForcingPeriod = {PFORC:.1f},
 externForcingCycle = {NREC * PFORC:.1f},
 &
 &PARM04
 usingCartesianGrid = .TRUE.,
 delX = {NX}*{DX:.1f},
 delY = {NY}*{DX:.1f},
 delR = {flist(dz)}
 &
 &PARM05
 bathyFile = 'bathy.bin',
 hydrogThetaFile = 'T_ini.bin',
 hydrogSaltFile = 'S_ini.bin',
 &
"""))

open(os.path.join(INP, "eedata"), "w").write(" &EEPARMS\n nTx=1,\n nTy=1,\n &\n")
open(os.path.join(INP, "data.pkg"), "w").write(
    " &PACKAGES\n useOBCS = .TRUE.,\n useGGL90 = .TRUE.,\n useDiagnostics = .TRUE.,\n &\n")
open(os.path.join(INP, "data.ggl90"), "w").write(nml("""\
 &GGL90_PARM01
 GGL90TKEmin = 1.E-7,
 GGL90viscMax = 1.,
 GGL90diffMax = 1.,
 mxlMaxFlag = 2,
 &
"""))
jS = f"{NX - 1}*1, 0"  # lignes < 200 car. (MITgcm tronque au-dela)
open(os.path.join(INP, "data.obcs"), "w").write(nml(f"""\
 &OBCS_PARM01
 OB_Iwest = {NY}*1,
 OB_Ieast = {NY}*{NX},
 OB_Jsouth = {jS},
 useOBCSprescribe = .TRUE.,
 useOBCSsponge = .TRUE.,
 useOBCSbalance = .FALSE.,
 OBWuFile = 'OBWu.bin', OBWtFile = 'OBWt.bin', OBWsFile = 'OBWs.bin',
 OBEuFile = 'OBEu.bin', OBEtFile = 'OBEt.bin', OBEsFile = 'OBEs.bin',
 OBSvFile = 'OBSv.bin', OBStFile = 'OBSt.bin', OBSsFile = 'OBSs.bin',
 &
 &OBCS_PARM03
 spongeThickness = {spong},
 Urelaxobcsbound = 1800.,
 Urelaxobcsinner = {T_M2:.1f},
 Vrelaxobcsbound = 1800.,
 Vrelaxobcsinner = {T_M2:.1f},
 &
"""))
open(os.path.join(INP, "data.diagnostics"), "w").write(nml(f"""\
 # Frequences = diviseurs de la periode (44640 s) ET multiples de dt (30 s)
 # -> chaque sortie tombe exactement a la meme phase d'un cycle a l'autre.
 &DIAGNOSTICS_LIST
 dumpAtLast = .FALSE.,
 fields(1,1) = 'ETAN    ',
   fileName(1) = 'eta2D',
   frequency(1) = -360.,
   timePhase(1) = 0.,
 fields(1:3,2) = 'UVEL    ','VVEL    ','momVort3',
   levels(1:3,2) = {lev[0]}., {lev[1]}., {lev[2]}.,
   fileName(2) = 'lev2D',
   frequency(2) = -360.,
   timePhase(2) = 0.,
 fields(1:6,3) = 'UVEL    ','VVEL    ','WVEL    ','THETA   ','SALT    ','RHOAnoma',
   fileName(3) = 'state3D',
   frequency(3) = -930.,
   timePhase(3) = 0.,
 fields(1:8,4) = 'UVEL    ','VVEL    ','WVEL    ','THETA   ','SALT    ',
                 'UVELSQ  ','VVELSQ  ','UV_VEL_C',
   fileName(4) = 'mean3D',
   frequency(4) = {T_M2:.1f},
 fields(1:6,5) = 'UVELMASS','VVELMASS','WVELMASS','USLTMASS','VSLTMASS','WSLTMASS',
   fileName(5) = 'flux3D',
   frequency(5) = {T_M2:.1f},
 &
 &DIAG_STATIS_PARMS
 &
"""))

# ----------------------------------------------------------------- code/
open(os.path.join(CODE, "SIZE.h"), "w").write(textwrap.dedent(f"""\
CBOP
C    !ROUTINE: SIZE.h
C    cas test synthetique mini ({NX}x{NY}x{NR}, dx={DX:.0f} m)
CEOP
      INTEGER sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nx, Ny, Nr
      PARAMETER (
     &           sNx = {NX // args.nsx},
     &           sNy = {NY},
     &           OLx =   4,
     &           OLy =   4,
     &           nSx =   {args.nsx},
     &           nSy =   1,
     &           nPx =   1,
     &           nPy =   1,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = {NR})
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx, MAX_OLY = OLy )
"""))
open(os.path.join(CODE, "packages.conf"), "w").write("gfd\nobcs\nggl90\ndiagnostics\n")
for src_h, dst_h, old, new in (
        ("model/inc/CPP_OPTIONS.h", "CPP_OPTIONS.h", "#undef NONLIN_FRSURF", "#define NONLIN_FRSURF"),
        ("pkg/obcs/OBCS_OPTIONS.h", "OBCS_OPTIONS.h", "#undef ALLOW_OBCS_SPONGE", "#define ALLOW_OBCS_SPONGE"),
        ("pkg/diagnostics/DIAGNOSTICS_SIZE.h", "DIAGNOSTICS_SIZE.h", "numDiags = 1*Nr", "numDiags = 24*Nr")):
    fsrc = os.path.join(args.mitgcm, src_h)
    if os.path.exists(fsrc):
        txt = open(fsrc).read()
        assert old in txt, (src_h, old)
        open(os.path.join(CODE, dst_h), "w").write(txt.replace(old, new))
    else:
        print("ATTENTION: options .h non copiees, --mitgcm introuvable:", fsrc)

print(f"grille {NX}x{NY}x{NR} dx={DX} m ; dz1={dz[0]:.2f} m ; Hmax={H.max():.0f} m")
print(f"A_mini={A_mini/1e6:.1f} km2  A_W={A_W:.3e}  A_E={A_E:.3e}  A_S={A_S:.3e} m2")
print(f"u_W max={np.abs(uW).max():.3f} m/s ; c_ES max={np.abs(cES).max():.4f} m/s")
print(f"NREC={NREC} externForcingCycle={NREC*PFORC:.0f} s ; nTimeSteps={nsteps}")
print(f"niveaux 2D (~1/10/30 m) : {lev} ; eponge {spong} cellules")
print("cols (talweg) :", [f"{h_axis(v*1e3):.1f}" for v in (5, 16, 29)])
