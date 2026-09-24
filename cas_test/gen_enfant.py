#!/usr/bin/env python3
"""
Enfant du cas test synthetique : sous-domaine raffine, force par le parent via OBCS hors ligne.

Config A : hydrostatique, z* (memes choix que le parent).
Config B (--nh) : non hydrostatique, surface libre lineaire (le solveur 3D refuse
nonlinFreeSurf != 0 dans checkpoint69k), GGL90viscMax plafonne car la viscosite
verticale est explicite sur w (calc_gw.F).

Ce script ecrit la grille, la bathymetrie, les namelists, code/ et enfant.json.
Les fichiers OBCS et les conditions initiales viennent ensuite de
imbrication/extraire_obcs.py (a partir des sorties du parent).

Usage : python3 gen_enfant.py [--ratio 2] [--nh] [--out enfant_A]
"""
import argparse, json, os, re, sys, textwrap
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometrie as geo

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
CODE, INP = os.path.join(OUT, "code"), os.path.join(OUT, "input")
os.makedirs(CODE, exist_ok=True); os.makedirs(INP, exist_ok=True)

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
assert NX % args.nsx == 0, "NX doit etre divisible par nsx"
for q in (P, 360.0, 930.0, T_M2):
    assert abs(q / DT - round(q / DT)) < 1e-9, f"{q} s n'est pas un multiple de dt"
assert abs(args.t0 / T_M2 - round(args.t0 / T_M2)) < 1e-9, "t0 doit etre un multiple de T_M2 (phase 0)"
DUR = args.ncyc * T_M2
NSTEP = int(round(DUR / DT))

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

# ------------------------------------------------ frontieres ouvertes (0-based)
ob = {"W": [int(j) for j in np.where(wet[:, 0])[0]],
      "E": [int(j) for j in np.where(wet[:, -1])[0]],
      "S": [int(i) for i in np.where(wet[0, :])[0]],
      "N": [int(i) for i in np.where(wet[-1, :])[0]]}
# coins : attribues aux frontieres E/O
for side, jrow in (("S", 0), ("N", NY - 1)):
    ob[side] = [i for i in ob[side] if not ((i == 0 and jrow in ob["W"]) or (i == NX - 1 and jrow in ob["E"]))]
ob = {k: v for k, v in ob.items() if v}

def rle(vals):
    """Liste Fortran compacte N*v (MITgcm tronque les lignes de plus de ~200 car.)."""
    out, i = [], 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        out.append(f"{j - i}*{vals[i]}" if j - i > 1 else f"{vals[i]}")
        i = j
    lines = [""]
    for tok in out:
        if len(lines[-1]) + len(tok) > 60:
            lines.append("")
        lines[-1] += tok + ", "
    return "\n   ".join(l.rstrip() for l in lines)

obidx = {"W": ("OB_Iwest", NY, 1), "E": ("OB_Ieast", NY, NX),
         "S": ("OB_Jsouth", NX, 1), "N": ("OB_Jnorth", NX, NY)}
ob_lines = []
for k, pts in ob.items():
    name, n, val = obidx[k]
    v = [0] * n
    for q in pts:
        v[q] = val
    ob_lines.append(f" {name} = {rle(v)}")

# --------------------------------------------------------------- namelists
def nml(txt):
    txt = textwrap.dedent(txt)
    return "\n".join((" " + l) if (l and not l.startswith(" ")) else l
                     for l in txt.splitlines()) + "\n"

def flist(v, fmt="{:.3f}", per=6):
    items = [fmt.format(a) for a in v]
    return ",\n  ".join(", ".join(items[i:i + per]) for i in range(0, len(items), per)) + ","

ptxt = open(os.path.join(args.parent_input, "data")).read()
tsref = re.search(r"(tRef\s*=.*?)(?=\n\s*eosType)", ptxt, re.S).group(1).strip()

nrec = int(round(DUR / P)) + 2
lev = [int(np.argmin(np.abs(zc - z))) + 1 for z in (1.0, 10.0, 30.0)]
spong = int(round(1000.0 / DX))
dzmin = float(dz.min())
# stabilite de la diffusion verticale explicite de w : nu*dt/dz^2 <= 0.2
viscmax = 1.0 if not args.nh else float(f"{0.2 * dzmin ** 2 / DT:.2g}")

if args.nh:
    fs = f""" nonHydrostatic = .TRUE.,
 implicitFreeSurface = .TRUE.,
 exactConserv = .TRUE.,
 hFacMin = {HFACMIN},
 hFacMinDr = {HFACMINDR},"""
    solver = """ cg2dMaxIters = 500,
 cg2dTargetResWunit = 1.E-12,
 cg3dMaxIters = 400,
 cg3dTargetResWunit = 1.E-10,"""
else:
    fs = f""" implicitFreeSurface = .TRUE.,
 exactConserv = .TRUE.,
 nonlinFreeSurf = 4,
 select_rStar = 2,
 hFacInf = 0.1,
 hFacSup = 5.,
 hFacMin = {HFACMIN},
 hFacMinDr = {HFACMINDR},"""
    solver = """ cg2dMaxIters = 500,
 cg2dTargetResWunit = 1.E-12,"""

cfg = "B (non hydrostatique, surface libre lineaire)" if args.nh else "A (hydrostatique, z*)"
open(os.path.join(INP, "data"), "w").write(nml(f"""\
 # Enfant du cas test - Config {cfg}, dx = {DX:.0f} m
 # Temps 0 de l'enfant = {args.t0:.0f} s du parent (phase M2 nulle)
 &PARM01
 {tsref}
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
{fs}
 readBinaryPrec = 64,
 writeBinaryPrec = 32,
 useSingleCpuIO = .FALSE.,
 debugLevel = 0,
 &
 &PARM02
{solver}
 &
 &PARM03
 nIter0 = 0,
 nTimeSteps = {NSTEP},
 deltaT = {DT:.1f},
 abEps = 0.1,
 pChkptFreq = {T_M2:.1f},
 chkptFreq = 0.,
 dumpFreq = 0.,
 monitorFreq = 1860.,
 monitorSelect = 2,
 periodicExternalForcing = .TRUE.,
 externForcingPeriod = {P:.1f},
 externForcingCycle = {nrec * P:.1f},
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
 uVelInitFile = 'U_ini.bin',
 vVelInitFile = 'V_ini.bin',
 pSurfInitFile = 'Eta_ini.bin',
 &
"""))

files = []
for k in ob:
    norm, tang = ("u", "v") if k in "WE" else ("v", "u")
    q = [norm, tang, "t", "s"] + (["w"] if args.nh else ["eta"])
    files += [f"OB{k}{c}File = 'OB{k}{c}.bin'" for c in q]
open(os.path.join(INP, "data.obcs"), "w").write(nml(
    " &OBCS_PARM01\n" + "\n".join(ob_lines) + "\n"
    + " useOBCSprescribe = .TRUE.,\n useOBCSsponge = .TRUE.,\n useOBCSbalance = .FALSE.,\n"
    + "".join(f" {f},\n" for f in files) + " &\n"
    + f""" &OBCS_PARM03
 spongeThickness = {spong},
 Urelaxobcsbound = 1800.,
 Urelaxobcsinner = {T_M2:.1f},
 Vrelaxobcsbound = 1800.,
 Vrelaxobcsinner = {T_M2:.1f},
 &
"""))
open(os.path.join(INP, "eedata"), "w").write(" &EEPARMS\n nTx=1,\n nTy=1,\n &\n")
open(os.path.join(INP, "data.pkg"), "w").write(
    " &PACKAGES\n useOBCS = .TRUE.,\n useGGL90 = .TRUE.,\n useDiagnostics = .TRUE.,\n &\n")
open(os.path.join(INP, "data.ggl90"), "w").write(nml(f"""\
 &GGL90_PARM01
 GGL90TKEmin = 1.E-7,
 GGL90viscMax = {viscmax},
 GGL90diffMax = 1.,
 mxlMaxFlag = 2,
 &
"""))
open(os.path.join(INP, "data.diagnostics"), "w").write(nml(f"""\
 # Memes sorties que le parent : frequences multiples de dt qui divisent 44640 s
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
np.asarray(-H, dtype=">f8").tofile(os.path.join(INP, "bathy.bin"))

# ----------------------------------------------------------------------- code/
def ecrire(path, txt):
    """N'ecrit que si le contenu change (evite de recompiler a chaque generation)."""
    if not (os.path.exists(path) and open(path).read() == txt):
        open(path, "w").write(txt)

ecrire(os.path.join(CODE, "SIZE.h"), textwrap.dedent(f"""\
CBOP
C    !ROUTINE: SIZE.h
C    enfant du cas test ({NX}x{NY}x{NR}, dx={DX:.0f} m)
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
ecrire(os.path.join(CODE, "packages.conf"), "gfd\nobcs\nggl90\ndiagnostics\n")
cpp = [("#undef NONLIN_FRSURF", "#define NONLIN_FRSURF")]
if args.nh:
    cpp.append(("#undef ALLOW_NONHYDROSTATIC", "#define ALLOW_NONHYDROSTATIC"))
for src_h, dst_h, subs in (
        ("model/inc/CPP_OPTIONS.h", "CPP_OPTIONS.h", cpp),
        ("pkg/obcs/OBCS_OPTIONS.h", "OBCS_OPTIONS.h", [("#undef ALLOW_OBCS_SPONGE", "#define ALLOW_OBCS_SPONGE")]),
        ("pkg/diagnostics/DIAGNOSTICS_SIZE.h", "DIAGNOSTICS_SIZE.h", [("numDiags = 1*Nr", "numDiags = 24*Nr")])):
    fsrc = os.path.join(args.mitgcm, src_h)
    if not os.path.exists(fsrc):
        print("ATTENTION: options .h non copiees, --mitgcm introuvable:", fsrc)
        continue
    txt = open(fsrc).read()
    for old, new in subs:
        assert old in txt, (src_h, old)
        txt = txt.replace(old, new)
    ecrire(os.path.join(CODE, dst_h), txt)
# Bogue de checkpoint69k (obcs_apply_r_star.F) : aux OB N/S, le facteur z* lit OBNeta(j) et
# OBSeta(j) au lieu de OB[N,S]eta(i). Copie corrigee dans code/ (z* avec fichiers OB*eta).
fsrc = os.path.join(args.mitgcm, "pkg/obcs/obcs_apply_r_star.F")
fdst = os.path.join(CODE, "obcs_apply_r_star.F")
if args.nh:
    if os.path.exists(fdst):
        os.remove(fdst)
elif os.path.exists(fsrc):
    txt = open(fsrc).read()
    for obn in ("OBNeta", "OBSeta"):
        old = f"     &      + {obn}(  j,bi,bj) / (rSurfS(i,j,bi,bj)-rLowS(i,j,bi,bj))"
        assert txt.count(old) == 1, ("obcs_apply_r_star.F a change", obn)
        txt = txt.replace(old, old.replace(f"{obn}(  j,", f"{obn}(  i,"))
    ecrire(fdst, txt)

# ------------------------------------------------ description pour l'extraction
desc = dict(nx=NX, ny=NY, dx=DX, dy=DX, x0=X0, y0=Y0, dz=[float(v) for v in dz],
            hFacMin=HFACMIN, hFacMinDr=HFACMINDR, bathy="input/bathy.bin", ob=ob,
            t0=args.t0, duree=DUR, periode=P, zstar=not args.nh, nonhydrostatique=args.nh,
            sponge=spong)
json.dump(desc, open(os.path.join(OUT, "enfant.json"), "w"), indent=1)

print(f"Config {cfg}")
print(f"grille {NX}x{NY}x{NR} dx={DX:.0f} m, dt={DT:.0f} s, {NSTEP} pas ({args.ncyc:g} cycles)")
print("OB :", {k: len(v) for k, v in ob.items()}, f"; eponge {spong} cellules ; GGL90viscMax={viscmax}")
print(f"-> {OUT}")
