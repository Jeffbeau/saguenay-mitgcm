"""
Configuration MITgcm d'un enfant force hors ligne (namelists, code/, enfant.json).

Partage par cas_test/gen_enfant.py (cas test) et imbrication/enfant_depuis_pipeline.py
(vrai enfant). Les fichiers OBCS et les conditions initiales viennent ensuite de
imbrication/extraire_obcs.py.

Config A : hydrostatique, z* ; fichiers OB*eta et copie corrigee de obcs_apply_r_star.F
           (bogue de checkpoint69k aux OB N/S).
Config B : non hydrostatique, surface libre lineaire (le solveur 3D refuse nonlinFreeSurf != 0) ;
           fichiers OB*w ; GGL90viscMax <= 0.2 dz_min^2/dt (viscosite verticale explicite sur w).
"""
import json, os, re, textwrap
import numpy as np

T_M2 = 44640.0

# parametres physiques par defaut (ceux du cas test) ; le pipeline passe les siens
PHYS = dict(f0=1.087e-4, hFacMin=0.3, hFacMinDr=2.0, hFacInf=0.1, hFacSup=5.0)


def frontieres(wet):
    """Frontieres ouvertes = bords mouilles (indices 0-based) ; les coins vont aux frontieres E/O."""
    ny, nx = wet.shape
    ob = {"W": [int(j) for j in np.where(wet[:, 0])[0]],
          "E": [int(j) for j in np.where(wet[:, -1])[0]],
          "S": [int(i) for i in np.where(wet[0, :])[0]],
          "N": [int(i) for i in np.where(wet[-1, :])[0]]}
    for side, jrow in (("S", 0), ("N", ny - 1)):
        ob[side] = [i for i in ob[side]
                    if not ((i == 0 and jrow in ob["W"]) or (i == nx - 1 and jrow in ob["E"]))]
    return {k: v for k, v in ob.items() if v}


def tsref_du_parent(data_parent):
    """Lignes tRef/sRef du fichier data du parent (meme etat de reference)."""
    txt = open(data_parent).read()
    return re.search(r"(tRef\s*=.*?)(?=\n\s*eosType)", txt, re.S).group(1).strip()


def nml(txt):
    """MITgcm reconnait le terminateur ' &' seulement avec une espace en tete."""
    txt = textwrap.dedent(txt)
    return "\n".join((" " + l) if (l and not l.startswith(" ")) else l
                     for l in txt.splitlines()) + "\n"


def flist(v, fmt="{:.3f}", per=6):
    items = [fmt.format(a) for a in v]
    return ",\n  ".join(", ".join(items[i:i + per]) for i in range(0, len(items), per)) + ","


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


def ecrire(path, txt):
    """N'ecrit que si le contenu change (evite de recompiler a chaque generation)."""
    if not (os.path.exists(path) and open(path).read() == txt):
        open(path, "w").write(txt)


def ecrire_config(out, *, H, dx, dz, x0, y0, ob, nh, dt, t0, duree, periode, tsref,
                  mitgcm, titre, phys=None, nsx=1, npx=1, eponge_m=1000.0):
    """Ecrit out/input (namelists, bathy.bin), out/code et out/enfant.json.
    H (ny, nx) : profondeur positive (terre = 0) ; x0, y0 : coin sud-ouest dans le repere du parent."""
    ph = dict(PHYS, **(phys or {}))
    ny, nx = H.shape
    nr = len(dz)
    code, inp = os.path.join(out, "code"), os.path.join(out, "input")
    os.makedirs(code, exist_ok=True); os.makedirs(inp, exist_ok=True)
    assert nx % (nsx * npx) == 0, "Nx doit etre divisible par nSx*nPx"
    for q in (periode, 360.0, 930.0, T_M2):
        assert abs(q / dt - round(q / dt)) < 1e-9, f"{q} s n'est pas un multiple de dt"
    assert abs(t0 / T_M2 - round(t0 / T_M2)) < 1e-9, "t0 doit etre un multiple de T_M2 (phase 0)"
    nstep = int(round(duree / dt))
    nrec = int(round(duree / periode)) + 2
    zc = np.cumsum(dz) - 0.5 * np.asarray(dz)
    lev = [int(np.argmin(np.abs(zc - z))) + 1 for z in (1.0, 10.0, 30.0)]
    spong = int(round(eponge_m / dx))
    # stabilite de la diffusion verticale explicite de w : nu*dt/dz^2 <= 0.2
    viscmax = 1.0 if not nh else float(f"{0.2 * float(np.min(dz)) ** 2 / dt:.2g}")

    obidx = {"W": ("OB_Iwest", ny, 1), "E": ("OB_Ieast", ny, nx),
             "S": ("OB_Jsouth", nx, 1), "N": ("OB_Jnorth", nx, ny)}
    ob_lines = []
    for k, pts in ob.items():
        name, n, val = obidx[k]
        v = [0] * n
        for q in pts:
            v[q] = val
        ob_lines.append(f" {name} = {rle(v)}")

    if nh:
        fs = f""" nonHydrostatic = .TRUE.,
 implicitFreeSurface = .TRUE.,
 exactConserv = .TRUE.,
 hFacMin = {ph['hFacMin']},
 hFacMinDr = {ph['hFacMinDr']},"""
        solver = """ cg2dMaxIters = 500,
 cg2dTargetResWunit = 1.E-12,
 cg3dMaxIters = 400,
 cg3dTargetResWunit = 1.E-10,"""
    else:
        fs = f""" implicitFreeSurface = .TRUE.,
 exactConserv = .TRUE.,
 nonlinFreeSurf = 4,
 select_rStar = 2,
 hFacInf = {ph['hFacInf']},
 hFacSup = {ph['hFacSup']:g}.,
 hFacMin = {ph['hFacMin']},
 hFacMinDr = {ph['hFacMinDr']},"""
        solver = """ cg2dMaxIters = 500,
 cg2dTargetResWunit = 1.E-12,"""

    cfg = "B (non hydrostatique, surface libre lineaire)" if nh else "A (hydrostatique, z*)"
    open(os.path.join(inp, "data"), "w").write(nml(f"""\
 # {titre} - Config {cfg}, dx = {dx:.0f} m
 # Temps 0 de l'enfant = {t0:.0f} s du parent (phase M2 nulle)
 &PARM01
 {tsref}
 eosType = 'JMD95Z',
 rhoConst = 1025.,
 rhoNil = 1025.,
 gravity = 9.81,
 f0 = {ph['f0']:.4e},
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
 nTimeSteps = {nstep},
 deltaT = {dt:.1f},
 abEps = 0.1,
 pChkptFreq = {T_M2:.1f},
 chkptFreq = 0.,
 dumpFreq = 0.,
 monitorFreq = 1860.,
 monitorSelect = 2,
 periodicExternalForcing = .TRUE.,
 externForcingPeriod = {periode:.1f},
 externForcingCycle = {nrec * periode:.1f},
 &
 &PARM04
 usingCartesianGrid = .TRUE.,
 delX = {nx}*{dx:.1f},
 delY = {ny}*{dx:.1f},
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
        q = [norm, tang, "t", "s"] + (["w"] if nh else ["eta"])
        files += [f"OB{k}{c}File = 'OB{k}{c}.bin'" for c in q]
    open(os.path.join(inp, "data.obcs"), "w").write(nml(
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
    open(os.path.join(inp, "eedata"), "w").write(" &EEPARMS\n nTx=1,\n nTy=1,\n &\n")
    open(os.path.join(inp, "data.pkg"), "w").write(
        " &PACKAGES\n useOBCS = .TRUE.,\n useGGL90 = .TRUE.,\n useDiagnostics = .TRUE.,\n &\n")
    open(os.path.join(inp, "data.ggl90"), "w").write(nml(f"""\
 &GGL90_PARM01
 GGL90TKEmin = 1.E-7,
 GGL90viscMax = {viscmax},
 GGL90diffMax = 1.,
 mxlMaxFlag = 2,
 &
"""))
    open(os.path.join(inp, "data.diagnostics"), "w").write(nml(f"""\
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
    np.asarray(-H, dtype=">f8").tofile(os.path.join(inp, "bathy.bin"))

    # ------------------------------------------------------------------- code/
    ecrire(os.path.join(code, "SIZE.h"), textwrap.dedent(f"""\
CBOP
C    !ROUTINE: SIZE.h
C    {titre} ({nx}x{ny}x{nr}, dx={dx:.0f} m)
CEOP
      INTEGER sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nx, Ny, Nr
      PARAMETER (
     &           sNx = {nx // (nsx * npx)},
     &           sNy = {ny},
     &           OLx =   4,
     &           OLy =   4,
     &           nSx =   {nsx},
     &           nSy =   1,
     &           nPx =   {npx},
     &           nPy =   1,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = {nr})
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx, MAX_OLY = OLy )
"""))
    ecrire(os.path.join(code, "packages.conf"), "gfd\nobcs\nggl90\ndiagnostics\n")
    cpp = [("#undef NONLIN_FRSURF", "#define NONLIN_FRSURF")]
    if nh:
        cpp.append(("#undef ALLOW_NONHYDROSTATIC", "#define ALLOW_NONHYDROSTATIC"))
    for src_h, dst_h, subs in (
            ("model/inc/CPP_OPTIONS.h", "CPP_OPTIONS.h", cpp),
            ("pkg/obcs/OBCS_OPTIONS.h", "OBCS_OPTIONS.h", [("#undef ALLOW_OBCS_SPONGE", "#define ALLOW_OBCS_SPONGE")]),
            ("pkg/diagnostics/DIAGNOSTICS_SIZE.h", "DIAGNOSTICS_SIZE.h", [("numDiags = 1*Nr", "numDiags = 24*Nr")])):
        fsrc = os.path.join(mitgcm, src_h)
        if not os.path.exists(fsrc):
            print("ATTENTION: options .h non copiees, --mitgcm introuvable:", fsrc)
            continue
        txt = open(fsrc).read()
        for old, new in subs:
            assert old in txt, (src_h, old)
            txt = txt.replace(old, new)
        ecrire(os.path.join(code, dst_h), txt)
    # Bogue de checkpoint69k (obcs_apply_r_star.F) : aux OB N/S, le facteur z* lit OBNeta(j) et
    # OBSeta(j) au lieu de OB[N,S]eta(i). Copie corrigee dans code/ (z* avec fichiers OB*eta).
    fsrc = os.path.join(mitgcm, "pkg/obcs/obcs_apply_r_star.F")
    fdst = os.path.join(code, "obcs_apply_r_star.F")
    if nh:
        if os.path.exists(fdst):
            os.remove(fdst)
    elif os.path.exists(fsrc):
        txt = open(fsrc).read()
        for obn in ("OBNeta", "OBSeta"):
            old = f"     &      + {obn}(  j,bi,bj) / (rSurfS(i,j,bi,bj)-rLowS(i,j,bi,bj))"
            assert txt.count(old) == 1, ("obcs_apply_r_star.F a change", obn)
            txt = txt.replace(old, old.replace(f"{obn}(  j,", f"{obn}(  i,"))
        ecrire(fdst, txt)

    # -------------------------------------------- description pour l'extraction
    desc = dict(nx=nx, ny=ny, dx=dx, dy=dx, x0=x0, y0=y0, dz=[float(v) for v in dz],
                hFacMin=ph["hFacMin"], hFacMinDr=ph["hFacMinDr"], bathy="input/bathy.bin", ob=ob,
                t0=t0, duree=duree, periode=periode, zstar=not nh, nonhydrostatique=nh,
                sponge=spong)
    json.dump(desc, open(os.path.join(out, "enfant.json"), "w"), indent=1)

    print(f"Config {cfg}")
    print(f"grille {nx}x{ny}x{nr} dx={dx:.0f} m, dt={dt:.0f} s, {nstep} pas ({duree / T_M2:g} cycles)")
    print("OB :", {k: len(v) for k, v in ob.items()}, f"; eponge {spong} cellules ; GGL90viscMax={viscmax}")
    print(f"-> {out}")
    return desc
