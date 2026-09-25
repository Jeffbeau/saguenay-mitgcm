"""
s05_forcing.py — Forçage OBCS, conditions initiales et namelists MITgcm du PARENT (Config A).

Approche phénoménologique (on cherche le phénomène, pas une reproduction fidèle) :
  * Marée : M2 pure. Le domaine est « pompé » par ses frontières de l'estuaire (E, S) :
        débit entrant Q(t) = A_surf · dη/dt - Q_riv,  η(t) = a · r(t) · sin(ωt)
    réparti en vitesse barotrope NORMALE uniforme sur les sections mouillées E + S.
    La longueur d'onde de marée (~2000 km) >> domaine : η quasi uniforme, hypothèse raisonnable.
    r(t) : rampe en cosinus sur RAMP_CYCLES cycles (départ au repos).
    Limite assumée : pas de courant de marée longitudinal de l'estuaire (le long du Saint-Laurent).
  * Rivière : Q_RIVER entrant par l'OB ouest (Chicoutimi), eau douce, vitesse uniforme.
  * T/S : profils types (config.profile_fjord / profile_estuary) ; état initial horizontalement
    uniforme dans le fjord et dans l'estuaire, transition vers 69,68°O. OB E/S = profil estuaire.
  * useOBCSbalance = .FALSE. : sinon MITgcm annulerait le flux net… donc la marée.

Conventions des fichiers OBCS (real*8 big-endian) :
  OBEu, OBWu : (Ny, Nr, Nt) ; OBSv : (Nx, Nr, Nt) — indice le plus rapide en premier.
  Enregistrement n (1..Nt) au temps (n - 1/2)·externForcingPeriod (convention MITgcm).
Sortie : <OUT>/parent/run/{input,code}/ + README_run.md + fig_forcing.png
"""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import gridlib as G
from s02_grids import load_grids


def ob_faces(hfac, wet):
    """hFac des faces normales des OB (face entre la cellule OB et la 1re cellule intérieure)."""
    faces = {}
    if wet[:, -1].any():
        faces["E"] = np.minimum(hfac[:, :, -1], hfac[:, :, -2]) * wet[None, :, -1]    # (Nr, Ny)
    if wet[:, 0].any():
        faces["W"] = np.minimum(hfac[:, :, 0], hfac[:, :, 1]) * wet[None, :, 0]
    if wet[0, :].any():
        faces["S"] = np.minimum(hfac[:, 0, :], hfac[:, 1, :]) * wet[None, 0, :]       # (Nr, Nx)
    if wet[-1, :].any():
        faces["N"] = np.minimum(hfac[:, -1, :], hfac[:, -2, :]) * wet[None, -1, :]
    return faces


def ramp(t, tr):
    r = np.where(t <= 0, 0.0, np.where(t < tr, 0.5 * (1 - np.cos(np.pi * t / tr)), 1.0))
    dr = np.where((t > 0) & (t < tr), 0.5 * np.pi / tr * np.sin(np.pi * t / tr), 0.0)
    return r, dr


def mitgcm_root():
    import os
    return Path(os.environ.get("MITGCM_ROOT", Path.home() / "MITgcm"))


def option_file(name, src, subs):
    """templates/<name> s'il existe, sinon copie des options du source MITgcm avec substitutions."""
    t = C.HERE / "templates" / name
    if t.exists():
        return t.read_text()
    txt = (mitgcm_root() / src).read_text()
    for a, b in subs:
        assert a in txt, (src, a)
        txt = txt.replace(a, b)
    return txt


def fmt_list(v, per=5, f="{:.4f}"):
    return G.fortran_list(v, per, f)


def main():
    gs, dz, _, _ = load_grids()
    P = gs["parent"]
    od = C.OUT / "parent"
    g = np.load(od / "grid.npz")
    H, wet = g["depth"], g["wet"]
    nr, ny, nx = len(dz), P.ny, P.nx
    dx = P.dx
    dt = C.DT[dx]
    per = C.T_M2 / C.REC_PER_CYCLE
    assert abs(per / dt - round(per / dt)) < 1e-9, "externForcingPeriod doit être un multiple de deltaT"
    run = od / "run"; inp = run / "input"; code = run / "code"
    for d in (inp, code):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- géométrie des frontières
    _, hfac = G.effective_depth(H, dz)
    hfac *= wet[None]
    faces = ob_faces(hfac, wet)
    area = {e: float((f * dz[:, None]).sum() * dx) for e, f in faces.items()}
    A_surf = float(wet.sum() * dx * dx)
    tide_edges = [e for e in faces if e in ("E", "S", "N")]
    river_edges = [e for e in faces if e == "W"]
    A_tide = sum(area[e] for e in tide_edges)
    # aire du fjord en amont de la coupe ouest (profil « mini ») : elle se remplit / se vide
    # à travers l'OB ouest ; lue sur la grille d'un autre profil (parent entier)
    A_up = 0.0
    if river_edges and C.W_OB == "fjord":
        fup = C.HERE / f"output_{C.UPSTREAM_AREA_FROM}" / "parent" / "grid.npz"
        if fup.exists():
            up = np.load(fup)
            xu = up["xc"]; dxu = float(xu[1] - xu[0])
            A_up = float((up["wet"][:, xu < P.x0]).sum() * dxu * dxu)
            src = f"profil {C.UPSTREAM_AREA_FROM}"
        else:
            A_up = C.UPSTREAM_AREA_DEFAULT; src = "valeur par défaut, output_lite absent"
        print(f"[05] OB ouest = coupe du fjord ; aire amont = {A_up/1e6:.0f} km² ({src})")

    # ---------------- séries temporelles
    # Enregistrement n (1..N) à t = (n - 1/2)·P ; N = durée/P + 1 (suite logique), puis l'état
    # avant le départ (t = -P/2, rivière seule) : pour 0 <= t < P/2, MITgcm interpole entre le
    # dernier enregistrement et le premier. externForcingCycle = nt·P > durée + P/2.
    nrun = C.N_CYCLES * C.REC_PER_CYCLE
    t = np.r_[(np.arange(nrun + 1) + 0.5) * per, -0.5 * per]
    nt = t.size
    w = 2 * np.pi / C.T_M2
    r, dr = ramp(t, C.RAMP_CYCLES * C.T_M2)
    eta = C.TIDE_AMP * r * np.sin(w * t)
    deta = C.TIDE_AMP * (dr * np.sin(w * t) + r * w * np.cos(w * t))
    Q_riv = C.Q_RIVER if river_edges else 0.0
    Q_tide_in = (A_surf + A_up) * deta - Q_riv    # entrant par l'estuaire
    U_tide = Q_tide_in / A_tide                   # m/s, normal, positif = entrant
    # OB ouest : rivière seule, ou coupe du fjord (rivière - remplissage du fjord amont)
    U_riv = (Q_riv - A_up * deta) / area["W"] if river_edges else np.zeros(nt)
    U_riv = np.broadcast_to(U_riv, (nt,)).astype(float)

    # ---------------- profils T/S
    zc = np.cumsum(dz) - dz / 2
    Tf, Sf = C.profile_fjord(zc)
    Te, Se = C.profile_estuary(zc)
    we = 0.5 * (1 + np.tanh((P.lonc - C.PROFILE_BLEND_LON) / C.PROFILE_BLEND_DLON))  # 1 = estuaire
    T3 = Tf[:, None, None] * (1 - we[None]) + Te[:, None, None] * we[None]
    S3 = Sf[:, None, None] * (1 - we[None]) + Se[:, None, None] * we[None]
    G.write_bin(inp / "theta_ini.bin", T3)
    G.write_bin(inp / "salt_ini.bin", S3)
    shutil.copy(od / "bathy.bin", inp / "bathy.bin")

    # ---------------- fichiers OBCS
    files = {}
    for e, f in faces.items():
        n_along = f.shape[1]
        msk = (f > 0).astype(float)                                          # (Nr, n)
        if e in tide_edges:
            sgn = {"E": -1.0, "S": 1.0, "N": -1.0}[e]                         # entrant -> signe
            vel = sgn * U_tide[:, None, None] * msk[None]                     # (Nt, Nr, n)
            Tb, Sb = Te, Se
        else:                                                                 # W : rivière
            vel = U_riv[:, None, None] * msk[None]
            if C.W_OB == "fjord":
                Tb, Sb = Tf, Sf
            else:
                Tb, Sb = np.full(nr, C.T_RIVER), np.full(nr, C.S_RIVER)
        comp = "u" if e in ("E", "W") else "v"
        G.write_bin(inp / f"OB{e}{comp}.bin", vel)
        G.write_bin(inp / f"OB{e}t.bin", np.broadcast_to(Tb[None, :, None], (nt, nr, n_along)))
        G.write_bin(inp / f"OB{e}s.bin", np.broadcast_to(Sb[None, :, None], (nt, nr, n_along)))
        files[e] = comp

    # ---------------- namelists
    ob_txt = (od / "data.obcs_OB.txt").read_text().splitlines()
    ob_lines = [l for l in ob_txt if l.strip().startswith("OB_")]
    file_lines = []
    for e, comp in files.items():
        file_lines += [f" OB{e}{comp}File = 'OB{e}{comp}.bin',",
                       f" OB{e}tFile = 'OB{e}t.bin',", f" OB{e}sFile = 'OB{e}s.bin',"]
    nsp = max(2, int(round(C.SPONGE_M / dx)))
    n_steps = int(round(C.N_CYCLES * C.T_M2 / dt))
    (inp / "data.obcs").write_text(f"""# data.obcs — parent {dx:g} m, généré par s05_forcing.py
 &OBCS_PARM01
{chr(10).join(ob_lines)}
 useOBCSprescribe = .TRUE.,
{chr(10).join(file_lines)}
# NE PAS équilibrer : le flux net aux OB EST la marée (+ rivière)
 useOBCSbalance = .FALSE.,
 useOBCSsponge = .TRUE.,
 &

 &OBCS_PARM02
 &

 &OBCS_PARM03
# éponge {nsp} cellules = {nsp*dx/1e3:.1f} km ; U* : bords E/O, V* : bords N/S
 spongeThickness = {nsp},
 Urelaxobcsbound = 3600.,
 Urelaxobcsinner = 43200.,
 Vrelaxobcsbound = 3600.,
 Vrelaxobcsinner = 43200.,
 &
""")
    parm04 = (od / "data_PARM04.txt").read_text()
    parm04 = parm04[parm04.index("\n &PARM04") + 2:parm04.index("\n &PARM05")]
    parm04 = parm04[:parm04.rindex("&") + 1]
    assert "delR" in parm04 and "delX" in parm04
    (inp / "data").write_text(f"""# data — Saguenay, parent {dx:g} m, Config A (hydrostatique), profil {C.PROFILE}
# Généré par s05_forcing.py — marée M2 {C.TIDE_AMP} m, Q = {C.Q_RIVER:.0f} m3/s
 &PARM01
 tRef = {fmt_list(Tf)}
 sRef = {fmt_list(Sf)}
 eosType = 'JMD95Z',
 rhoConst = 1025.,
 gravity = 9.81,
 f0 = {C.F0:.4e},
 beta = 0.,
# advection / quantité de mouvement
 vectorInvariantMomentum = .TRUE.,
 tempAdvScheme = 33,
 saltAdvScheme = 33,
 staggerTimeStep = .TRUE.,
# viscosité / diffusion (GGL90 calcule le mélange vertical)
 viscC2smag = 2.2,
 viscAr = 1.E-5,
 diffKhT = 0.,
 diffKhS = 0.,
 diffKrT = 1.E-6,
 diffKrS = 1.E-6,
 implicitDiffusion = .TRUE.,
 implicitViscosity = .TRUE.,
# frottement (parois glissantes + frottement de fond quadratique ; à tester en sensibilité)
 no_slip_sides = .FALSE.,
 no_slip_bottom = .FALSE.,
 bottomDragQuadratic = 2.5E-3,
# surface libre non linéaire z*
 implicitFreeSurface = .TRUE.,
 exactConserv = .TRUE.,
 nonlinFreeSurf = 4,
 select_rStar = 2,
 hFacInf = 0.2,
 hFacSup = 2.0,
# partial cells (identiques à la construction de bathy.bin)
 hFacMin = {C.HFAC_MIN},
 hFacMinDr = {C.HFAC_MIN_DR},
# E/S
 readBinaryPrec = 64,
 writeBinaryPrec = 32,
 useSingleCpuIO = .TRUE.,
 &

 &PARM02
 cg2dMaxIters = 1000,
 cg2dTargetResidual = 1.E-9,
 &

 &PARM03
 nIter0 = 0,
 deltaT = {dt:.1f},
 nTimeSteps = {n_steps},
 abEps = 0.1,
 pChkptFreq = {C.T_M2:.1f},
 chkptFreq = 0.,
 dumpFreq = 0.,
 monitorFreq = {per:.1f},
 periodicExternalForcing = .TRUE.,
 externForcingPeriod = {per:.1f},
 externForcingCycle = {nt * per:.1f},
 &

 {parm04.strip()}

 &PARM05
 bathyFile = 'bathy.bin',
 hydrogThetaFile = 'theta_ini.bin',
 hydrogSaltFile = 'salt_ini.bin',
 &
""")
    (inp / "data.pkg").write_text(""" &PACKAGES
 useOBCS = .TRUE.,
 useGGL90 = .TRUE.,
 useDiagnostics = .TRUE.,
 &
""")
    (inp / "data.ggl90").write_text("""# GGL90 : valeurs par défaut, longueur de mélange limitée (Blanke & Delecluse)
 &GGL90_PARM01
 mxlMaxFlag = 2,
 &
""")
    # Sorties = sagdiag/data.diagnostics.mini_recommande : fréquences multiples de dt qui divisent
    # T_M2 (930 et 360 s), instantanés sans décalage (timePhase = 0). state3D (avec WVEL) et eta2D
    # servent aussi à forcer l'enfant (imbrication/extraire_obcs.py).
    for f in (930.0, 360.0):
        assert abs(f / dt - round(f / dt)) < 1e-9 and abs(C.T_M2 / f - round(C.T_M2 / f)) < 1e-9
    lv = [int(np.argmin(np.abs(zc - z))) + 1 for z in (1.0, 10.0, 30.0)]
    (inp / "data.diagnostics").write_text(f"""# data.diagnostics — généré par s05_forcing.py (= sagdiag/data.diagnostics.mini_recommande)
 &DIAGNOSTICS_LIST
 dumpAtLast = .FALSE.,
# 1. eta instantané, 124 par cycle
 fields(1,1) = 'ETAN    ',
   fileName(1) = 'eta2D',
   frequency(1) = -360.,
   timePhase(1) = 0.,
# 2. niveaux ~1, 10, 30 m, 124 par cycle (tourbillons)
 fields(1:3,2) = 'UVEL    ','VVEL    ','momVort3',
   levels(1:3,2) = {lv[0]}., {lv[1]}., {lv[2]}.,
   fileName(2) = 'lev2D',
   frequency(2) = -360.,
   timePhase(2) = 0.,
# 3. état 3D instantané, 48 par cycle (moyenne de phase, énergétique, forçage de l'enfant)
 fields(1:6,3) = 'UVEL    ','VVEL    ','WVEL    ','THETA   ','SALT    ','RHOAnoma',
   fileName(3) = 'state3D',
   frequency(3) = -930.,
   timePhase(3) = 0.,
# 4. moyennes par cycle
 fields(1:8,4) = 'UVEL    ','VVEL    ','WVEL    ','THETA   ','SALT    ',
                 'UVELSQ  ','VVELSQ  ','UV_VEL_C',
   fileName(4) = 'mean3D',
   frequency(4) = {C.T_M2:.1f},
# 5. transports moyens par cycle (flux de sel aux seuils)
 fields(1:6,5) = 'UVELMASS','VVELMASS','WVELMASS','USLTMASS','VSLTMASS','WSLTMASS',
   fileName(5) = 'flux3D',
   frequency(5) = {C.T_M2:.1f},
# 6. mélange vertical, 48 par cycle
 fields(1:2,6) = 'GGL90TKE','GGL90Kr ',
   fileName(6) = 'mix3D',
   frequency(6) = -930.,
   timePhase(6) = 0.,
 &
 &DIAG_STATIS_PARMS
 &
""")
    (inp / "eedata").write_text(""" &EEPARMS
 &
 &EESUPPORT
 nTx = 1,
 nTy = 1,
 &
""")
    # ---------------- code/
    shutil.copy(od / "SIZE.h", code / "SIZE.h")
    (code / "CPP_OPTIONS.h").write_text(option_file(
        "CPP_OPTIONS.h", "model/inc/CPP_OPTIONS.h", [("#undef NONLIN_FRSURF", "#define NONLIN_FRSURF")]))
    (code / "OBCS_OPTIONS.h").write_text(option_file(
        "OBCS_OPTIONS.h", "pkg/obcs/OBCS_OPTIONS.h", [("#undef ALLOW_OBCS_SPONGE", "#define ALLOW_OBCS_SPONGE")]))
    (code / "packages.conf").write_text("gfd\nobcs\nggl90\ndiagnostics\n")
    (code / "DIAGNOSTICS_SIZE.h").write_text(f"""C     DIAGNOSTICS_SIZE.h — généré par s05_forcing.py
      INTEGER    ndiagMax
      INTEGER    numlists, numperlist, numLevels
      INTEGER    numDiags
      INTEGER    nRegions, sizRegMsk, nStats
      INTEGER    diagSt_size
      PARAMETER( ndiagMax = 500 )
      PARAMETER( numlists = 10, numperlist = 50, numLevels=2*Nr )
      PARAMETER( numDiags = 26*Nr )
      PARAMETER( nRegions = 0 , sizRegMsk = 1 , nStats = 4 )
      PARAMETER( diagSt_size = 10*Nr )
""")

    # ---------------- rapport + figure
    info = dict(profil=C.PROFILE, dx=dx, Nx=nx, Ny=ny, Nr=nr, deltaT=dt, nTimeSteps=n_steps,
                duree_h=n_steps * dt / 3600, T_M2_modele=C.T_M2, externForcingPeriod=per,
                n_records=nt, A_surface_km2=A_surf / 1e6,
                sections_OB_m2={e: round(a) for e, a in area.items()},
                Q_maree_ampl_m3s=float(A_surf * C.TIDE_AMP * w), U_OB_estuaire_max_ms=float(np.abs(U_tide).max()),
                U_OB_ouest_max_ms=float(np.abs(U_riv).max()), A_amont_km2=A_up / 1e6, eponge_cellules=nsp,
                volume_3D_Mo_par_champ=nx * ny * nr * 4 / 1e6)
    (run / "forcing_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(Tf, -zc, label="T fjord"); ax[0].plot(Te, -zc, "--", label="T estuaire")
    ax[0].set_xlabel("°C"); ax[0].set_ylabel("z (m)"); ax[0].legend()
    ax[1].plot(Sf, -zc, label="S fjord"); ax[1].plot(Se, -zc, "--", label="S estuaire")
    ax[1].set_xlabel("S"); ax[1].legend()
    ax[2].plot(t[:-1] / 3600, eta[:-1], label="η visé (m)")
    ax[2].plot(t[:-1] / 3600, U_tide[:-1] * 10, label="U normal OB estuaire ×10 (m/s)")
    if river_edges:
        ax[2].plot(t[:-1] / 3600, U_riv[:-1], label="U OB ouest (m/s)")
    ax[2].axvline(C.N_CYCLES * C.T_M2 / 3600, color="k", lw=.8, ls=":")
    ax[2].set_xlabel("t (h)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run / "fig_forcing.png", dpi=100); plt.close(fig)
    readme(run, info)
    print("[05] " + json.dumps(info, ensure_ascii=False))


def readme(run, info):
    (run / "README_run.md").write_text(f"""# Run parent {info['dx']:g} m — Config A (profil {info['profil']})

Généré par `s05_forcing.py`. Marée M2 {C.TIDE_AMP} m (pompage par l'estuaire), rivière {C.Q_RIVER:.0f} m³/s,
profils T/S types. {info['Nx']}×{info['Ny']}×{info['Nr']}, dt = {info['deltaT']:g} s,
{info['nTimeSteps']} pas = {info['duree_h']:.1f} h ({C.N_CYCLES} cycles dont {C.RAMP_CYCLES} de rampe).

## Avant tout
`theta_ini.bin.gz` et `salt_ini.bin.gz` sont compressés (limite de taille du transfert) :
`gunzip input/*.gz`.

## Compiler (MITgcm checkpoint69k)
```bash
mkdir build && cd build
$MITGCM/tools/genmake2 -mods ../code -mpi -optfile $MITGCM/tools/build_options/linux_amd64_gfortran
make depend && make -j
```
Décomposition : éditer `code/SIZE.h` (sNx = Nx/nPx, sNy = Ny/nPy) puis recompiler.
SIZE.h fourni : nPx = {C.NPX if info['Nx'] % C.NPX == 0 else 1} -> `mpirun -np {C.NPX if info['Nx'] % C.NPX == 0 else 1}`.

## Lancer
```bash
cd ../input && ln -sf ../build/mitgcmuv .
mpirun -np <nPx*nPy> ./mitgcmuv > output.txt
```

## Contrôles au premier run
- `grep advcfl output.txt` : CFL horizontal (< 0,5) et vertical `advcfl_W_hf_max` (< 1).
  Si W dépasse, baisser deltaT (diviseur de {info['externForcingPeriod']:.0f} s : 15, 12, 10…).
- `%MON dynstat_eta_max/min` : doit osciller à ± {C.TIDE_AMP} m après la rampe.
- Sorties : `eta2D.*`, `lev2D.*` (124/cycle), `state3D.*`, `mix3D.*` (48/cycle,
  {info['volume_3D_Mo_par_champ']:.0f} Mo par champ 3D), `mean3D.*`, `flux3D.*` (moyennes par cycle).
""")


if __name__ == "__main__":
    main()
