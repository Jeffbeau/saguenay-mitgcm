"""
s03_bathy.py — Bathymétrie modèle (parent puis enfant) + corrections + fichiers MITgcm.

Pour chaque grille :
  1. NONNA-10 : agrégation des pixels par cellule (moyenne de H sur la partie EAU, fraction
     mouillée, couverture). NONNA-100 : échantillonnage bilinéaire sub x sub par cellule.
     Chaque cellule prend NONNA-10 s'il la couvre à >= N10_FULL_COVER, sinon NONNA-100.
  2. Masque : mouillée si fraction d'eau >= WET_FRAC_MIN, H > H_DRY et cellule couverte.
  3. H = max(H, H_MIN) sur les cellules mouillées.
  4. Bords fermés imposés (CLOSED_EDGES).
  5. Nettoyage itératif : cellules mouillées avec >= MAX_LAND_NEIGH voisins terre (culs-de-sac
     d'une cellule, cellules isolées) -> terre.
  6. Connexité (4-conn) : on ne garde que l'eau reliée à une frontière ouverte (lacs retirés).
  7. [enfant] cohérence avec le parent le long des frontières ouvertes :
       NEST_COPY cellules parent : masque + H de l'enfant = parent (flux identiques) ;
       NEST_BLEND cellules suivantes : H mélangé linéairement parent -> enfant.
  8. Arrondi hFac identique à MITgcm -> profondeur effective écrite dans bathy.bin.
Sorties : output/<grille>/bathy.bin, SIZE.h, data_PARM04.txt, data.obcs_OB.txt, grid.npz
"""
import json
import numpy as np
from scipy import ndimage as ndi

import config as C
import gridlib as G
from s02_grids import load_grids

EDGES = ("W", "E", "S", "N")


def edge_slices(name):
    return {"W": (slice(None), 0), "E": (slice(None), -1),
            "S": (0, slice(None)), "N": (-1, slice(None))}[name]


def aggregate(g, prods, log):
    out = None
    if "n100" in prods:
        d = prods["n100"]
        Hm, wf, cf = G.sample_to_grid(g, d["H"].astype(float), d["water"], d["covered"],
                                      d["lon"], d["lat"])
        src = np.where(cf > 0, 100, 0)
        out = [Hm, wf, cf, src]
    if "n10" in prods:
        d = prods["n10"]
        Ha, wa, ca = G.bin_to_grid(g, d["H"].astype(float), d["water"], d["covered"],
                                   d["lon"], d["lat"])
        if out is None:
            out = [Ha, wa, ca, np.where(ca > 0, 10, 0)]
        else:
            use = ca >= C.N10_FULL_COVER
            # écart n10 / n100 là où les deux existent (contrôle de raccord)
            both = use & (wa >= .5) & (out[1] >= .5)
            if both.any():
                dH = Ha[both] - out[0][both]
                log["n10_minus_n100_mean_m"] = float(np.mean(dH))
                log["n10_minus_n100_rms_m"] = float(np.sqrt(np.mean(dH ** 2)))
                log["mask_agreement_n10_n100"] = float(np.mean(((wa >= .5) == (out[1] >= .5))[use]))
            for k, v in enumerate((Ha, wa, ca)):
                out[k] = np.where(use, v, out[k])
            out[3] = np.where(use, 10, out[3])
            log["cells_from_n10"] = int((use & (out[1] > 0)).sum())
    return out


def clean_mask(wet, H, closed, log):
    wet = wet.copy()
    for e in closed:
        wet[edge_slices(e)] = False
    # segments de frontière ouverte trop courts -> fermés (évite des OB parasites de 1-3 cellules)
    n_seg = 0
    for e in EDGES:
        line = wet[edge_slices(e)]
        lab, n = ndi.label(line)
        for k in range(1, n + 1):
            if (lab == k).sum() < C.OB_MIN_SEG:
                line[lab == k] = False; n_seg += 1
        wet[edge_slices(e)] = line
    log["short_ob_segments_closed"] = log.get("short_ob_segments_closed", 0) + n_seg
    n_tot = 0
    while True:
        bad = wet & (G.land_neighbours(wet) >= C.MAX_LAND_NEIGH)
        if not bad.any():
            break
        wet[bad] = False; n_tot += int(bad.sum())
    log["removed_deadend_cells"] = n_tot
    seeds = np.zeros_like(wet)
    for e in EDGES:
        if e not in closed:
            seeds[edge_slices(e)] = True
    wet, n_lake = G.keep_connected(wet, seeds)
    log["removed_disconnected_cells"] = n_lake
    return wet


def extrude_ob(wet, H, closed, log):
    """Bande de OB_EXTRUDE cellules le long des bords ouverts : masque et profondeur recopiés
    depuis la ligne intérieure (fond « droit » perpendiculaire à l'OB), H >= H_OB_MIN.
    Évite les poches et hauts-fonds collés aux OB, où la vitesse imposée empile l'eau."""
    n = C.OB_EXTRUDE
    if n <= 0:
        return wet, H
    wet, H = wet.copy(), H.copy()
    band = np.zeros(wet.shape, bool)
    for e in EDGES:
        if e in closed or not wet[edge_slices(e)].any():
            continue
        if e == "W":
            wet[:, :n] = wet[:, n:n + 1]; H[:, :n] = H[:, n:n + 1]; band[:, :n] = True
        elif e == "E":
            wet[:, -n:] = wet[:, -n - 1:-n]; H[:, -n:] = H[:, -n - 1:-n]; band[:, -n:] = True
        elif e == "S":
            wet[:n, :] = wet[n:n + 1, :]; H[:n, :] = H[n:n + 1, :]; band[:n, :] = True
        else:
            wet[-n:, :] = wet[-n - 1:-n, :]; H[-n:, :] = H[-n - 1:-n, :]; band[-n:, :] = True
    deep = band & wet & (H < C.H_OB_MIN)
    H = np.where(deep, C.H_OB_MIN, H)
    log["ob_band_cells"] = int((band & wet).sum()); log["ob_band_deepened"] = int(deep.sum())
    return wet, H


def open_boundaries(wet, closed):
    ob = {}
    for e in EDGES:
        if e in closed:
            continue
        line = wet[edge_slices(e)]
        if line.any():
            ob[e] = line.copy()
    return ob


def rle_fortran(vals):
    out, i = [], 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        out.append(f"{j - i}*{vals[i]}" if j - i > 1 else f"{vals[i]}")
        i = j
    s = ", ".join(out)
    return s


def write_obcs_snippet(path, g, ob):
    lines = [f"# Frontières ouvertes détectées pour {g.name} ({g.nx}x{g.ny})",
             "# à copier dans data.obcs / &OBCS_PARM01", ""]
    key = {"E": ("OB_Ieast", -1), "W": ("OB_Iwest", 1), "N": ("OB_Jnorth", -1), "S": ("OB_Jsouth", 1)}
    for e, line in ob.items():
        nm, v = key[e]
        vals = [v if w else 0 for w in line]
        lines.append(f" {nm} = {rle_fortran(vals)},")
        idx = np.nonzero(line)[0]
        lines.append(f"#   {e}: {line.sum()} points mouillés, indices {idx.min()+1}..{idx.max()+1}")
    path.write_text("\n".join(lines) + "\n")


def write_size_h(path, g, nr):
    npx = C.NPX if g.nx % C.NPX == 0 else 1
    txt = f"""C     SIZE.h — généré par s03_bathy.py pour la grille {g.name} ({g.dx:g} m)
C     Décomposition MPI : nPx = {npx} (lancer avec mpirun -np {npx}).
C     Nx = {g.nx} = sNx*nSx*nPx ; Ny = {g.ny} = sNy*nSy*nPy (multiples de {C.PAD_MULT})
      INTEGER sNx
      INTEGER sNy
      INTEGER OLx
      INTEGER OLy
      INTEGER nSx
      INTEGER nSy
      INTEGER nPx
      INTEGER nPy
      INTEGER Nx
      INTEGER Ny
      INTEGER Nr
      PARAMETER (
     &           sNx = {g.nx // npx:4d},
     &           sNy = {g.ny:4d},
     &           OLx =   4,
     &           OLy =   4,
     &           nSx =   1,
     &           nSy =   1,
     &           nPx = {npx:3d},
     &           nPy =   1,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = {nr:4d})

      INTEGER MAX_OLX
      INTEGER MAX_OLY
      PARAMETER ( MAX_OLX = OLx,
     &            MAX_OLY = OLy )
"""
    path.write_text(txt)


def write_parm04(path, g, dz):
    txt = f"""# Extrait data / &PARM04 et &PARM05 pour {g.name}
 &PARM04
 usingCartesianGrid = .TRUE.,
 delX = {g.nx}*{g.dx:.1f},
 delY = {g.ny}*{g.dx:.1f},
 delR = {G.fortran_list(dz)}
 &

 &PARM05
 bathyFile = 'bathy.bin',
 &
# hFacMin = {C.HFAC_MIN}, hFacMinDr = {C.HFAC_MIN_DR}  (doivent être identiques dans &PARM01,
#   la profondeur de bathy.bin est déjà arrondie de la même façon)
# f0 = {C.F0:.4e}
# Origine UTM 19N de la grille : x0 = {g.x0:.0f} m, y0 = {g.y0:.0f} m
"""
    path.write_text(txt)


def process(g, prods, dz, parent=None, nest=None):
    log = {"grid": repr(g)}
    Hm, wf, cf, src = aggregate(g, prods, log)
    covered = cf >= 0.5
    wet0 = (wf >= C.WET_FRAC_MIN) & (np.nan_to_num(Hm) > C.H_DRY) & covered
    log["cells_wet_raw"] = int(wet0.sum())
    H = np.where(wet0, np.maximum(np.nan_to_num(Hm), C.H_MIN), 0.0)
    log["cells_deepened_to_Hmin"] = int((wet0 & (np.nan_to_num(Hm) < C.H_MIN)).sum())
    closed = C.CLOSED_EDGES.get(g.name, [])
    wet = clean_mask(wet0, H, closed, log)

    if parent is not None:
        Pwet, PHeff, ioff, joff = parent
        r = C.NEST_RATIO
        jc, ic = np.mgrid[0:g.ny, 0:g.nx]
        jp, ip = joff + jc // r, ioff + ic // r
        Hp, Wp = PHeff[jp, ip], Pwet[jp, ip]
        obs = [e for e in EDGES if e not in closed and wet[edge_slices(e)].any()]
        # distance au bord ouvert le plus proche, en cellules parent
        dist = np.full(wet.shape, 1e9)
        for e in obs:
            dd = {"W": ic // r, "E": (g.nx - 1 - ic) // r,
                  "S": jc // r, "N": (g.ny - 1 - jc) // r}[e]
            dist = np.minimum(dist, dd)
        copy = dist < C.NEST_COPY
        blend = (dist >= C.NEST_COPY) & (dist < C.NEST_COPY + C.NEST_BLEND)
        w = np.clip(1.0 - (dist - C.NEST_COPY + 0.5) / C.NEST_BLEND, 0, 1)
        wet = np.where(copy, Wp, wet)
        H = np.where(copy, Hp, H)
        both = blend & wet & Wp
        H = np.where(both, w * Hp + (1 - w) * H, H)
        log["nest_copy_cells"] = int(copy.sum()); log["nest_blend_cells"] = int(blend.sum())
        wet = clean_mask(wet, H, closed, log)
        wet = np.where(copy, Wp, wet)      # la bande copiée reste strictement celle du parent
        # petites zones d'eau isolées dans l'enfant (morceaux du masque parent aux OB)
        lab, n = ndi.label(wet, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        if n > 1:
            sz = ndi.sum(wet, lab, range(1, n + 1))
            small = np.isin(lab, np.nonzero(sz < C.MIN_COMPONENT)[0] + 1)
            wet &= ~small
            log["child_small_components_removed_cells"] = int(small.sum())

    if parent is None:                     # l'enfant a déjà la bande copiée du parent
        wet, H = extrude_ob(wet, H, closed, log)
        seeds = np.zeros_like(wet)
        for e in EDGES:
            if e not in closed:
                seeds[edge_slices(e)] = True
        wet, _ = G.keep_connected(wet, seeds)
    # cellule de terre isolée (4 voisins mouillés) = artefact de raccord -> eau
    lonely = ~wet & (G.land_neighbours(wet) == 0)
    if lonely.any():
        k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        Hn = ndi.convolve(np.where(wet, H, 0.0), k, mode="constant") / 4.0
        H = np.where(lonely, Hn, H); wet = wet | lonely
    log["lonely_land_cells_filled"] = int(lonely.sum())
    H = np.where(wet, np.maximum(H, C.H_MIN), 0.0)
    if C.APPLY_HFAC_ROUNDING:
        Heff, hfac = G.effective_depth(H, dz)
    else:
        Heff, hfac = H, None
    Heff = np.where(wet, Heff, 0.0)
    log["max_depth"] = float(Heff.max())
    log["cells_wet_final"] = int(wet.sum())
    log["wet_fraction_domain"] = float(wet.mean())
    log["hfac_rounding_max_change_m"] = float(np.abs(Heff - H)[wet].max())
    # murs artificiels : eau adjacente à une zone non couverte par NONNA
    nocov = ~covered
    near_nocov = ndi.binary_dilation(nocov, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]) & wet
    log["wet_cells_touching_nodata"] = int(near_nocov.sum())
    ob = open_boundaries(wet, closed)
    log["open_boundaries"] = {e: int(v.sum()) for e, v in ob.items()}
    log["closed_edges_forced"] = closed
    return dict(Hm=Hm, wf=wf, cf=cf, wet=wet, H=Heff, ob=ob, near_nocov=near_nocov, src=src), log


def save(g, res, dz, log):
    od = C.OUT / g.name
    od.mkdir(parents=True, exist_ok=True)
    G.write_bin(od / "bathy.bin", -res["H"])
    write_size_h(od / "SIZE.h", g, len(dz))
    write_parm04(od / "data_PARM04.txt", g, dz)
    write_obcs_snippet(od / "data.obcs_OB.txt", g, res["ob"])
    np.savez_compressed(od / "grid.npz", xc=g.xc, yc=g.yc, xg=g.xg, yg=g.yg,
                        lonc=g.lonc, latc=g.latc, depth=res["H"], wet=res["wet"],
                        wetfrac=res["wf"], coverfrac=res["cf"], Hraw=res["Hm"], dz=dz,
                        near_nodata=res["near_nocov"], source=res["src"])
    (od / "log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"[03] {g.name}: " + json.dumps(log, ensure_ascii=False))


def main():
    prods = {t: np.load(C.WORK / f"{t}_clean.npz") for t in C.PRODUCTS
             if (C.WORK / f"{t}_clean.npz").exists()}
    gs, dz, ioff, joff = load_grids()
    P, E = gs["parent"], gs.get("child")
    resP, logP = process(P, prods, dz)
    save(P, resP, dz, logP)
    if E is not None:
        resE, logE = process(E, prods, dz, parent=(resP["wet"], resP["H"], ioff, joff))
        save(E, resE, dz, logE)


if __name__ == "__main__":
    main()
