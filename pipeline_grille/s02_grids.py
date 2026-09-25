"""
s02_grids.py — Grilles horizontales (parent 100 m, enfant 25 m) + grille verticale commune.

  * UTM 19N, bornes arrondies VERS L'INTÉRIEUR à des multiples de 100 m ;
    Nx, Ny multiples de PAD_MULT (tuilage MPI) -> les faces de l'enfant tombent
    exactement sur des faces du parent (rapport 4).
  * bbox lon/lat -> plus grand rectangle UTM inscrit (les méridiens convergent).
  * HMAX auto = profondeur NONNA max dans le parent + 5 m, arrondie à 10 m.
  * Contrôles : alignement des faces, enfant à >= NEST_MARGIN cellules des bords du parent,
    rapport dz(k+1)/dz(k).
Sortie : work/grids.npz
"""
import numpy as np
import config as C
import gridlib as G


def bbox_to_utm_inner(lon0, lon1, lat0, lat1):
    """Rectangle UTM inscrit dans le quadrilatère lon/lat (les méridiens convergent)."""
    xs0 = [G.to_utm(lon0, la)[0] for la in (lat0, lat1)]
    xs1 = [G.to_utm(lon1, la)[0] for la in (lat0, lat1)]
    ys0 = [G.to_utm(lo, lat0)[1] for lo in (lon0, lon1)]
    ys1 = [G.to_utm(lo, lat1)[1] for lo in (lon0, lon1)]
    return max(xs0), min(xs1), max(ys0), min(ys1)


def main():
    C.WORK.mkdir(exist_ok=True)
    grids = {}
    for name, p in C.GRIDS.items():
        P = grids.get("parent")
        align = (P.x0, P.y0, P.dx) if name == "child" else None     # faces sur celles du parent
        box = p["utm"] if "utm" in p else bbox_to_utm_inner(*p["bbox"])   # utm : (x0, x1, y0, y1) en m
        g = G.make_grid(name, *box, p["dx"], align=align)
        grids[name] = g
        print(f"[02] {g}")

    P = grids["parent"]; E = grids.get("child")
    ioff = joff = -1
    if E is not None:
        # --- contrôles d'imbrication
        assert abs(P.dx / E.dx - C.NEST_RATIO) < 1e-9
        for a, b in ((E.x0, P.x0), (E.x1, P.x0), (E.y0, P.y0), (E.y1, P.y0)):
            assert abs(((a - b) / P.dx) - round((a - b) / P.dx)) < 1e-9, "faces non alignées"
        m = C.NEST_MARGIN * P.dx
        marg = [(E.x0 - P.x0) / P.dx, (P.x1 - E.x1) / P.dx, (E.y0 - P.y0) / P.dx, (P.y1 - E.y1) / P.dx]
        ok = min(marg) >= C.NEST_MARGIN
        print(f"[02] marges enfant/parent (cellules parent) O,E,S,N = {marg}  -> {'OK' if ok else 'TROP PRÈS'}")
        assert ok, "enfant trop près du bord du parent"
        ioff = int(round((E.x0 - P.x0) / P.dx)); joff = int(round((E.y0 - P.y0) / P.dx))
        print(f"[02] faces enfant = faces parent i={ioff}..{ioff + E.nx // C.NEST_RATIO}, "
              f"j={joff}..{joff + E.ny // C.NEST_RATIO}")

    hmax = C.HMAX
    if hmax is None:
        lo0, lo1, la0, la1 = C.GRIDS["parent"]["bbox"]
        hm = 0.0
        for t in C.PRODUCTS:
            f = C.WORK / f"{t}_clean.npz"
            if f.exists():
                d = np.load(f)
                ii = (d["lon"] >= lo0) & (d["lon"] <= lo1); jj = (d["lat"] >= la0) & (d["lat"] <= la1)
                hm = max(hm, float(np.nanmax(d["H"][np.ix_(jj, ii)])))
        hmax = float(np.ceil((hm + 5.0) / 10.0) * 10.0)
        print(f"[02] HMAX auto : H max NONNA dans le parent = {hm:.1f} m -> {hmax:.0f} m")
    dz, r = G.vertical_grid(hmax)
    print(f"[02] vertical : Nr={len(dz)}, {C.NSURF}x{C.DZ_SURF} m puis r={r:.4f}, "
          f"dz_max={dz.max():.2f} m, total={dz.sum():.1f} m")

    out = dict(dz=dz, nest_ioff=ioff, nest_joff=joff)
    for name, g in grids.items():
        for k in ("x0", "y0", "dx", "nx", "ny"):
            out[f"{name}_{k}"] = getattr(g, k)
    C.OUT.mkdir(parents=True, exist_ok=True)
    np.savez(C.OUT / "grids.npz", **out)
    print(f"[02] -> {C.OUT/'grids.npz'}")


def load_grids():
    d = np.load(C.OUT / "grids.npz")
    gs = {n: G.Grid(n, float(d[f"{n}_x0"]), float(d[f"{n}_y0"]), float(d[f"{n}_dx"]),
                    int(d[f"{n}_nx"]), int(d[f"{n}_ny"])) for n in C.GRIDS}
    return gs, d["dz"], int(d["nest_ioff"]), int(d["nest_joff"])


if __name__ == "__main__":
    main()
