"""
s04_verify.py — Contrôles et figures.

  * Relecture de bathy.bin (big-endian real*8, Nx x Ny), terre = 0, H >= H_MIN.
  * Seuils : profondeur de col (« saddle ») entre bassins voisins, sur 10 m / enfant / parent.
    col(A,B) = plus grande profondeur h telle que A et B restent reliés par de l'eau > h.
  * Imbrication : bande copiée identique au parent ; section mouillée parent vs enfant
    le long de chaque frontière ouverte de l'enfant.
  * Figures : output/fig_*.png ; rapport : output/verification.md
"""
import json
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import gridlib as G
from s02_grids import load_grids

ST4 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
# Points « intérieur de bassin » (lon, lat) : on prend la cellule la plus profonde à < 600 m
BASINS = {
    "estuaire": (-69.630, 48.150),
    "bassin ext.": (-69.775, 48.143),  # rayon 1,5 km
    "bassin méd.": (-69.935, 48.232),
    "amont (Éternité)": (-70.050, 48.256)
}
SILL_BBOX = (-70.060, -69.550, 48.080, 48.280)   # zone des seuils (analyse 10 m)
PAIRS = [("estuaire", "bassin ext."), ("bassin ext.", "bassin méd."),
         ("bassin méd.", "amont (Éternité)")]


def seed_index(H, X, Y, lon, lat, r=1500.0):
    x, y = G.to_utm(lon, lat)
    dd = (X - x) ** 2 + (Y - y) ** 2
    cand = np.where(dd < r * r, H, -1)
    if np.nanmax(cand) <= 0:
        return None
    k = np.nanargmax(cand)
    return np.unravel_index(k, H.shape)


def saddle(H, a, b, step=0.5):
    if a is None or b is None:
        return np.nan, None
    lo, hi = 0.0, min(H[a], H[b])
    lab, _ = ndi.label(H > lo, ST4)
    if lab[a] == 0 or lab[a] != lab[b]:
        return np.nan, None
    while hi - lo > step:
        mid = 0.5 * (lo + hi)
        lab, _ = ndi.label(H > mid, ST4)
        (lo, hi) = (mid, hi) if (lab[a] and lab[a] == lab[b]) else (lo, mid)
    # position : cellules > lo qui touchent les deux composantes séparées à hi
    lab_hi, _ = ndi.label(H > hi, ST4)
    A, B = lab_hi == lab_hi[a], lab_hi == lab_hi[b]
    loc = np.empty((0, 2))
    for _ in range(200):                    # on fait croître les deux composantes dans l'eau > lo
        A = ndi.binary_dilation(A, ST4) & (H > lo)
        B = ndi.binary_dilation(B, ST4) & (H > lo)
        loc = np.argwhere(A & B)
        if len(loc):
            break
    return lo, (loc.mean(0) if len(loc) else None)


def grid_xy(g):
    return np.meshgrid(g.xc, g.yc)


def sills_for(H, X, Y):
    seeds = {k: seed_index(H, X, Y, *v) for k, v in BASINS.items()}
    out = []
    for a, b in PAIRS:
        s, loc = saddle(H, seeds[a], seeds[b])
        if loc is not None:
            j, i = loc
            xs = np.interp(i, np.arange(X.shape[1]), X[0]); ys = np.interp(j, np.arange(Y.shape[0]), Y[:, 0])
            lon, lat = G.to_geo(xs, ys)
        else:
            lon = lat = np.nan
        out.append(dict(pair=f"{a} / {b}", depth=float(s), lon=float(lon), lat=float(lat)))
    return out


def main():
    gs, dz, ioff, joff = load_grids()
    P, E = gs["parent"], gs.get("child")
    GR = [g for g in (P, E) if g is not None]
    pkey = f"parent {P.dx:g} m"
    rep = [f"# Vérification grille + bathy Saguenay (profil {C.PROFILE})", ""]
    res = {}
    for g in GR:
        od = C.OUT / g.name
        H = -G.read_bin(od / "bathy.bin", (g.ny, g.nx))
        npz = np.load(od / "grid.npz")
        assert np.allclose(H, npz["depth"]), "relecture bathy.bin != profondeur calculée"
        assert (od / "bathy.bin").stat().st_size == g.nx * g.ny * 8
        wet = H > 0
        assert (H[wet] >= C.H_MIN - 1e-6).all()
        res[g.name] = (H, npz)
        rep += [f"## {g.name} — {g!r}",
                f"- bathy.bin relu : {g.nx}x{g.ny} real*8 big-endian, OK ; "
                f"cellules mouillées {wet.sum()} ({100*wet.mean():.1f} %), "
                f"H ∈ [{H[wet].min():.2f}, {H.max():.1f}] m",
                f"- log : `{(od/'log.json').read_text().strip()}`", ""]

    # --- seuils
    d10 = np.load(C.WORK / "n10_clean.npz")
    lo0, lo1, la0, la1 = SILL_BBOX
    ii = (d10["lon"] >= lo0) & (d10["lon"] <= lo1); jj = (d10["lat"] >= la0) & (d10["lat"] <= la1)
    H10 = np.nan_to_num(d10["H"][np.ix_(jj, ii)].astype(np.float32), nan=-1.0)
    LON, LAT = np.meshgrid(d10["lon"][ii], d10["lat"][jj])
    X10, Y10 = G.to_utm(LON, LAT)
    table = {"NONNA 10 m": sills_for(H10, X10, Y10)}
    for g in GR[::-1]:
        X, Y = grid_xy(g)
        table[f"{g.name} {g.dx:g} m"] = sills_for(res[g.name][0], X, Y)
    rep += ["## Seuils (profondeur de col sous NMM, m)", "",
            "| col | " + " | ".join(table) + " |", "|---|" + "---|" * len(table)]
    for k, (a, b) in enumerate(PAIRS):
        row = [f"{table[t][k]['depth']:.1f} ({table[t][k]['lon']:.3f}°, {table[t][k]['lat']:.3f}°)"
               for t in table]
        rep.append(f"| {a} / {b} | " + " | ".join(row) + " |")
    # distance le long du chenal (enfant 25 m) depuis le seuil d'entrée
    from skimage.graph import MCP_Geometric
    GD = E if E is not None else P
    HEc = res[GD.name][0]
    def ij(lo, la):
        x, y = G.to_utm(lo, la)
        j, i = int((y - GD.y0) // GD.dx), int((x - GD.x0) // GD.dx)
        jj, ii = np.nonzero(HEc > 0); k = np.argmin((jj - j) ** 2 + (ii - i) ** 2)
        return jj[k], ii[k]
    tkey = f"{GD.name} {GD.dx:g} m"
    s0 = table[tkey][0]
    Dk, _ = MCP_Geometric(np.where(HEc > 0, 1.0, np.inf)).find_costs([ij(s0["lon"], s0["lat"])])
    Dk *= GD.dx / 1e3
    dist = [Dk[ij(s["lon"], s["lat"])] for s in table[tkey] if np.isfinite(s["lon"])]
    rep += ["", f"Distance le long du chenal depuis le seuil d'entrée ({tkey}) : "
            + ", ".join(f"{dd:.1f} km" for dd in dist)]
    HPx = res["parent"][0]; lonP = res["parent"][1]["lonc"]
    if (lonP < -70.2).any():
        rep += ["", f"Bassin intérieur (O de 70,2°O, parent) : H max = {HPx[lonP < -70.2].max():.1f} m"]
    rep += ["", f"Référence (Belzile et al. 2016) : {C.EXPECTED_SILLS} (km, m) ; bassin {C.EXPECTED_BASIN} m.",
            f"Z0 (NMM-ZC) utilisé : {C.Z0_STATIONS} (Tadoussac 03425, Port-Alfred).", ""]
    bias = C.WORK / "n100_bias.json"
    if bias.exists():
        rep += ["Correction NONNA-100 (biais haut-fond) : `" + bias.read_text().replace(chr(10), " ") + "`", ""]

    HP = res["parent"][0]
    if E is not None:
        HE = res["child"][0]
        rep += nesting_report(P, E, HP, HE, ioff, joff)
    return figures(P, E, HP, res, table, dz, X10, Y10, H10, rep)


def nesting_report(P, E, HP, HE, ioff, joff):
    rep = []
    r = C.NEST_RATIO
    jc, ic = np.mgrid[0:E.ny, 0:E.nx]
    HPc = HP[joff + jc // r, ioff + ic // r]
    rep.append("## Imbrication parent -> enfant")
    for e, sl, slP in (("W", (slice(None), 0), None), ("E", (slice(None), -1), None),
                       ("S", (0, slice(None)), None), ("N", (-1, slice(None)), None)):
        le, lp = HE[sl], HPc[sl]
        if (le > 0).any():
            ae, ap = le.sum() * E.dx, lp.sum() * E.dx
            same = np.allclose(le, lp)
            rep.append(f"- OB {e} : section enfant {ae/1e3:.2f}e3 m², parent {ap/1e3:.2f}e3 m², "
                       f"ratio {ae/ap:.4f}, profil identique : {same}")
    ob_edges = json.loads((C.OUT / "child" / "log.json").read_text())["open_boundaries"]
    rep.append(f"- OB enfant : {ob_edges}")
    rep.append("")
    return rep


def figures(P, E, HP, res, table, dz, X10, Y10, H10, rep):
    pkey = f"parent {P.dx:g} m"

    # --- figures
    fig, ax = plt.subplots(figsize=(15, 7.5))
    Hp = np.where(HP > 0, HP, np.nan)
    pc = ax.pcolormesh(P.xg / 1e3, P.yg / 1e3, Hp, cmap="viridis_r", vmin=0, vmax=300)
    src = res["parent"][1]["source"]
    ax.contour(P.xc / 1e3, P.yc / 1e3, ((src == 10) & (HP > 0)).astype(float), [0.5],
               colors="orange", linewidths=.7)
    if E is not None:
        ax.plot(np.r_[E.x0, E.x1, E.x1, E.x0, E.x0] / 1e3, np.r_[E.y0, E.y0, E.y1, E.y1, E.y0] / 1e3,
                "r-", lw=1.5, label="enfant 25 m")
    for t, rows in table.items():
        if t.startswith("parent"):
            for s in rows:
                if not np.isfinite(s["lon"]):
                    continue
                x, y = G.to_utm(s["lon"], s["lat"])
                ax.plot(x / 1e3, y / 1e3, "wv", mec="k", ms=9)
                ax.annotate(f"{s['depth']:.0f} m", (x / 1e3, y / 1e3), xytext=(4, 6),
                            textcoords="offset points", fontsize=9, color="k",
                            bbox=dict(fc="w", alpha=.7, lw=0))
    ax.set_aspect(1); ax.set_xlabel("x UTM19N (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"Parent {P.dx:g} m ({P.nx}x{P.ny}) — profondeur sous NMM ; ▼ = cols ; "
                 f"orange = limite NONNA-10 (ailleurs NONNA-100 corrigé)")
    plt.colorbar(pc, ax=ax, label="m", shrink=.8)
    if E is not None:
        ax.legend(loc="upper right")
    fig.savefig(C.OUT / "fig_parent.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    cols = [("NONNA 10 m", X10, Y10, np.where(H10 > 0, H10, np.nan))]
    if E is not None:
        cols += child_figure(E, res)
    cols += [(pkey, *grid_xy(P), np.where(HP > 0, HP, np.nan))]
    return sills_and_vertical(cols, table, dz, pkey, rep)


def child_figure(E, res):
    HE = res["child"][0]
    band = C.NEST_COPY * C.NEST_RATIO
    strip = np.zeros(HE.shape, bool)
    strip[:band] = strip[-band:] = True; strip[:, :band] = True; strip[:, -band:] = True
    fig, ax = plt.subplots(figsize=(15, 9.5))
    He = np.where(HE > 0, HE, np.nan)
    pc = ax.pcolormesh(E.xg / 1e3, E.yg / 1e3, He, cmap="viridis_r", vmin=0, vmax=260)
    ax.contour(E.xc / 1e3, E.yc / 1e3, strip & (HE > 0), [0.5], colors="r", linewidths=.6)
    ax.set_aspect(1); ax.set_title(f"Enfant {E.dx:g} m ({E.nx}x{E.ny}) ; rouge = bande copiée du parent")
    plt.colorbar(pc, ax=ax, label="m", shrink=.8)
    fig.savefig(C.OUT / "fig_child.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return [("child 25 m", *grid_xy(E), He)]


def sills_and_vertical(cols, table, dz, pkey, rep):
    # zooms sur les cols : 10 m / enfant / parent
    fig, axs = plt.subplots(3, len(cols), figsize=(5 * len(cols), 13), squeeze=False)
    for k, s in enumerate(table["NONNA 10 m"][:3]):
        x, y = G.to_utm(s["lon"], s["lat"]); w = 2500
        for col, (lab, XX, YY, HH) in enumerate(cols):
            a = axs[k, col]
            m = (np.abs(XX - x) < w) & (np.abs(YY - y) < w)
            jj, ii = np.nonzero(m)
            sl = (slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1))
            pcm = a.pcolormesh(XX[sl] / 1e3, YY[sl] / 1e3, HH[sl], cmap="viridis_r",
                               vmin=0, vmax=max(3 * s["depth"], 60), shading="auto")
            a.contour(XX[sl] / 1e3, YY[sl] / 1e3, np.nan_to_num(HH[sl]), [s["depth"] + 1],
                      colors="w", linewidths=.8)
            a.plot(x / 1e3, y / 1e3, "rv"); a.set_aspect(1)
            a.set_title(f"{s['pair']} — {lab}\ncol {table[lab][k]['depth']:.0f} m", fontsize=9)
            plt.colorbar(pcm, ax=a, shrink=.7)
    fig.tight_layout(); fig.savefig(C.OUT / "fig_sills.png", dpi=95); plt.close(fig)

    # grille verticale
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    zc = np.cumsum(dz) - dz / 2
    ax[0].plot(dz, -zc, ".-"); ax[0].set_xlabel("dz (m)"); ax[0].set_ylabel("z (m)")
    for s in table[pkey]:
        if np.isfinite(s["depth"]):
            ax[0].axhline(-s["depth"], color="r", lw=.6)
    ax[0].set_title(f"Nr={len(dz)}, somme {dz.sum():.0f} m")
    ax[1].plot(dz[1:] / dz[:-1], -zc[1:], ".-"); ax[1].axvline(C.R_MAX_ALLOWED, color="r")
    ax[1].set_xlabel("dz(k+1)/dz(k)")
    fig.tight_layout(); fig.savefig(C.OUT / "fig_vertical.png", dpi=95); plt.close(fig)

    rep += ["## Figures", "- fig_parent.png, fig_sills.png, fig_vertical.png (+ fig_child.png si enfant)"]
    (C.OUT / "verification.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
