"""
s01_mosaic.py — Tuiles NONNA -> mosaïques nettoyées, profondeur sous NMM.
Chaque produit de C.PRODUCTS (NONNA-10, NONNA-100) est traité séparément.

Étapes (par produit)
  1. Fusion des tuiles (grille native WGS84 : 1e-4° pour n10, 1e-3° pour n100).
  2. Masque d'eau : données présentes + trous de sondage fermés.
     Un trou fermé est une ÎLE si son pourtour touche l'estran (max > ISLAND_RING_ZMAX,
     ~ZC) ; sinon c'est un trou de sondage (EAU). Les trous minuscules sont toujours de l'eau.
  3. Remplissage des trous d'eau par interpolation linéaire (Delaunay) depuis leur pourtour.
  4. ZC -> NMM : H = -z_ZC + Z0(lon).  H > 0 = sous le niveau moyen.

Biais de NONNA-100 : chaque pixel 100 m vaut la cote la PLUS HAUTE (le plus petit fond)
des sondages qu'il contient (produit « sécurité de navigation ») — vérifié sur le recouvrement
avec NONNA-10 : écart médian au max des pixels 10 m = 0,00 m. On corrige
  z_corr = z_100 - (a + b |grad z_100|)
avec (a, b) ajustés par moindres carrés sur le recouvrement (moyenne 10 m vs valeur 100 m).
Sortie : work/<produit>_clean.npz  (H, water, filled, covered, lon, lat)
"""
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from scipy import ndimage as ndi
from scipy.interpolate import griddata

import config as C


def load_mosaic(pattern):
    files = sorted(glob.glob(str(C.NONNA_DIR / "**" / pattern), recursive=True))
    srcs = [rasterio.open(f) for f in files]
    if C.CROP is not None:                     # seulement les tuiles qui touchent l'emprise
        lo0, lo1, la0, la1 = C.CROP
        keep = [s for s in srcs if s.bounds.right > lo0 and s.bounds.left < lo1
                and s.bounds.top > la0 and s.bounds.bottom < la1]
        for s in srcs:
            if s not in keep:
                s.close()
        srcs = keep; files = [s.name for s in srcs]
    if not srcs:
        return None
    for s in srcs:
        assert s.crs.to_string() == C.CRS_SRC, (s.name, s.crs)
    if C.CROP is not None:
        z, tr = merge(srcs, nodata=np.nan, bounds=(C.CROP[0], C.CROP[2], C.CROP[1], C.CROP[3]))
    else:
        z, tr = merge(srcs, nodata=np.nan)
    z = z[0].astype(np.float64)
    z[~np.isfinite(z) | (np.abs(z) > 1e30)] = np.nan
    ny, nx = z.shape
    lon = tr.c + tr.a * (np.arange(nx) + 0.5)
    lat = tr.f + tr.e * (np.arange(ny) + 0.5)
    # couverture : pixel appartenant à une tuile existante (hors tuile = « no data », pas terre)
    covered = np.zeros_like(z, bool)
    for s in srcs:
        b = s.bounds
        ii = (lon >= b.left) & (lon <= b.right)
        jj = (lat >= b.bottom) & (lat <= b.top)
        covered[np.ix_(jj, ii)] = True
    # les bords de tuiles contiennent souvent une rangée « no data » : on retire EDGE_ERODE
    # pixels de couverture sur le pourtour extérieur de la mosaïque (un produit plus grossier
    # prend alors le relais au lieu de créer une ligne de terre artificielle)
    if C.EDGE_ERODE > 0:
        covered = ndi.binary_erosion(covered, iterations=C.EDGE_ERODE, border_value=0)
    print(f"[01] {pattern}: {len(files)} tuiles -> mosaïque {nx}x{ny}, "
          f"lon [{lon[0]:.3f},{lon[-1]:.3f}] lat [{lat[-1]:.3f},{lat[0]:.3f}]")
    return z, lon, lat, covered, files


def water_mask(z, close, min_isl):
    valid = np.isfinite(z)
    if close > 0:
        pad = close + 1
        v = np.pad(valid, pad)
        closed = ndi.binary_closing(v, iterations=close)[pad:-pad, pad:-pad] | valid
    else:
        closed = valid
    filled = ndi.binary_fill_holes(closed)
    holes = filled & ~valid
    lab, n = ndi.label(holes, structure=np.ones((3, 3)))
    if n == 0:
        return valid, np.zeros_like(valid)
    ids = np.arange(1, n + 1)
    size = ndi.sum(np.ones_like(lab), lab, ids)
    ring_lab = ndi.maximum_filter(lab, size=3)
    ring_lab[~valid] = 0
    zr = np.where(valid, z, -1e9)
    rmax = ndi.maximum(zr, ring_lab, ids)
    is_water = (size < min_isl) | (rmax < C.ISLAND_RING_ZMAX)
    lut = np.zeros(n + 1, bool); lut[1:] = is_water
    gap = lut[lab]
    n_isl = int((~is_water).sum())
    print(f"[01] trous fermés : {n} ({int(is_water.sum())} trous de sondage -> eau, "
          f"{n_isl} îles), pixels comblés = {int(gap.sum())} "
          f"({100*gap.sum()/max(valid.sum(),1):.1f} % des pixels mesurés)")
    return valid | gap, gap


def fill_gaps(z, gap, valid):
    if not gap.any():
        return z
    ring = ndi.binary_dilation(gap, iterations=3) & valid
    jr, ir = np.nonzero(ring)
    jg, ig = np.nonzero(gap)
    out = z.copy()
    vals = griddata((ir, jr), z[jr, ir], (ig, jg), method="linear")
    bad = ~np.isfinite(vals)
    if bad.any():
        vals[bad] = griddata((ir, jr), z[jr, ir], (ig[bad], jg[bad]), method="nearest")
    out[jg, ig] = vals
    return out


def z0_of_lon(lon):
    st = sorted(C.Z0_STATIONS)
    return np.interp(lon, [s[0] for s in st], [s[1] for s in st])


def slope(z, lon, lat):
    """|grad z| (m/m) sur la grille lon/lat, z prolongé sous la terre par plus proche voisin."""
    v = np.isfinite(z)
    idx = ndi.distance_transform_edt(~v, return_distances=False, return_indices=True)
    ze = z[tuple(idx)]
    dx = abs(lon[1] - lon[0]) * 111_320.0 * np.cos(np.radians(lat.mean()))
    dy = abs(lat[1] - lat[0]) * 110_950.0
    gy, gx = np.gradient(ze, dy, dx)
    return np.hypot(gx, gy)


def fit_shoal_bias(z10, lon10, lat10, z100, lon100, lat100, g100):
    """Écart (valeur 100 m - moyenne des pixels 10 m) régressé sur la pente 100 m."""
    t100_a = lon100[1] - lon100[0]; t100_e = lat100[1] - lat100[0]
    ib = np.floor((lon10 - (lon100[0] - t100_a / 2)) / t100_a).astype(int)
    jb = np.floor((lat10 - (lat100[0] - t100_e / 2)) / t100_e).astype(int)
    okc = (ib >= 0) & (ib < len(lon100)); okr = (jb >= 0) & (jb < len(lat100))
    JB, IB = np.meshgrid(jb, ib, indexing="ij")
    m = np.isfinite(z10) & okr[:, None] & okc[None, :]
    key = (JB * len(lon100) + IB)[m]
    n = np.bincount(key, minlength=z100.size)
    s = np.bincount(key, weights=z10[m], minlength=z100.size)
    sel = n >= 100                                       # cellule 100 m bien couverte en 10 m
    zb = z100.ravel(); gb = g100.ravel()
    sel &= np.isfinite(zb)
    d = zb[sel] - s[sel] / n[sel]
    X = np.c_[np.ones(sel.sum()), gb[sel]]
    (a, b), *_ = np.linalg.lstsq(X, d, rcond=None)
    res = d - X @ [a, b]
    return dict(a=float(a), b=float(b), n=int(sel.sum()), bias_mean=float(d.mean()),
                rms_before=float(np.sqrt(np.mean(d ** 2))), rms_after=float(np.sqrt(np.mean(res ** 2))))


def process(tag, prm, r, bias=None):
    z, lon, lat, covered, files = r
    if bias is not None:
        corr = np.minimum(bias["a"] + bias["b"] * slope(z, lon, lat), C.N100_BIAS_CAP)
        z = np.where(np.isfinite(z), z - corr, np.nan)
        print(f"[01] {tag}: correction de biais « fond le plus haut » appliquée "
              f"(a={bias['a']:.2f} m, b={bias['b']:.1f} m ; rms {bias['rms_before']:.1f} -> "
              f"{bias['rms_after']:.1f} m sur {bias['n']} cellules de recouvrement)")
    valid = np.isfinite(z)
    water, gap = water_mask(z, prm["close"], prm["min_isl"])
    zf = fill_gaps(z, gap, valid)
    Z0 = z0_of_lon(lon)[None, :]
    H = np.where(water, -zf + Z0, np.nan)                 # m sous NMM
    print(f"[01] {tag}: H max {np.nanmax(H):.1f} m sous NMM ; "
          f"pixels au-dessus du NMM : {int((H < 0).sum())}")
    np.savez_compressed(C.WORK / f"{tag}_clean.npz",
                        H=H.astype(np.float32), water=water, filled=gap,
                        covered=covered, lon=lon, lat=lat,
                        files=np.array([str(f) for f in files]))


def main():
    C.WORK.mkdir(parents=True, exist_ok=True)
    print(f"[01] Z0 (NMM au-dessus du ZC) = {C.Z0_STATIONS}")
    raw = {}
    for tag, prm in C.PRODUCTS.items():
        r = load_mosaic(prm["glob"])
        if r is None:
            print(f"[01] {tag}: aucune tuile {prm['glob']} dans {C.NONNA_DIR} -> ignoré")
            (C.WORK / f"{tag}_clean.npz").unlink(missing_ok=True)
        else:
            raw[tag] = r
    bias = None
    if "n100" in raw:
        z1, lo1, la1 = raw["n100"][:3]
        if "n10" in raw:
            z0, lo0, la0 = raw["n10"][:3]
            bias = fit_shoal_bias(z0, lo0, la0, z1, lo1, la1, slope(z1, lo1, la1))
        else:
            bias = C.N100_BIAS_DEFAULT
        (C.WORK / "n100_bias.json").write_text(__import__("json").dumps(bias, indent=2))
    for tag, r in raw.items():
        process(tag, C.PRODUCTS[tag], r, bias if tag == "n100" else None)
    assert any((C.WORK / f"{t}_clean.npz").exists() for t in C.PRODUCTS), "aucune donnée NONNA"


if __name__ == "__main__":
    main()
