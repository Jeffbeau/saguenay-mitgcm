#!/usr/bin/env python3
"""
sagdiag -- diagnostics tourbillonnaires pour MITgcm (fjord du Saguenay)
=======================================================================
Dependances : numpy, scipy.  (matplotlib seulement pour sagdiag_run.py)
Lecteur MDS autonome : fichiers globaux ou par tuile (runs MPI), float32/64.

Decomposition (cahier des charges) :  q = <q>_phi + q'
  <q>_phi = moyenne de phase M2 = moyenne des instantanes a phase egale
            sur les cycles complets apres la rampe.
Calcul hors ligne en deux passes, instantane par instantane :
  passe 1 : sommes par phase ecrites sur disque (memmap float32) ;
  passe 2 : ecarts q' -> EKE, productions, flux turbulents, par phase.
La memoire vive reste de l'ordre de quelques champs 3D (VM de 3,8 Go ok).
"""
import csv, glob, math, os, re
import numpy as np
from scipy import ndimage

G = 9.81

# =================================================================== namelists
def nml_value(path, name, default=None):
    """Premiere valeur scalaire de `name` dans un fichier namelist MITgcm."""
    if not os.path.exists(path):
        return default
    txt = "\n".join(l.split("#")[0] for l in open(path, errors="ignore").read().splitlines())
    m = re.search(r"(?<![\w(])" + re.escape(name) + r"\s*=\s*([^,\n]+)", txt, re.I)
    if not m:
        return default
    v = m.group(1).strip()
    if v.upper() in (".TRUE.", "T", ".T."):
        return True
    if v.upper() in (".FALSE.", "F", ".F."):
        return False
    try:
        return float(v.replace("D", "E").replace("d", "e"))
    except ValueError:
        return v.strip("'\" ")


def diag_levels(rundir, prefix):
    """Niveaux (1-based) d'une liste de data.diagnostics dont fileName == prefix."""
    p = os.path.join(rundir, "data.diagnostics")
    if not os.path.exists(p):
        return None
    txt = "\n".join(l.split("#")[0] for l in open(p, errors="ignore").read().splitlines())
    m = re.search(r"fileName\s*\(\s*(\d+)\s*\)\s*=\s*'" + re.escape(prefix) + r"\s*'", txt, re.I)
    if not m:
        return None
    n = m.group(1)
    m2 = re.search(r"levels\s*\(\s*[^,)]*,\s*" + n + r"\s*\)\s*=\s*([0-9.,\s]+)", txt, re.I)
    if not m2:
        return None
    return [int(float(x)) for x in m2.group(1).replace(",", " ").split()]

# ========================================================================= MDS
_META = re.compile(r"(\w+)\s*=\s*\[(.*?)\];", re.S)


def parse_meta(fn):
    txt = open(fn).read()
    meta = {}
    for key, val in _META.findall(txt):
        val = val.strip()
        if key == "dataprec":
            meta[key] = val.strip("' ")
            continue
        out = []
        for tk in val.replace(",", " ").split():
            try:
                out.append(int(tk))
            except ValueError:
                try:
                    out.append(float(tk.replace("D", "E")))
                except ValueError:
                    out = val
                    break
        meta[key] = out
    m = re.search(r"fldList\s*=\s*\{(.*?)\};", txt, re.S)
    meta["fldList"] = [s.strip() for s in re.findall(r"'([^']*)'", m.group(1))] if m else []
    return meta


def _meta_files(base):
    f = glob.glob(base + ".meta")
    return f if f else sorted(glob.glob(base + ".[0-9][0-9][0-9].[0-9][0-9][0-9].meta"))


def read_mds(base, recs=None, levels=None, dtype=np.float32):
    """Lit base(.meta/.data) global ou par tuiles -> (nrec, [nz,] ny, nx), meta.
    Seuls les enregistrements / niveaux demandes sont lus (memmap)."""
    files = _meta_files(base)
    if not files:
        raise FileNotFoundError(base + ".meta")
    m0 = parse_meta(files[0])
    nd = int(m0["nDims"][0])
    gshape = np.array(m0["dimList"], int).reshape(nd, 3)[:, 0][::-1]
    prec = ">f4" if "32" in m0["dataprec"] else ">f8"
    nrec = int(m0["nrecords"][0])
    rl = list(range(nrec)) if recs is None else [int(r) for r in np.atleast_1d(recs)]
    if nd == 3:
        kl = np.arange(gshape[0]) if levels is None else np.atleast_1d(levels)
        out = np.zeros((len(rl), len(kl), gshape[1], gshape[2]), dtype)
    else:
        out = np.zeros((len(rl),) + tuple(gshape), dtype)
    for f in files:
        m = m0 if f == files[0] else parse_meta(f)
        d = np.array(m["dimList"], int).reshape(nd, 3)
        lo, hi = d[:, 1] - 1, d[:, 2]
        mm = np.memmap(f[:-5] + ".data", dtype=prec, mode="r",
                       shape=(nrec,) + tuple((hi - lo)[::-1]))
        for ir, r in enumerate(rl):
            if nd == 3:
                out[ir, :, lo[1]:hi[1], lo[0]:hi[0]] = mm[r][kl]
            elif nd == 2:
                out[ir, lo[1]:hi[1], lo[0]:hi[0]] = mm[r]
            else:
                out[ir, lo[0]:hi[0]] = mm[r]
        del mm
    return out, m0


def read_fields(base, levels=None):
    """dict nom -> tableau pour un fichier de diagnostics (une liste)."""
    arr, meta = read_mds(base, levels=levels)
    names = meta["fldList"]
    if not names:
        return {"field": arr[0]}, meta
    if len(names) == arr.shape[0]:
        return {n: arr[i] for i, n in enumerate(names)}, meta
    per = arr.shape[0] // len(names)          # liste 2D ecrite niveau par niveau
    return {n: arr[i * per:(i + 1) * per] for i, n in enumerate(names)}, meta


def list_iters(rundir, prefix):
    pat = re.compile(re.escape(prefix) + r"\.(\d{10})(?:\.\d{3}\.\d{3})?\.meta$")
    return sorted({int(m.group(1)) for m in map(pat.match, os.listdir(rundir)) if m})


def scan_run(rundir):
    """Inventaire des sorties : prefixe -> champs, nz, moyenne ou instantane."""
    pat = re.compile(r"(.+?)\.(\d{10})(?:\.\d{3}\.\d{3})?\.meta$")
    first = {}
    for f in sorted(os.listdir(rundir)):
        m = pat.match(f)
        if m and not m.group(1).lower().startswith("pickup"):
            first.setdefault(m.group(1), f)
    info = {}
    for p, f in first.items():
        mt = parse_meta(os.path.join(rundir, f))
        nd = int(mt["nDims"][0])
        dl = np.array(mt["dimList"], int).reshape(nd, 3)
        ti = mt.get("timeInterval", [])
        avg = isinstance(ti, list) and len(ti) == 2 and ti[1] > ti[0]
        info[p] = dict(fields=mt["fldList"], nz=int(dl[2, 0]) if nd == 3 else 1, avg=avg)
    return info


def guess_prefixes(info, nr):
    """(3D instantanes, niveaux 2D instantanes, ETAN instantane, flux moyens)."""
    snap = {p: v for p, v in info.items() if not v["avg"]}
    has = lambda v, *n: all(x in v["fields"] for x in n)
    s3 = sorted([p for p, v in snap.items() if v["nz"] == nr and has(v, "UVEL", "VVEL")],
                key=lambda p: -len(snap[p]["fields"]))
    l2 = [p for p, v in snap.items() if 1 <= v["nz"] < nr and has(v, "UVEL", "VVEL")]
    et = sorted([p for p, v in snap.items() if "ETAN" in v["fields"]],
                key=lambda p: len(snap[p]["fields"]))
    fl = [p for p, v in info.items() if v["avg"] and has(v, "USLTMASS")]
    return (s3[0] if s3 else None, l2[0] if l2 else None,
            et[0] if et else None, fl[0] if fl else None)

# ======================================================================= grille
class Grid:
    def __init__(self, rundir):
        self.rundir = rundir
        rd = lambda n: read_mds(os.path.join(rundir, n))[0][0]
        for n in ("XC", "YC", "XG", "YG", "DXC", "DYC", "DXG", "DYG", "RAC", "RAZ", "Depth"):
            setattr(self, n, rd(n).astype(np.float64))
        for n in ("hFacC", "hFacW", "hFacS"):
            setattr(self, n, rd(n).astype(np.float32))
        self.DRF = rd("DRF").ravel().astype(np.float64)
        self.RC = rd("RC").ravel().astype(np.float64)
        self.nz, self.ny, self.nx = self.hFacC.shape
        self.maskC = self.hFacC > 0
        self.dz = (self.hFacC * self.DRF[:, None, None]).astype(np.float32)   # m (reference z*)
        self.dxF = self.DXC.copy(); self.dxF[:, :-1] = 0.5 * (self.DXC[:, :-1] + self.DXC[:, 1:])
        self.dyF = self.DYC.copy(); self.dyF[:-1, :] = 0.5 * (self.DYC[:-1, :] + self.DYC[1:, :])
        data = os.path.join(rundir, "data")
        self.dt = (nml_value(data, "deltaT") or nml_value(data, "deltaTClock")
                   or nml_value(data, "deltaTmom"))
        self.rho0 = nml_value(data, "rhoConst") or nml_value(data, "rhoNil") or 999.8
        self.cartesian = bool(nml_value(data, "usingCartesianGrid", False))
        if self.cartesian:
            f0 = nml_value(data, "f0", 1.0e-4); beta = nml_value(data, "beta", 0.0)
            self.fC = f0 + beta * self.YC
        else:
            self.fC = 2 * 7.2921e-5 * np.sin(np.deg2rad(self.YC))
        self.dx_mean = float(np.sqrt(np.median(self.RAC[self.maskC[0]])))
        # bandes d'eponge OBCS exclues des integrales (bande de spongeThickness le long des 4 bords)
        dobcs = os.path.join(rundir, "data.obcs")
        nsp = int(nml_value(dobcs, "spongeThickness", 0) or 0) if nml_value(dobcs, "useOBCSsponge", False) else 0
        self.nsponge = nsp
        self.interior = np.ones((self.ny, self.nx), bool)
        if nsp:
            self.interior[:nsp, :] = self.interior[-nsp:, :] = False
            self.interior[:, :nsp] = self.interior[:, -nsp:] = False

    def to_metres(self, x, y):
        """Coordonnees de grille -> m (identite en cartesien)."""
        if self.cartesian:
            return np.asarray(x, float), np.asarray(y, float)
        lat0 = float(np.mean(self.YC))
        return (np.asarray(x) * 111320.0 * math.cos(math.radians(lat0)),
                np.asarray(y) * 110540.0)

    def region_mask(self, box):
        x0, x1, y0, y1 = box
        return ((self.XC >= x0) & (self.XC <= x1) & (self.YC >= y0) & (self.YC <= y1)
                & self.maskC[0] & self.interior)

    def section(self, name, sec):
        """Coupe a x constant (faces u) ou y constant (faces v), bornee."""
        if "x" in sec:
            i = int(np.argmin(np.abs(self.XG[self.ny // 2, :] - sec["x"])))
            y0, y1 = sec.get("y", [-np.inf, np.inf])
            jj = np.where((self.YC[:, i] >= y0) & (self.YC[:, i] <= y1))[0]
            area = self.hFacW[:, jj, i] * self.DYG[jj, i][None] * self.DRF[:, None]
            return dict(name=name, kind="x", i=i, idx=jj, area=area.astype(np.float64))
        j = int(np.argmin(np.abs(self.YG[:, self.nx // 2] - sec["y"])))
        x0, x1 = sec.get("x", [-np.inf, np.inf])
        ii = np.where((self.XC[j, :] >= x0) & (self.XC[j, :] <= x1))[0]
        area = self.hFacS[:, j, ii] * self.DXG[j, ii][None] * self.DRF[:, None]
        return dict(name=name, kind="y", j=j, idx=ii, area=area.astype(np.float64))


def print_levels(rundir, depths=(1.0, 10.0, 30.0)):
    """Indices (1-based) des niveaux les plus proches de `depths` (pour data.diagnostics)."""
    rc = -read_mds(os.path.join(rundir, "RC"))[0].ravel()
    ks = [int(np.argmin(np.abs(rc - d))) + 1 for d in depths]
    for d, k in zip(depths, ks):
        print(f"~{d:g} m -> niveau {k} (centre a {rc[k-1]:.2f} m)")
    return ks

# ============================================================ grille C : outils
def _nan_full(shape):
    return np.full(shape, np.nan, np.float32)


def c2c(q):
    """Coins (point de vorticite SW = [j,i]) -> centres, moyenne des coins finis."""
    a = np.stack([q[..., :-1, :-1], q[..., :-1, 1:], q[..., 1:, :-1], q[..., 1:, 1:]])
    ok = np.isfinite(a)
    n = ok.sum(0)
    out = _nan_full(q.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[..., :-1, :-1] = np.where(n > 0, np.where(ok, a, 0).sum(0) / np.maximum(n, 1), np.nan)
    return out


def u2c(u, sq=False):
    a = u * u if sq else u
    out = _nan_full(u.shape); out[..., :-1] = 0.5 * (a[..., :-1] + a[..., 1:]); return out


def v2c(v, sq=False):
    a = v * v if sq else v
    out = _nan_full(v.shape); out[..., :-1, :] = 0.5 * (a[..., :-1, :] + a[..., 1:, :]); return out


def w2c(w):
    out = np.empty_like(w, dtype=np.float32)
    out[:-1] = 0.5 * (w[:-1] + w[1:]); out[-1] = 0.5 * w[-1]
    return out


def corner_mask(wet):
    cw = np.zeros(wet.shape, bool)
    cw[..., 1:, 1:] = wet[..., 1:, 1:] & wet[..., 1:, :-1] & wet[..., :-1, 1:] & wet[..., :-1, :-1]
    return cw


def kinematics(u, v, g, wet):
    """zeta, cisaillement s_s (coins), s_n, Okubo-Weiss W (centres).  u, v sur grille C."""
    vdy, udx = v * g.DYC, u * g.DXC
    dvx = vdy[..., 1:, 1:] - vdy[..., 1:, :-1]
    duy = udx[..., 1:, 1:] - udx[..., :-1, 1:]
    zeta, ss = _nan_full(u.shape), _nan_full(u.shape)
    zeta[..., 1:, 1:] = (dvx - duy) / g.RAZ[1:, 1:]
    ss[..., 1:, 1:] = (dvx + duy) / g.RAZ[1:, 1:]
    cw = corner_mask(wet)
    zeta[~cw] = np.nan; ss[~cw] = np.nan
    uy, vx = u * g.DYG, v * g.DXG
    dux, dvy = _nan_full(u.shape), _nan_full(v.shape)
    dux[..., :, :-1] = uy[..., :, 1:] - uy[..., :, :-1]
    dvy[..., :-1, :] = vx[..., 1:, :] - vx[..., :-1, :]
    sn = (dux - dvy) / g.RAC
    sn[~wet] = np.nan
    zc, sc = c2c(zeta), c2c(ss)
    ow = sn ** 2 + sc ** 2 - zc ** 2
    ow[~wet] = np.nan
    return dict(zeta=zeta, zeta_c=zc, sn=sn, ss_c=sc, ow=ow)


def ddz(q, rc):
    """d/dz (z vers le haut) de q aux centres (nz,ny,nx), decentre pres du fond/surface."""
    dq = (q[:-1] - q[1:]) / (rc[:-1] - rc[1:])[:, None, None]
    above, below = _nan_full(q.shape), _nan_full(q.shape)
    above[1:], below[:-1] = dq, dq
    return np.where(np.isfinite(above) & np.isfinite(below), 0.5 * (above + below),
                    np.where(np.isfinite(above), above, below)).astype(np.float32)

# ================================================================== phases
def make_bins(times, period, t0=0.0):
    """Indice de phase, cycle, nb de phases, alignement, phase (deg) de chaque classe.
    Tolere un decalage constant des instantanes (timePhase de MITgcm)."""
    t = np.asarray(times, float)
    dtout = float(np.median(np.diff(t)))
    nb = int(round(period / dtout))
    step = period / nb
    rel = t - t0
    off = float(np.median(np.mod(rel, step)))
    if off > 0.5 * step:
        off -= step
    ph = np.mod(rel - off, period)
    b = np.rint(ph / step).astype(int) % nb
    mis = np.abs(ph - b * step)
    mis = np.minimum(mis, period - mis)
    aligned = bool(np.all(mis < 1e-3 * step)) and abs(nb * dtout - period) < 1e-6 * period
    cyc = np.floor((rel - off) / period + 0.5 / nb).astype(int)
    return b, cyc, nb, aligned, 360.0 * (off + np.arange(nb) * step) / period


def complete_cycles(b, cyc, nb, skip):
    """Selection des instantanes des cycles complets d'indice >= skip."""
    sel = np.zeros(b.size, bool)
    for c in np.unique(cyc):
        if c < skip:
            continue
        m = cyc == c
        if np.unique(b[m]).size == nb:
            sel |= m
    return sel

# ============================================================ analyse 3D
def phase_analysis_3d(rundir, g, prefix, period, outdir, skip=1, t0=0.0,
                      regions=None, sections=None, log=print, eta_fn=None):
    """Passe 1 (moyennes de phase, memmap) + passe 2 (statistiques des ecarts)."""
    its = np.array(list_iters(rundir, prefix))
    t = its * g.dt
    b, cyc, nb, aligned, phdeg = make_bins(t, period, t0)
    sel = complete_cycles(b, cyc, nb, skip)
    if not sel.any():
        raise RuntimeError("aucun cycle complet apres la rampe")
    ncyc = np.unique(cyc[sel]).size
    log(f"[3D] {prefix}: {its.size} instantanes, {nb} phases/cycle, {ncyc} cycles analyses"
        + ("" if aligned else "  ATTENTION : sorties non alignees sur la periode"))
    base = lambda it: os.path.join(rundir, f"{prefix}.{it:010d}")
    names = read_fields(base(its[0]))[1]["fldList"]
    want = [f for f in ("UVEL", "VVEL", "WVEL", "SALT", "THETA", "RHOAnoma") if f in names]
    shape = (nb, g.nz, g.ny, g.nx)
    os.makedirs(outdir, exist_ok=True)
    mm = {}
    for f in want:
        mm[f] = np.lib.format.open_memmap(os.path.join(outdir, f"phasemean_{f}.npy"), "w+",
                                          np.float32, shape)
        mm[f][:] = 0.0
    cnt = np.zeros(nb, int)
    for n in np.where(sel)[0]:                                     # ---- passe 1
        d, _ = read_fields(base(its[n]))
        for f in want:
            mm[f][b[n]] += d[f]
        cnt[b[n]] += 1
    for f in want:
        for k in range(nb):
            mm[f][k] /= cnt[k]
        mm[f].flush()

    regions = regions or {"domaine": None}
    rmask = {r: ((g.maskC[0] & g.interior) if box is None else g.region_mask(box)) for r, box in regions.items()}
    secs = [g.section(k, v) for k, v in (sections or {}).items()]
    acc3 = {k: np.zeros((g.nz, g.ny, g.nx)) for k in ("eke", "Ph", "Pv", "B")}
    acc2 = {k: np.zeros((nb, g.ny, g.nx)) for k in ("eke", "mke", "Ph", "Pv", "B")}
    sec_e = np.zeros((len(secs), nb)); sec_t = np.zeros((len(secs), nb)); sec_q = np.zeros((len(secs), nb))
    reg_eke_snap = []
    wet = g.maskC
    dz = g.dz
    for k in range(nb):                                            # ---- passe 2
        U, V = np.array(mm["UVEL"][k]), np.array(mm["VVEL"][k])
        Wm = np.array(mm["WVEL"][k]) if "WVEL" in mm else None
        Uc, Vc = u2c(U), v2c(V)
        Ux = _nan_full(U.shape); Ux[..., :-1] = (U[..., 1:] - U[..., :-1]) / g.dxF[:, :-1]
        Vy = _nan_full(V.shape); Vy[..., :-1, :] = (V[..., 1:, :] - V[..., :-1, :]) / g.dyF[:-1, :]
        Uy = _nan_full(U.shape); Uy[..., 1:, :] = (U[..., 1:, :] - U[..., :-1, :]) / g.DYC[1:, :]
        Vx = _nan_full(V.shape); Vx[..., :, 1:] = (V[..., :, 1:] - V[..., :, :-1]) / g.DXC[:, 1:]
        cw = corner_mask(wet)
        Uy[~cw] = np.nan; Vx[~cw] = np.nan
        shear = c2c(Uy) + c2c(Vx)
        Uz, Vz = ddz(np.where(wet, Uc, np.nan), g.RC), ddz(np.where(wet, Vc, np.nan), g.RC)
        mke = 0.25 * (u2c(U, True) + v2c(V, True))
        acc2["mke"][k] = np.nansum(np.where(wet, mke, 0) * dz, 0)
        Sm = np.array(mm["SALT"][k]) if "SALT" in mm else None
        Rm = np.array(mm["RHOAnoma"][k]) if "RHOAnoma" in mm else None
        for n in np.where(sel & (b == k))[0]:
            d, _ = read_fields(base(its[n]))
            up, vp = d["UVEL"] - U, d["VVEL"] - V
            upc, vpc = u2c(up), v2c(vp)
            eke = 0.25 * (u2c(up, True) + v2c(vp, True))
            eke = np.where(wet & np.isfinite(eke), eke, 0.0)
            Ph = -(u2c(up, True) * Ux + upc * vpc * shear + v2c(vp, True) * Vy)
            Ph = np.where(wet & np.isfinite(Ph), Ph, 0.0)
            if Wm is not None:
                wpc = w2c(d["WVEL"] - Wm)
                Pv = -(upc * wpc * Uz + vpc * wpc * Vz)
                Pv = np.where(wet & np.isfinite(Pv), Pv, 0.0)
                Bf = (-(G / g.rho0) * (d["RHOAnoma"] - Rm) * wpc) if Rm is not None else np.zeros_like(eke)
                Bf = np.where(wet & np.isfinite(Bf), Bf, 0.0)
            else:
                Pv = Bf = np.zeros_like(eke)
            for key, fld in (("eke", eke), ("Ph", Ph), ("Pv", Pv), ("B", Bf)):
                acc3[key] += fld
                acc2[key][k] += (fld * dz).sum(0)
            vint = (eke * dz).sum(0) * g.RAC
            reg_eke_snap.append([t[n], k, cyc[n]] + [float(vint[m].sum()) * g.rho0 for m in rmask.values()])
            if secs:
                # epaisseur z* : h = h0 (H + eta)/H, eta interpole au temps de l'instantane
                sf = np.ones((g.ny, g.nx)) if eta_fn is None else np.where(
                    g.Depth > 0, (g.Depth + eta_fn(t[n])) / np.where(g.Depth > 0, g.Depth, 1.0), 1.0)
                S_ = d.get("SALT")
                Sp = (S_ - Sm) if Sm is not None else None
                for s_i, s in enumerate(secs):
                    if s["kind"] == "x":
                        i, jj = s["i"], s["idx"]
                        A = s["area"] * (0.5 * (sf[jj, i - 1] + sf[jj, i]))[None, :]
                        vel, velp = d["UVEL"][:, jj, i], up[:, jj, i]
                        face = lambda q: 0.5 * (q[:, jj, i - 1] + q[:, jj, i])
                    else:
                        j, ii = s["j"], s["idx"]
                        A = s["area"] * (0.5 * (sf[j - 1, ii] + sf[j, ii]))[None, :]
                        vel, velp = d["VVEL"][:, j, ii], vp[:, j, ii]
                        face = lambda q: 0.5 * (q[:, j - 1, ii] + q[:, j, ii])
                    sec_q[s_i, k] += np.sum(A * vel)
                    if Sp is not None:
                        sec_t[s_i, k] += np.sum(A * vel * face(S_))
                        sec_e[s_i, k] += np.sum(A * velp * face(Sp))
        fb = cnt[k] / (cnt[k] - 1.0) if cnt[k] > 1 else np.nan   # biais N/(N-1) des moments d'ordre 2
        for key in ("eke", "Ph", "Pv", "B"):
            acc2[key][k] *= fb / cnt[k]
        sec_e[:, k] *= fb / cnt[k]; sec_t[:, k] /= cnt[k]; sec_q[:, k] /= cnt[k]
    nsel = sel.sum()
    fN = ncyc / (ncyc - 1.0) if ncyc > 1 else np.nan
    for key in acc3:
        acc3[key] *= fN / nsel
    reg_eke_snap = np.array(reg_eke_snap)
    if reg_eke_snap.size:
        reg_eke_snap[:, 3:] *= fN
    # totaux regionaux par phase (J pour l'energie, W pour les conversions)
    reg = {}
    for r, m in rmask.items():
        A = g.RAC * m
        reg[r] = {key: g.rho0 * (acc2[key] * A).sum((1, 2)) for key in acc2}
    res = dict(nb=nb, ncyc=ncyc, aligned=aligned, phase_deg=phdeg, count=cnt, acc3=acc3, acc2=acc2, reg=reg,
               reg_names=list(rmask), reg_eke_snap=reg_eke_snap,
               sections=[s["name"] for s in secs], sec_q=sec_q, sec_mean_salt=sec_t - sec_e, sec_total_salt=sec_t,
               sec_eddy_salt=sec_e, prefix=prefix, fields=want)
    return res


def flux_check(rundir, g, prefix, sections, period, skip=1):
    """Transport de sel moyen par cycle (USLTMASS) a travers les coupes, cycles >= skip."""
    out = {}
    its = list_iters(rundir, prefix)
    secs = [g.section(k, v) for k, v in sections.items()]
    vals = {s["name"]: [] for s in secs}
    for it in its:
        cyc = int(round(it * g.dt / period)) - 1          # moyenne du cycle precedent
        if cyc < skip:
            continue
        d, _ = read_fields(os.path.join(rundir, f"{prefix}.{it:010d}"))
        for s in secs:
            if s["kind"] == "x" and "USLTMASS" in d:
                a = s["area"] / np.where(g.hFacW[:, s["idx"], s["i"]] > 0, g.hFacW[:, s["idx"], s["i"]], 1)
                vals[s["name"]].append(float(np.sum(a * d["USLTMASS"][:, s["idx"], s["i"]])))
            elif s["kind"] == "y" and "VSLTMASS" in d:
                a = s["area"] / np.where(g.hFacS[:, s["j"], s["idx"]] > 0, g.hFacS[:, s["j"], s["idx"]], 1)
                vals[s["name"]].append(float(np.sum(a * d["VSLTMASS"][:, s["j"], s["idx"]])))
    for k, v in vals.items():
        out[k] = float(np.mean(v)) if v else float("nan")
    return out

# ================================================== niveaux 2D : tourbillons
def eddy_analysis_2d(rundir, g, prefix, period, skip=1, level=None, t0=0.0,
                     min_diam_cells=8.0, ow_k=0.2, ro_min=0.2, vmax=2.5, log=print):
    """Detection Okubo-Weiss (W < -ow_k*sigma_W) sur le champ complet, taille minimale
    min_diam_cells*dx (diametre equivalent), |Ro| >= ro_min ; fraction verrouillee en
    phase = zeta(moyenne de phase)/zeta(complet) sur le coeur ; suivi par plus proche voisin."""
    its = np.array(list_iters(rundir, prefix))
    t = its * g.dt
    b, cyc, nb, aligned, phdeg = make_bins(t, period, t0)
    sel = complete_cycles(b, cyc, nb, skip)
    base = lambda it: os.path.join(rundir, f"{prefix}.{it:010d}")
    d0, _ = read_fields(base(its[0]))
    nlev = d0["UVEL"].shape[0] if d0["UVEL"].ndim == 3 else 1
    levs = diag_levels(rundir, prefix) or list(range(1, nlev + 1))
    li = (min(1, nlev - 1) if level is None else level)
    kmod = levs[li] - 1
    wet = g.maskC[kmod]
    log(f"[2D] {prefix}: {its.size} instantanes, {nb} phases/cycle ; niveau modele {kmod+1} "
        f"({-g.RC[kmod]:.1f} m) ; taille min {min_diam_cells:g} dx")
    getuv = lambda d: ((d["UVEL"][li], d["VVEL"][li]) if d["UVEL"].ndim == 3 else (d["UVEL"], d["VVEL"]))
    Us = np.zeros((nb, g.ny, g.nx)); Vs = np.zeros((nb, g.ny, g.nx)); cnt = np.zeros(nb, int)
    for n in np.where(sel)[0]:
        u, v = getuv(read_fields(base(its[n]))[0])
        Us[b[n]] += u; Vs[b[n]] += v; cnt[b[n]] += 1
    Us /= np.maximum(cnt, 1)[:, None, None]; Vs /= np.maximum(cnt, 1)[:, None, None]
    zeta_mean = np.stack([kinematics(Us[k].astype(np.float32), Vs[k].astype(np.float32), g, wet)["zeta_c"]
                          for k in range(nb)])
    xm, ym = g.to_metres(g.XC, g.YC)
    fC = g.fC
    min_area = math.pi * (0.5 * min_diam_cells * g.dx_mean) ** 2
    rows = []
    snaps = {}
    for n in np.where(sel)[0]:
        u, v = getuv(read_fields(base(its[n]))[0])
        kin = kinematics(u, v, g, wet)
        W, zc = kin["ow"], kin["zeta_c"]
        sig = np.nanstd(W)
        core = np.isfinite(W) & (W < -ow_k * sig)
        lab, nl = ndimage.label(core)
        if nl:
            idx = lab.ravel()
            A = np.bincount(idx, g.RAC.ravel(), nl + 1)
            Ad = np.maximum(A, 1e-30)
            X = np.bincount(idx, (g.RAC * xm).ravel(), nl + 1) / Ad
            Y = np.bincount(idx, (g.RAC * ym).ravel(), nl + 1) / Ad
            Z = np.bincount(idx, (g.RAC * np.nan_to_num(zc)).ravel(), nl + 1) / Ad
            ZM = np.bincount(idx, (g.RAC * np.nan_to_num(zeta_mean[b[n]])).ravel(), nl + 1) / Ad
            F = np.bincount(idx, (g.RAC * fC).ravel(), nl + 1) / Ad
            romax = ndimage.maximum(np.abs(np.nan_to_num(zc) / fC), lab, np.arange(1, nl + 1))
            for e in range(1, nl + 1):
                ro = Z[e] / F[e]
                if A[e] < min_area or abs(ro) < ro_min:
                    continue
                rows.append(dict(t=t[n], cycle=int(cyc[n]), phase_bin=int(b[n]),
                                 phase_deg=float(phdeg[b[n]]), x=X[e], y=Y[e],
                                 r_eq=math.sqrt(A[e] / math.pi), ro=ro, ro_max=float(romax[e - 1]),
                                 sign=int(np.sign(ro)), locked=ZM[e] / Z[e] if Z[e] != 0 else np.nan))
        if int(cyc[n]) == int(cyc[sel].max()):
            snaps[int(b[n])] = zc / fC
    tid = track(rows, float(np.median(np.diff(t))), vmax)
    for r, i in zip(rows, tid):
        r["track"] = int(i)
    return dict(rows=rows, nb=nb, level_depth=-g.RC[kmod], kmod=kmod, zeta_mean_ro=zeta_mean / fC,
                last_cycle_ro=snaps, ncyc=int(np.unique(cyc[sel]).size), aligned=aligned,
                phase_deg=phdeg, dtout=float(np.median(np.diff(t))))


def track(rows, dt, vmax=2.5, max_gap=2):
    """Suivi glouton : une detection prolonge la trajectoire active de meme polarite la plus
    proche, vue il y a au plus max_gap instantanes, a une distance < vmax*ecart + 0.5*max(r)."""
    tid = np.full(len(rows), -1, int)
    if not rows:
        return tid
    t = np.array([r["t"] for r in rows])
    active, nxt, lim = {}, 0, (max_gap + 0.5) * dt
    for tt in np.unique(t):
        taken = set()
        for c in sorted(np.where(t == tt)[0], key=lambda i: -rows[i]["r_eq"]):
            best, bd = None, np.inf
            for k, p in active.items():
                gap = tt - rows[p]["t"]
                if k in taken or rows[p]["sign"] != rows[c]["sign"] or gap > lim:
                    continue
                dd = math.hypot(rows[c]["x"] - rows[p]["x"], rows[c]["y"] - rows[p]["y"])
                if dd < vmax * gap + 0.5 * max(rows[c]["r_eq"], rows[p]["r_eq"]) and dd < bd:
                    best, bd = k, dd
            if best is None:
                best, nxt = nxt, nxt + 1
            tid[c] = best; taken.add(best); active[best] = c
        active = {k: p for k, p in active.items() if tt - rows[p]["t"] <= lim}
    return tid


def eta_interpolator(rundir, g, prefix):
    """eta(t) interpole lineairement entre les instantanes ETAN (pour l'epaisseur z*)."""
    its = np.array(list_iters(rundir, prefix))
    tt = its * g.dt
    E = np.stack([read_fields(os.path.join(rundir, f"{prefix}.{it:010d}"))[0]["ETAN"].reshape(g.ny, g.nx)
                  for it in its]).astype(np.float64)
    def f(tq):
        k = int(np.searchsorted(tt, tq))
        if k <= 0:
            return E[0]
        if k >= tt.size:
            return E[-1]
        w = (tq - tt[k - 1]) / (tt[k] - tt[k - 1])
        return (1 - w) * E[k - 1] + w * E[k]
    return f


def track_table(rows, dt):
    tr = {}
    for r in rows:
        tr.setdefault(r["track"], []).append(r)
    out = []
    for k, lst in sorted(tr.items()):
        lst.sort(key=lambda r: r["t"])
        out.append(dict(track=k, sign=lst[0]["sign"], n=len(lst), duree_s=lst[-1]["t"] - lst[0]["t"] + dt,
                        t0=lst[0]["t"], phase0_deg=lst[0]["phase_deg"], x0=lst[0]["x"], y0=lst[0]["y"],
                        r_eq_moy=float(np.mean([r["r_eq"] for r in lst])),
                        ro_moy=float(np.mean([r["ro"] for r in lst])),
                        locked_moy=float(np.nanmean([r["locked"] for r in lst]))))
    return out


def write_csv(path, rows):
    if not rows:
        open(path, "w").write("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

# ============================================================ controle du run
def read_monitor(path):
    """Series %MON de STDOUT.0000 (ou de la sortie standard en serie)."""
    d = {}
    for line in open(path, errors="ignore"):
        if "%MON" in line:
            m = re.search(r"%MON\s+(\S+)\s*=\s*(\S+)", line)
            if m:
                try:
                    d.setdefault(m.group(1), []).append(float(m.group(2).replace("D", "E")))
                except ValueError:
                    pass
    n = len(d.get("time_secondsf", []))
    return {k: np.array(v) for k, v in d.items() if len(v) == n}


def eta_fit(rundir, g, prefix, period, skip=1, t0=0.0):
    """Moyenne de domaine de ETAN -> ajustement M2 + M4 sur les cycles complets.
    eta ~ A2 sin(w t + phi2) : phi2 = 0 si le forcage est en sin(w t) sans decalage."""
    its = np.array(list_iters(rundir, prefix))
    t = its * g.dt
    A = g.RAC * g.maskC[0]
    eta = np.array([(read_fields(os.path.join(rundir, f"{prefix}.{it:010d}"))[0]["ETAN"] * A).sum() / A.sum()
                    for it in its]).ravel()
    b, cyc, nb, _, _ = make_bins(t, period, t0)
    sel = complete_cycles(b, cyc, nb, skip)
    w = 2 * np.pi / period
    X = np.column_stack([np.ones(sel.sum()), np.cos(w * t[sel]), np.sin(w * t[sel]),
                         np.cos(2 * w * t[sel]), np.sin(2 * w * t[sel])])
    c = np.linalg.lstsq(X, eta[sel], rcond=None)[0]
    return dict(t=t, eta=eta, sel=sel, mean=c[0], A2=math.hypot(c[1], c[2]),
                phi2_deg=math.degrees(math.atan2(c[1], c[2])), A4=math.hypot(c[3], c[4]))
