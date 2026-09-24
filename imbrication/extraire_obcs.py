#!/usr/bin/env python3
"""
Extraction parent -> enfant : OBCS hors ligne et conditions initiales.

Entrees
  - un run parent MITgcm : fichiers de grille MDS, instantanes 3D (UVEL, VVEL,
    THETA, SALT, et WVEL pour un enfant non hydrostatique) et instantanes ETAN ;
  - un dossier enfant contenant enfant.json (grille, OB, temps ; voir
    cas_test/gen_enfant.py) et sa bathymetrie.

Methode
  - Grilles cartesiennes uniformes, faces de l'enfant alignees sur celles du
    parent, meme grille verticale.
  - Interpolation bilineaire horizontale ; les points secs du parent sont d'abord
    remplis par la valeur mouillee la plus proche (niveau par niveau).
  - Interpolation lineaire en temps. L'enregistrement n tombe a
    t = (n - 1/2) P (temps de l'enfant) ; on ecrit N + 1 enregistrements, puis
    l'etat a t = -P/2, et externForcingCycle = (N + 2) P.
  - Correction de flux par frontiere : pour chaque frontiere et chaque
    enregistrement, une vitesse uniforme est ajoutee a la vitesse normale pour
    que le debit entrant dans l'enfant egale le debit du parent a travers la
    meme ligne de faces, moins le remplissage des cellules OB (entre la ligne
    du parent et la face ou MITgcm impose la vitesse). En z*, le debit de
    l'enfant compte le facteur (1 + eta_OB/H) qu'utilise obcs_apply_r_star.F.
  - Parent en z* et enfant en surface libre lineaire (config B) : la vitesse
    verticale imposee est la vitesse vraie w = w*(1 + eta/H) + (1 - z/H) deta/dt.

Memoire : le parent est lu par bandes (memmap), les champs initiaux niveau par
niveau.

Usage
  python3 imbrication/extraire_obcs.py PARENT_RUN ENFANT_DIR [--state3d state3D] [--eta eta2D]
Ecrit ENFANT_DIR/input/OB*.bin, *_ini.bin et ENFANT_DIR/obcs_rapport.txt.
"""
import argparse, json, os, sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sagdiag"))
import sagdiag as sd

SIGN = {"W": 1.0, "E": -1.0, "S": 1.0, "N": -1.0}   # debit entrant positif


# =============================================================== lecture MDS
def lire_bloc(base, rec, j0, j1, i0, i1):
    """Enregistrement `rec` de base(.meta/.data), limite a [j0:j1, i0:i1] -> (nz, nj, ni) float64.
    Fichiers globaux ou par tuiles ; seules les tuiles concernees sont lues (memmap)."""
    files = sd._meta_files(base)
    if not files:
        raise FileNotFoundError(base + ".meta")
    out = None
    for f in files:
        m = sd.parse_meta(f)
        nd = int(m["nDims"][0])
        d = np.array(m["dimList"], int).reshape(nd, 3)
        lo, hi = d[:, 1] - 1, d[:, 2]
        nz = int(d[2, 0]) if nd == 3 else 1
        if out is None:
            out = np.zeros((nz, j1 - j0, i1 - i0))
        a0, a1 = max(i0, lo[0]), min(i1, hi[0])
        b0, b1 = max(j0, lo[1]), min(j1, hi[1])
        if a0 >= a1 or b0 >= b1:
            continue
        prec = ">f4" if "32" in m["dataprec"] else ">f8"
        nrec = int(m["nrecords"][0])
        mm = np.memmap(f[:-5] + ".data", dtype=prec, mode="r",
                       shape=(nrec, nz, hi[1] - lo[1], hi[0] - lo[0]))
        out[:, b0 - j0:b1 - j0, a0 - i0:a1 - i0] = mm[rec, :, b0 - lo[1]:b1 - lo[1], a0 - lo[0]:a1 - lo[0]]
        del mm
    return out


def hfac_c(H, dz, hfacmin, hfacmindr):
    """hFacC comme MITgcm (coordonnee z, cellules partielles)."""
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    hf = np.zeros((len(dz),) + np.shape(H))
    for k in range(len(dz)):
        mn = max(hfacmin, min(hfacmindr / dz[k], 1.0))
        t = np.clip((H - zf[k]) / dz[k], 0.0, 1.0)
        hf[k] = np.where(t < mn, np.where(t < 0.5 * mn, 0.0, mn), t)
    return hf


# ==================================================================== parent
class Parent:
    def __init__(self, rundir, state3d=None, eta=None):
        self.rundir = rundir
        info = sd.scan_run(rundir)
        self.DRF = sd.read_mds(os.path.join(rundir, "DRF"))[0].ravel().astype(float)
        self.nr = self.DRF.size
        s3, _, et, _ = sd.guess_prefixes(info, self.nr)
        self.s3, self.eta = state3d or s3, eta or et
        if not self.s3 or not self.eta:
            raise SystemExit("instantanes 3D (UVEL, VVEL...) ou ETAN introuvables dans " + rundir)
        self.fl3 = info[self.s3]["fields"]
        self.fle = info[self.eta]["fields"]
        for n in ("UVEL", "VVEL", "THETA", "SALT"):
            assert n in self.fl3, f"{n} manque dans {self.s3}"
        XG = sd.read_mds(os.path.join(rundir, "XG"))[0][0]
        YG = sd.read_mds(os.path.join(rundir, "YG"))[0][0]
        DXG = sd.read_mds(os.path.join(rundir, "DXG"))[0][0]
        DYG = sd.read_mds(os.path.join(rundir, "DYG"))[0][0]
        self.ny, self.nx = XG.shape
        self.dx, self.dy = float(np.median(DXG)), float(np.median(DYG))
        if np.ptp(DXG) > 1e-6 * self.dx or np.ptp(DYG) > 1e-6 * self.dy:
            raise SystemExit("le parent doit avoir une grille cartesienne uniforme")
        self.xg0, self.yg0 = float(XG[0, 0]), float(YG[0, 0])
        data = os.path.join(rundir, "data")
        self.dt = sd.nml_value(data, "deltaT") or sd.nml_value(data, "deltaTMom")
        self.zstar = (int(sd.nml_value(data, "nonlinFreeSurf", 0) or 0) > 0
                      and int(sd.nml_value(data, "select_rStar", 0) or 0) > 0)
        self.t3 = np.array(sd.list_iters(rundir, self.s3)) * self.dt
        self.te = np.array(sd.list_iters(rundir, self.eta)) * self.dt
        self._cache = {}

    def grille(self, name, box):
        return lire_bloc(os.path.join(self.rundir, name), 0, *box)

    def _snap(self, prefix, fields, t, name, box):
        key = (prefix, t, name, box)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            it = int(round(t / self.dt))
            self._cache[key] = lire_bloc(os.path.join(self.rundir, f"{prefix}.{it:010d}"),
                                         fields.index(name), *box)
        return self._cache[key]

    def champ(self, name, t, box):
        """Champ du parent a l'instant t (s), interpole lineairement entre instantanes."""
        pre, tt, fl = (self.eta, self.te, self.fle) if name == "ETAN" else (self.s3, self.t3, self.fl3)
        k = int(np.searchsorted(tt, t - 1e-6))
        if k < tt.size and abs(tt[k] - t) < 1e-6:
            return self._snap(pre, fl, tt[k], name, box)
        if k == 0 or k >= tt.size:
            raise SystemExit(f"t = {t:.0f} s hors des instantanes {pre} ({tt[0]:.0f}-{tt[-1]:.0f} s)")
        w = (t - tt[k - 1]) / (tt[k] - tt[k - 1])
        return (1 - w) * self._snap(pre, fl, tt[k - 1], name, box) + w * self._snap(pre, fl, tt[k], name, box)


def remplir(f, mask):
    """Remplit les points secs par la valeur mouillee la plus proche (niveau par niveau) ;
    un niveau entierement sec reprend le niveau du dessus."""
    f = f.copy()
    for k in range(f.shape[0]):
        m = mask[k]
        if m.all():
            continue
        if not m.any():
            f[k] = f[k - 1] if k > 0 else 0.0
            continue
        idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
        f[k] = f[k][idx[0], idx[1]]
    return f


def interp(f, fj, fi):
    """Bilineaire : f (nz, nj, ni), indices fractionnaires (npts) -> (nz, npts)."""
    return np.stack([ndimage.map_coordinates(f[k], [fj, fi], order=1, mode="nearest")
                     for k in range(f.shape[0])])


# ==================================================================== enfant
class Enfant:
    def __init__(self, d):
        self.dir = d
        c = json.load(open(os.path.join(d, "enfant.json")))
        self.c = c
        self.nx, self.ny, self.dx, self.dy = c["nx"], c["ny"], c["dx"], c["dy"]
        self.x0, self.y0 = c["x0"], c["y0"]
        self.dz = np.array(c["dz"], float)
        self.nr = self.dz.size
        H = -np.fromfile(os.path.join(d, c["bathy"]), ">f8").reshape(self.ny, self.nx)
        self.hC = hfac_c(H, self.dz, c["hFacMin"], c["hFacMinDr"])
        self.hW = np.zeros_like(self.hC); self.hW[:, :, 1:] = np.minimum(self.hC[:, :, :-1], self.hC[:, :, 1:])
        self.hS = np.zeros_like(self.hC); self.hS[:, 1:, :] = np.minimum(self.hC[:, :-1, :], self.hC[:, 1:, :])
        self.ob = {k: np.array(v, int) for k, v in c["ob"].items()}

    def cellules_ob(self, side):
        """Indices (j, i) des cellules OB d'une frontiere."""
        q = self.ob[side]
        if side in "WE":
            return q, np.full(q.size, 0 if side == "W" else self.nx - 1)
        return np.full(q.size, 0 if side == "S" else self.ny - 1), q

    def alimente(self, side):
        """True si la face normale imposee alimente une cellule interieure (mouillee, non OB).
        Au coin de deux frontieres, la face peut deboucher sur une cellule OB voisine : son
        debit ne compte pas dans le bilan de l'interieur."""
        interieur = self.hC[0] > 0
        for s in self.ob:
            interieur[self.cellules_ob(s)] = False
        jo, io = self.cellules_ob(side)
        dj, di = {"W": (0, 1), "E": (0, -1), "S": (1, 0), "N": (-1, 0)}[side]
        return interieur[jo + dj, io + di]

    def points(self, side):
        """Coordonnees (x, y) des points OB : centre C, face normale, face tangentielle ;
        epaisseurs de la face normale (nz, npts)."""
        dx, dy, x0, y0 = self.dx, self.dy, self.x0, self.y0
        q = self.ob[side]
        if side in "WE":
            i = 0 if side == "W" else self.nx - 1
            iu = 1 if side == "W" else self.nx - 1          # face ou MITgcm impose u
            yc = y0 + (q + 0.5) * dy
            C = (np.full(q.size, x0 + (i + 0.5) * dx), yc)
            Nf = (np.full(q.size, x0 + iu * dx), yc)
            T = (C[0], y0 + q * dy)
            hn = self.hW[:, q, iu] * self.dz[:, None] * dy
        else:
            j = 0 if side == "S" else self.ny - 1
            jv = 1 if side == "S" else self.ny - 1
            xc = x0 + (q + 0.5) * dx
            C = (xc, np.full(q.size, y0 + (j + 0.5) * dy))
            Nf = (xc, np.full(q.size, y0 + jv * dy))
            T = (x0 + q * dx, C[1])
            hn = self.hS[:, jv, q] * self.dz[:, None] * dx
        return C, Nf, T, hn


# ================================================================ extraction
def boite(P, xs, ys, marge=3):
    """Bloc d'indices du parent (j0, j1, i0, i1) couvrant les points (x, y)."""
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    i0 = int(np.floor((xs.min() - P.xg0) / P.dx)) - marge
    i1 = int(np.floor((xs.max() - P.xg0) / P.dx)) + marge + 1
    j0 = int(np.floor((ys.min() - P.yg0) / P.dy)) - marge
    j1 = int(np.floor((ys.max() - P.yg0) / P.dy)) + marge + 1
    return (max(j0, 0), min(j1, P.ny), max(i0, 0), min(i1, P.nx))


def frac(P, box, x, y, stag):
    """Indices fractionnaires dans le bloc ; stag = 'C', 'U' ou 'V'."""
    fi = (x - P.xg0) / P.dx - (0.0 if stag == "U" else 0.5) - box[2]
    fj = (y - P.yg0) / P.dy - (0.0 if stag == "V" else 0.5) - box[0]
    return fj, fi


class Bord:
    """Donnees du parent pour une frontiere de l'enfant."""
    def __init__(self, P, E, side):
        self.side = side
        self.C, self.Nf, self.T, self.hn = E.points(side)
        # ligne de faces du parent sur le bord de l'enfant + cellules du parent concernees
        if side in "WE":
            xe = E.x0 if side == "W" else E.x0 + E.nx * E.dx
            self.pf = int(round((xe - P.xg0) / P.dx))
            self.pc = np.unique(np.floor((self.C[1] - P.yg0) / P.dy).astype(int))
            pts = ([xe - P.dx, xe + P.dx], [self.C[1].min(), self.C[1].max()])
        else:
            ye = E.y0 if side == "S" else E.y0 + E.ny * E.dy
            self.pf = int(round((ye - P.yg0) / P.dy))
            self.pc = np.unique(np.floor((self.C[0] - P.xg0) / P.dx).astype(int))
            pts = ([self.C[0].min(), self.C[0].max()], [ye - P.dy, ye + P.dy])
        self.box = boite(P, [self.C[0], self.T[0], pts[0]], [self.C[1], self.T[1], pts[1]])
        b = self.box
        hC = P.grille("hFacC", b); hW = P.grille("hFacW", b); hS = P.grille("hFacS", b)
        self.mC, self.mW, self.mS = hC > 0, hW > 0, hS > 0
        # geometrie de la ligne du parent (face normale)
        if side in "WE":
            fi = self.pf - b[2]; rows = self.pc - b[0]
            self.hp = hW[:, rows, fi] * P.DRF[:, None] * P.dy
            self.cells = (rows, fi - 1, rows, fi)            # cellules de part et d'autre
        else:
            fj = self.pf - b[0]; cols = self.pc - b[2]
            self.hp = hS[:, fj, cols] * P.DRF[:, None] * P.dx
            self.cells = (fj - 1, cols, fj, cols)
        self.Hp = self.hp.sum(0) / (P.dy if side in "WE" else P.dx)
        self.depthC = (hC * P.DRF[:, None, None]).sum(0)
        self.hC = hC

    def debit_parent(self, P, t):
        """Debit entrant du parent a travers la ligne de faces (m3/s)."""
        b = self.box
        name = "UVEL" if self.side in "WE" else "VVEL"
        v = P.champ(name, t, b)
        vn = v[:, self.cells[2], self.cells[3]]
        fac = 1.0
        if P.zstar:
            e = P.champ("ETAN", t, b)[0]
            ef = 0.5 * (e[self.cells[0], self.cells[1]] + e[self.cells[2], self.cells[3]])
            fac = 1.0 + ef / np.where(self.Hp > 0, self.Hp, 1.0)
        return SIGN[self.side] * float((vn * self.hp * fac).sum())

    def extraire(self, P, E, t, dtd):
        """Valeurs brutes aux points OB de l'enfant a l'instant parent t."""
        b, s = self.box, self.side
        nm, tm = ("UVEL", "VVEL") if s in "WE" else ("VVEL", "UVEL")
        mn, mt = (self.mW, self.mS) if s in "WE" else (self.mS, self.mW)
        st_n, st_t = ("U", "V") if s in "WE" else ("V", "U")
        out = {"n": interp(remplir(P.champ(nm, t, b), mn), *frac(P, b, *self.Nf, st_n)),
               "t_": interp(remplir(P.champ(tm, t, b), mt), *frac(P, b, *self.T, st_t))}
        fC = frac(P, b, *self.C, "C")
        for n, key in (("THETA", "t"), ("SALT", "s")):
            out[key] = interp(remplir(P.champ(n, t, b), self.mC), *fC)
        e = remplir(P.champ("ETAN", t, b), self.mC[:1])
        out["eta"] = interp(e, *fC)[0]
        # remplissage des cellules OB de l'enfant : deta/dt au centre des cellules OB
        ep = interp(remplir(P.champ("ETAN", t + dtd, b), self.mC[:1]), *fC)[0]
        em = interp(remplir(P.champ("ETAN", t - dtd, b), self.mC[:1]), *fC)[0]
        out["deta"] = (ep - em) / (2 * dtd)
        if E.c["nonhydrostatique"]:
            w = P.champ("WVEL", t, b)
            if P.zstar and not E.c["zstar"]:
                # vitesse verticale vraie : w = w*(1 + eta/H) + (1 - zf/H) deta/dt
                ep_, em_ = P.champ("ETAN", t + dtd, b)[0], P.champ("ETAN", t - dtd, b)[0]
                de = (ep_ - em_) / (2 * dtd)
                Hs = np.where(self.depthC > 0, self.depthC, 1.0)
                zf = np.concatenate([[0.0], np.cumsum(P.DRF)])[:-1]
                w = w * (1 + e[0] / Hs)[None] + (1 - zf[:, None, None] / Hs[None]) * de[None]
            out["w"] = interp(remplir(w, self.mC), *fC)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("parent"); ap.add_argument("enfant")
    ap.add_argument("--state3d"); ap.add_argument("--eta")
    ap.add_argument("--sans-correction", action="store_true", help="pas de correction de flux (test)")
    a = ap.parse_args()

    P = Parent(a.parent, a.state3d, a.eta)
    E = Enfant(a.enfant)
    c = E.c
    assert P.nr == E.nr and np.allclose(P.DRF, E.dz, atol=1e-3), "grilles verticales differentes"
    assert "WVEL" in P.fl3 or not c["nonhydrostatique"], "WVEL requis pour un enfant non hydrostatique"
    Pp, t0 = c["periode"], c["t0"]
    N = int(round(c["duree"] / Pp))
    trec = np.concatenate([(np.arange(1, N + 2) - 0.5) * Pp, [-0.5 * Pp]])    # temps enfant
    nrec = trec.size
    inp = os.path.join(E.dir, "input")
    print(f"parent {P.nx}x{P.ny}x{P.nr} dx={P.dx:.0f} m {'z*' if P.zstar else 'SL lineaire'} ; "
          f"{P.s3} {P.t3.size} instantanes, {P.eta} {P.te.size}")
    print(f"enfant {E.nx}x{E.ny}x{E.nr} dx={E.dx:.0f} m ; {nrec} enregistrements de {Pp:.0f} s "
          f"(externForcingCycle = {nrec * Pp:.1f})")

    rap = [f"Extraction parent -> enfant : {os.path.abspath(a.parent)} -> {os.path.abspath(E.dir)}",
           f"t0 = {t0:.0f} s (parent), {nrec} enregistrements de {Pp:.0f} s, externForcingCycle = {nrec * Pp:.1f}", ""]
    sumQ = np.zeros(nrec)
    for side in E.ob:
        B = Bord(P, E, side)
        q = E.ob[side]
        nlen = E.ny if side in "WE" else E.nx
        arrs = {k: np.zeros((nrec, E.nr, nlen)) for k in ("n", "t_", "t", "s", "w")}
        eta = np.zeros((nrec, nlen))
        Qp = np.zeros(nrec); Qc0 = np.zeros(nrec); dl = np.zeros(nrec); rmsu = np.zeros(nrec)
        wet = B.hn > 0
        wetn = wet & E.alimente(side)[None]          # faces corrigees et comptees
        jo, io = E.cellules_ob(side)
        aob = E.dx * E.dy * (E.hC[0][jo, io] > 0)
        for r, tc in enumerate(trec):
            t = t0 + tc
            v = B.extraire(P, E, t, 0.5 * Pp)
            Hn = np.where(B.hn.sum(0) > 0, B.hn.sum(0) / (E.dy if side in "WE" else E.dx), 1.0)
            fac = (1.0 + v["eta"] / Hn)[None] if c["zstar"] else 1.0
            area = B.hn * fac
            un = v["n"] * wetn
            Qc = SIGN[side] * float((un * area).sum())
            Qt = B.debit_parent(P, t) - float((v["deta"] * aob).sum())
            d = 0.0 if a.sans_correction else (Qt - Qc) / float((area * wetn).sum()) * SIGN[side]
            un = (v["n"] + d * wetn) * wet
            Qp[r], Qc0[r], dl[r] = Qt, Qc, d
            rmsu[r] = np.sqrt((un ** 2 * area * wetn).sum() / (area * wetn).sum())
            arrs["n"][r][:, q] = un
            arrs["t_"][r][:, q] = v["t_"]
            arrs["t"][r][:, q] = v["t"]
            arrs["s"][r][:, q] = v["s"]
            if "w" in v:
                arrs["w"][r][:, q] = v["w"] * (E.hC[:, jo, io] > 0)
            eta[r, q] = v["eta"]
        sumQ += Qp
        norm, tang = ("u", "v") if side in "WE" else ("v", "u")
        wr = lambda n, x: np.asarray(x, ">f8").tofile(os.path.join(inp, f"OB{side}{n}.bin"))
        wr(norm, arrs["n"]); wr(tang, arrs["t_"]); wr("t", arrs["t"]); wr("s", arrs["s"])
        if c["nonhydrostatique"]:
            wr("w", arrs["w"])
        if c["zstar"]:
            wr("eta", eta)
        ecart = np.abs(Qp - Qc0).max() / max(np.abs(Qp).max(), 1e-9)
        line = (f"OB {side} : {q.size:4d} points | debit parent max {np.abs(Qp).max():9.1f} m3/s, "
                f"moyen {Qp[:-2].mean():8.1f} | ecart avant correction {100 * ecart:5.1f} % | "
                f"correction max {np.abs(dl).max():.4f} m/s ({100 * np.abs(dl).max() / max(rmsu.max(), 1e-9):.1f} % de u rms max)")
        print(line); rap.append(line)

    # bilan de volume : sum des debits = remplissage de l'enfant (eta du parent)
    Q = sumQ[:-2]
    boxC = boite(P, [np.array([E.x0, E.x0 + E.nx * E.dx])], [np.array([E.y0, E.y0 + E.ny * E.dy])])
    xc = E.x0 + (np.arange(E.nx) + 0.5) * E.dx; yc = E.y0 + (np.arange(E.ny) + 0.5) * E.dy
    Xc, Yc = np.meshgrid(xc, yc)
    wetC = E.hC[0] > 0
    interior = wetC.copy()
    for side in E.ob:
        interior[E.cellules_ob(side)] = False
    mCb = P.grille("hFacC", boxC)[:1] > 0
    fC = frac(P, boxC, Xc.ravel(), Yc.ravel(), "C")
    eta_at = lambda t: interp(remplir(P.champ("ETAN", t, boxC), mCb), *fC)[0].reshape(E.ny, E.nx)
    dV = np.array([((eta_at(t0 + tc + 0.5 * Pp) - eta_at(t0 + tc - 0.5 * Pp)) / Pp * interior).sum() * E.dx * E.dy
                   for tc in trec[:-2]])
    xcp = P.xg0 + (np.arange(boxC[2], boxC[3]) + 0.5) * P.dx
    ycp = P.yg0 + (np.arange(boxC[0], boxC[1]) + 0.5) * P.dy
    dans = (((ycp > E.y0) & (ycp < E.y0 + E.ny * E.dy))[:, None]
            & ((xcp > E.x0) & (xcp < E.x0 + E.nx * E.dx))[None, :])
    Ap = (mCb[0] & dans).sum() * P.dx * P.dy
    res = np.abs(Q - dV).max() / max(np.abs(Q).max(), 1e-9)
    lines = ["",
             f"Bilan de volume (debits imposes - remplissage de l'enfant avec l'eta du parent) : "
             f"max {np.abs(Q - dV).max():.1f} m3/s = {100 * res:.2f} % du debit max",
             f"Surface mouillee : enfant {wetC.sum() * E.dx * E.dy / 1e6:.2f} km2, "
             f"parent sur l'emprise {Ap / 1e6:.2f} km2"]
    for l in lines:
        print(l)
    rap += lines

    # conditions initiales a t0 (niveau par niveau)
    wetW, wetS = E.hW > 0, E.hS > 0
    xu = E.x0 + np.arange(E.nx) * E.dx; yv = E.y0 + np.arange(E.ny) * E.dy
    XU, YU = np.meshgrid(xu, yc); XV, YV = np.meshgrid(xc, yv)
    fU = frac(P, boxC, XU.ravel(), YU.ravel(), "U"); fV = frac(P, boxC, XV.ravel(), YV.ravel(), "V")
    hCb, hWb, hSb = (P.grille(n, boxC) > 0 for n in ("hFacC", "hFacW", "hFacS"))
    for name, fn, msk, ff, mout in (("THETA", "T_ini.bin", hCb, fC, None), ("SALT", "S_ini.bin", hCb, fC, None),
                                    ("UVEL", "U_ini.bin", hWb, fU, wetW), ("VVEL", "V_ini.bin", hSb, fV, wetS)):
        f = remplir(P.champ(name, t0, boxC), msk)
        with open(os.path.join(inp, fn), "wb") as fo:
            for k in range(E.nr):
                lv = interp(f[k:k + 1], *ff)[0].reshape(E.ny, E.nx)
                if mout is not None:
                    lv = lv * mout[k]
                lv.astype(">f8").tofile(fo)
    (eta_at(t0) * wetC).astype(">f8").tofile(os.path.join(inp, "Eta_ini.bin"))
    rap.append(f"Conditions initiales : parent a t0 = {t0:.0f} s (T, S, U, V, eta)")
    open(os.path.join(E.dir, "obcs_rapport.txt"), "w").write("\n".join(rap) + "\n")
    print("->", os.path.join(E.dir, "obcs_rapport.txt"))


if __name__ == "__main__":
    main()
