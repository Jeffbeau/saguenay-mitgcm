#!/usr/bin/env python3
"""
Coherence de l'imbrication : parent contre enfant sur la fenetre commune.

  1. Maree : amplitude et phase M2 de eta, cellule par cellule sur la grille du
     parent (enfant moyenne par blocs), hors eponge de l'enfant.
  2. Debits aux coupes (sections du fichier de coupes situees dans l'enfant) :
     moyenne et amplitude M2 du debit, avec l'epaisseur z* si le run est en z*.
  3. Salinite : moyenne temporelle sur la fenetre, ecart rms sur les cellules
     communes, rapporte a l'etendue de S.

Critere du cahier des charges : ecarts < 5 %.

Usage
  python3 imbrication/comparer.py PARENT_RUN ENFANT_DIR [--run run] [--config coupes.json] [--skip 1]
Ecrit ENFANT_DIR/<run>/diag/imbrication.txt et .json.
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sagdiag"))
import sagdiag as sd

T_M2 = 44640.0


def zstar(rundir):
    d = os.path.join(rundir, "data")
    return (int(sd.nml_value(d, "nonlinFreeSurf", 0) or 0) > 0
            and int(sd.nml_value(d, "select_rStar", 0) or 0) > 0)


def fit_m2(rundir, g, prefix, t_lo, t_hi, toff=0.0, w=2 * np.pi / T_M2):
    """Ajustement cellule par cellule eta = a0 + a cos(wt) + b sin(wt) (+ M4), equations normales
    cumulees instantane par instantane (memoire : quelques champs 2D)."""
    its = np.array(sd.list_iters(rundir, prefix))
    t = its * g.dt + toff
    sel = (t >= t_lo - 1e-6) & (t < t_hi - 1e-6)
    A = np.zeros((5, 5)); B = np.zeros((5, g.ny * g.nx))
    for it, tt in zip(its[sel], t[sel]):
        x = np.array([1, math.cos(w * tt), math.sin(w * tt), math.cos(2 * w * tt), math.sin(2 * w * tt)])
        e = sd.read_fields(os.path.join(rundir, f"{prefix}.{it:010d}"))[0]["ETAN"].reshape(-1)
        A += np.outer(x, x); B += x[:, None] * e[None]
    c = np.linalg.solve(A, B).reshape(5, g.ny, g.nx)
    return np.hypot(c[1], c[2]), np.degrees(np.arctan2(c[1], c[2])), c[0], int(sel.sum())


def debits(rundir, g, prefix, eta_prefix, secs, t_lo, t_hi, toff, zs):
    """Debit (m3/s) a travers chaque coupe pour chaque instantane de la fenetre."""
    its = np.array(sd.list_iters(rundir, prefix))
    t = its * g.dt + toff
    sel = (t >= t_lo - 1e-6) & (t < t_hi - 1e-6)
    etaf = sd.eta_interpolator(rundir, g, eta_prefix) if zs else None
    H = (g.hFacW * g.DRF[:, None, None]).sum(0)
    out = {s["name"]: [] for s in secs}
    for it, tt in zip(its[sel], t[sel]):
        u = sd.read_fields(os.path.join(rundir, f"{prefix}.{it:010d}"))[0]["UVEL"]
        if zs:
            e = etaf(tt - toff)
            ew = e.copy(); ew[:, 1:] = 0.5 * (e[:, :-1] + e[:, 1:])
        for s in secs:
            i, jj = s["i"], s["idx"]
            fac = 1.0 + ew[jj, i] / np.where(H[jj, i] > 0, H[jj, i], 1.0) if zs else 1.0
            out[s["name"]].append(float((u[:, jj, i] * s["area"] * fac).sum()))
    tsel = t[sel]
    res = {}
    w = 2 * np.pi / T_M2
    X = np.column_stack([np.ones(tsel.size), np.cos(w * tsel), np.sin(w * tsel)])
    for k, v in out.items():
        c = np.linalg.lstsq(X, np.array(v), rcond=None)[0]
        res[k] = dict(moyen=float(c[0]), M2=float(math.hypot(c[1], c[2])),
                      phase=float(math.degrees(math.atan2(c[1], c[2]))), n=int(tsel.size))
    return res


def s_moyen(rundir, g, prefix, t_lo, t_hi, toff):
    its = np.array(sd.list_iters(rundir, prefix))
    t = its * g.dt + toff
    sel = (t >= t_lo - 1e-6) & (t < t_hi - 1e-6)
    acc = None
    for it in its[sel]:
        s = sd.read_fields(os.path.join(rundir, f"{prefix}.{it:010d}"))[0]["SALT"].astype(np.float64)
        acc = s if acc is None else acc + s
    return acc / max(sel.sum(), 1)


def blocs(q, r, poids):
    """Moyenne ponderee par blocs r x r (enfant -> cellules du parent). q (..., ny, nx)."""
    sh = q.shape[:-2] + (q.shape[-2] // r, r, q.shape[-1] // r, r)
    num = (q * poids).reshape(sh).sum((-1, -3))
    den = np.broadcast_to(poids, q.shape).reshape(sh).sum((-1, -3))
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent"); ap.add_argument("enfant")
    ap.add_argument("--run", default="run", help="sous-dossier du run de l'enfant")
    ap.add_argument("--config", default=None, help="coupes (x constant, m, repere du parent) ; sans : pas de debits")
    ap.add_argument("--skip", type=float, default=1.0, help="cycles de l'enfant ignores (ajustement)")
    a = ap.parse_args()

    c = json.load(open(os.path.join(a.enfant, "enfant.json")))
    rp, rc = a.parent, os.path.join(a.enfant, a.run)
    gp, gc = sd.Grid(rp), sd.Grid(rc)
    r = int(round(gp.dx_mean / gc.dx_mean))
    x0, y0, t0 = c["x0"], c["y0"], c["t0"]
    ip0 = int(round((x0 - gp.XG[0, 0]) / gp.dx_mean)); jp0 = int(round((y0 - gp.YG[0, 0]) / gp.dx_mean))
    nxp, nyp = gc.nx // r, gc.ny // r
    t_lo, t_hi = t0 + a.skip * T_M2, t0 + c["duree"]
    pre_p = sd.guess_prefixes(sd.scan_run(rp), gp.nz); pre_c = sd.guess_prefixes(sd.scan_run(rc), gc.nz)
    zp, zc = zstar(rp), zstar(rc)
    res = dict(fenetre_s=[t_lo, t_hi], ratio=r)
    lines = [f"Imbrication : parent {rp} / enfant {rc}",
             f"fenetre (temps du parent) : {t_lo:.0f}-{t_hi:.0f} s ({(t_hi - t_lo) / T_M2:.2f} cycle(s)) ; "
             f"parent {'z*' if zp else 'SL lineaire'}, enfant {'z*' if zc else 'SL lineaire'}", ""]

    # --- masque de comparaison (grille du parent) : hors eponge de l'enfant, cellules pleines
    nsp = int(math.ceil(c.get("sponge", 0) / r)) + 1
    wetc = gc.maskC[0].astype(float)
    frac_wet, _ = blocs(wetc, r, np.ones_like(wetc))
    msk = np.zeros((nyp, nxp), bool); msk[nsp:-nsp, nsp:-nsp] = True
    msk &= gp.maskC[0][jp0:jp0 + nyp, ip0:ip0 + nxp] & (frac_wet > 0.99)

    # --- 1. maree M2
    Ap, php, mp, n1 = fit_m2(rp, gp, pre_p[2], t_lo, t_hi)
    Ac, phc, mc, n2 = fit_m2(rc, gc, pre_c[2], t_lo, t_hi, toff=t0)
    Acb, _ = blocs(Ac, r, gc.maskC[0]); phcb, _ = blocs(phc, r, gc.maskC[0]); mcb, _ = blocs(mc, r, gc.maskC[0])
    Aps, phps = Ap[jp0:jp0 + nyp, ip0:ip0 + nxp][msk], php[jp0:jp0 + nyp, ip0:ip0 + nxp][msk]
    Acs, phcs = Acb[msk], phcb[msk]
    dA = (Acs.mean() - Aps.mean()) / Aps.mean()
    rmsA = np.sqrt(np.mean((Acs - Aps) ** 2)) / Aps.mean()
    dph = float(np.mean((phcs - phps + 180) % 360 - 180))
    biais = float(np.mean(mcb[msk] - mp[jp0:jp0 + nyp, ip0:ip0 + nxp][msk]))
    res["maree"] = dict(A_parent=float(Aps.mean()), A_enfant=float(Acs.mean()), ecart_moyen=float(dA),
                        ecart_rms=float(rmsA), dphase_deg=dph, biais_moyen_m=biais, ncell=int(msk.sum()))
    lines.append(f"Maree M2 ({msk.sum()} cellules, {n1}/{n2} instantanes) : A parent {Aps.mean():.3f} m, "
                 f"enfant {Acs.mean():.3f} m -> ecart {100 * dA:+.2f} % (rms {100 * rmsA:.2f} %), "
                 f"phase {dph:+.2f} deg ; niveau moyen enfant - parent {100 * biais:+.1f} cm "
                 f"({100 * biais / Aps.mean():+.2f} % de A)")

    # --- 2. debits aux coupes situees dans l'enfant
    cfg = json.load(open(a.config)) if a.config else {}
    xe0, xe1 = x0 + nsp * r * gc.dx_mean, x0 + gc.nx * gc.dx_mean - nsp * r * gc.dx_mean
    secp, secc = [], []
    for k, s in cfg.get("sections", {}).items():
        if "x" in s and xe0 < s["x"] < xe1:
            secp.append(gp.section(k, s))
            secc.append(gc.section(k, {"x": s["x"] - x0, "y": [v - y0 for v in s["y"]]}))
    res["debits"] = {}
    if secp:
        qp = debits(rp, gp, pre_p[0], pre_p[2], secp, t_lo, t_hi, 0.0, zp)
        qc = debits(rc, gc, pre_c[0], pre_c[2], secc, t_lo, t_hi, t0, zc)
        for k in qp:
            if qp[k]["M2"] < 1.0:                  # coupe seche ou hors du chenal
                lines.append(f"Debit {k:8s}: coupe sans debit dans le parent, ignoree")
                continue
            e2 = (qc[k]["M2"] - qp[k]["M2"]) / qp[k]["M2"]
            res["debits"][k] = dict(parent=qp[k], enfant=qc[k], ecart_M2=float(e2))
            lines.append(f"Debit {k:8s}: M2 parent {qp[k]['M2']:8.0f}, enfant {qc[k]['M2']:8.0f} m3/s -> "
                         f"{100 * e2:+.2f} % ; phase {qc[k]['phase'] - qp[k]['phase']:+.2f} deg ; "
                         f"moyen {qp[k]['moyen']:7.0f} / {qc[k]['moyen']:7.0f} m3/s")

    # --- 3. salinite moyenne
    Sp = s_moyen(rp, gp, pre_p[0], t_lo, t_hi, 0.0)[:, jp0:jp0 + nyp, ip0:ip0 + nxp]
    Sc = s_moyen(rc, gc, pre_c[0], t_lo, t_hi, t0)
    Scb, _ = blocs(Sc, r, gc.hFacC)
    m3 = msk[None] & (gp.hFacC[:, jp0:jp0 + nyp, ip0:ip0 + nxp] > 0) & np.isfinite(Scb)
    etendue = float(Sp[m3].max() - Sp[m3].min())
    rmsS = float(np.sqrt(np.mean((Scb[m3] - Sp[m3]) ** 2)))
    res["salinite"] = dict(rms=rmsS, etendue=etendue, rapport=rmsS / etendue,
                           biais=float(np.mean(Scb[m3] - Sp[m3])))
    lines.append(f"Salinite moyenne : ecart rms {rmsS:.3f} (biais {res['salinite']['biais']:+.3f}) "
                 f"pour une etendue de {etendue:.2f} -> {100 * rmsS / etendue:.2f} %")

    crit = [abs(dA), abs(biais) / Aps.mean()] + [abs(v["ecart_M2"]) for v in res["debits"].values()] + [rmsS / etendue]
    res["critere_5pc"] = bool(max(crit) < 0.05)
    lines += ["", f"Critere < 5 % : {'OK' if res['critere_5pc'] else 'NON'} (ecart max {100 * max(crit):.2f} %)"]
    od = os.path.join(rc, "diag"); os.makedirs(od, exist_ok=True)
    open(os.path.join(od, "imbrication.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(od, "imbrication.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
