#!/usr/bin/env python3
"""
Chaine complete de diagnostics tourbillonnaires sur un run MITgcm.

  python3 sagdiag_run.py RUNDIR [--config coupes.json] [--out RUNDIR/diag]

1. Controle : %MON (CFL, eta, KE) et ajustement M2/M4 de eta moyen.
2. 3D : moyenne de phase M2 (2 passes), EKE, MKE, productions P_h, P_v,
   flux de flottabilite B, flux de sel moyen / tourbillonnaire aux coupes.
3. Niveau 2D : Okubo-Weiss, detection, suivi, fraction verrouillee en phase.
Les prefixes de fichiers sont detectes automatiquement (voir --state3d, etc.).
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sagdiag as sd


def km(g, a):
    return a / 1e3 if g.cartesian else a


def landmap(ax, g):
    ax.pcolormesh(km(g, g.XG[0]), km(g, g.YG[:, 0]), np.where(g.maskC[0], np.nan, 1.0)[:-1, :-1],
                  cmap="Greys", vmin=0, vmax=2, shading="flat")
    ax.set_aspect("equal" if g.cartesian else 1 / np.cos(np.deg2rad(np.mean(g.YC))))


def pmap(ax, g, fld, cmap, vmin, vmax, title):
    x, y = km(g, g.XC[0]), km(g, g.YC[:, 0])
    im = ax.pcolormesh(x, y, np.where(g.maskC[0] & g.interior, fld, np.nan), cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="nearest")
    landmap(ax, g)
    ax.set_title(title, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out")
    ap.add_argument("--config", help="JSON : regions, coupes, period, skip_cycles")
    ap.add_argument("--period", type=float, default=44640.0)
    ap.add_argument("--skip", type=int, default=1, help="cycles ignores (rampe)")
    ap.add_argument("--t0", type=float, default=0.0, help="origine de phase (s)")
    ap.add_argument("--state3d"); ap.add_argument("--lev2d"); ap.add_argument("--eta"); ap.add_argument("--flux")
    ap.add_argument("--stdout", help="STDOUT.0000 ou sortie standard du run")
    ap.add_argument("--level", type=int, help="indice 0-based du niveau dans la liste 2D (defaut 1)")
    ap.add_argument("--min-diam", type=float, default=8.0, help="diametre min. en dx (defaut 8)")
    ap.add_argument("--ow", type=float, default=0.2)
    ap.add_argument("--ro-min", type=float, default=0.2)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, "diag")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "resume.txt"), "w")

    def log(s=""):
        print(s); logf.write(s + "\n"); logf.flush()

    cfg = json.load(open(a.config)) if a.config else {}
    T = float(cfg.get("period", a.period)); skip = int(cfg.get("skip_cycles", a.skip))
    regions = cfg.get("regions") or None
    sections = cfg.get("sections") or {}
    g = sd.Grid(a.run)
    s3, l2, et, fl = sd.guess_prefixes(sd.scan_run(a.run), g.nz)
    s3, l2, et, fl = a.state3d or s3, a.lev2d or l2, a.eta or et, a.flux or fl
    log(f"Run : {a.run}")
    log(f"Grille {g.nx}x{g.ny}x{g.nz}, dx~{g.dx_mean:.0f} m, dt={g.dt:g} s, rho0={g.rho0:g}, "
        f"eponges exclues des integrales : {g.nsponge} mailles")
    log(f"Fichiers : 3D={s3}  niveaux2D={l2}  eta={et}  flux={fl}")
    summary = {}

    # ------------------------------------------------------------ 1. controle
    std = a.stdout or next((os.path.join(a.run, f) for f in ("STDOUT.0000", "output.txt", "out.txt")
                            if os.path.exists(os.path.join(a.run, f))), None)
    mon = sd.read_monitor(std) if std else {}
    ef = sd.eta_fit(a.run, g, et, T, skip, a.t0) if et else None
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    if mon:
        th = mon["time_secondsf"] / 3600
        for k, c in (("dynstat_eta_max", "C3"), ("dynstat_eta_min", "C0"), ("dynstat_eta_mean", "k")):
            if k in mon:
                ax[0].plot(th, mon[k], c, lw=1, label=k.replace("dynstat_", ""))
        for k in ("advcfl_uvel_max", "advcfl_vvel_max", "advcfl_wvel_max"):
            if k in mon:
                ax[1].plot(th, mon[k], lw=1, label=k.replace("advcfl_", "CFL ").replace("_max", ""))
                summary["max_" + k] = float(np.nanmax(mon[k]))
        ax[1].axhline(0.5, color="r", ls=":", lw=1)
        for k in ("ke_mean", "ke_max"):
            if k in mon:
                ax[2].semilogy(th, mon[k], lw=1, label=k)
        bad = [k for k, v in mon.items() if not np.all(np.isfinite(v))]
        log(f"Controle : {len(th)} sorties %MON ; NaN : {'OUI ' + str(bad[:3]) if bad else 'non'}")
        log("CFL max : " + ", ".join(f"{k[7:12]} {summary['max_' + k]:.3f}" for k in
                                     ("advcfl_uvel_max", "advcfl_vvel_max", "advcfl_wvel_max") if "max_" + k in summary))
    if ef:
        ax[0].plot(ef["t"] / 3600, ef["eta"], "C2", lw=0.8, alpha=0.7, label="eta moyen (ETAN)")
        log(f"Maree (eta moyen, cycles analyses) : M2 = {ef['A2']:.3f} m, phase = {ef['phi2_deg']:+.1f} deg "
            f"(0 = sin(wt)), M4 = {ef['A4']:.3f} m, moyenne = {ef['mean']:+.3f} m")
        summary.update(eta_M2=ef["A2"], eta_phase=ef["phi2_deg"], eta_M4=ef["A4"])
    for x in ax:
        x.legend(fontsize=7, loc="upper left"); x.grid(alpha=0.3)
    ax[0].set_ylabel("eta (m)"); ax[1].set_ylabel("CFL"); ax[2].set_ylabel("KE (m2/s2)")
    ax[2].set_xlabel("temps (h)")
    fig.suptitle("Controle du run"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_controle.png"), dpi=110); plt.close(fig)

    # ------------------------------------------------------------ 2. analyse 3D
    if s3:
        eta_fn = sd.eta_interpolator(a.run, g, et) if et else None
        r = sd.phase_analysis_3d(a.run, g, s3, T, os.path.join(out, "phasemean"), skip, a.t0,
                                 regions, sections, log, eta_fn=eta_fn)
        ph = r["phase_deg"]
        np.savez_compressed(os.path.join(out, "energetique.npz"), phase_deg=ph,
                            **{f"{k}_vint": v for k, v in r["acc2"].items()},
                            **{f"{k}_3d_moy": v for k, v in r["acc3"].items()},
                            reg_eke_snap=r["reg_eke_snap"], sec_q=r["sec_q"],
                            sec_mean_salt=r["sec_mean_salt"], sec_eddy_salt=r["sec_eddy_salt"],
                            sec_total_salt=r["sec_total_salt"])
        nreg = len(r["reg_names"])
        fig, ax = plt.subplots(nreg, 1, figsize=(7, 2.2 * nreg + 0.6), sharex=True, squeeze=False)
        log(f"Energie integree (moyenne sur la maree ; moments d'ordre 2 corriges du biais N/(N-1), N = {r['ncyc']} cycles) :")
        for i, name in enumerate(r["reg_names"]):
            R = r["reg"][name]
            ax[i, 0].semilogy(ph, R["eke"] / 1e9, "C3", label="EKE (GJ)")
            ax[i, 0].semilogy(ph, R["mke"] / 1e9, "C0", label="MKE phase (GJ)")
            ax[i, 0].set_title(name, fontsize=9); ax[i, 0].grid(alpha=0.3)
            eke, mke = R["eke"].mean(), R["mke"].mean()
            log(f"  {name:12s} EKE {eke/1e9:8.3f} GJ  MKE {mke/1e9:8.3f} GJ  EKE/MKE {eke/max(mke,1e-30):.3f}"
                f" | P_h {R['Ph'].mean()/1e6:+7.3f} MW  P_v {R['Pv'].mean()/1e6:+7.3f} MW"
                f"  B {R['B'].mean()/1e6:+7.3f} MW")
            summary[f"reg_{name}"] = dict(EKE_GJ=eke / 1e9, MKE_GJ=mke / 1e9, Ph_MW=R["Ph"].mean() / 1e6,
                                          Pv_MW=R["Pv"].mean() / 1e6, B_MW=R["B"].mean() / 1e6)
        snap = r["reg_eke_snap"]
        if snap.size:
            cy = np.unique(snap[:, 2]).astype(int)
            ev = [snap[snap[:, 2] == c, 3].mean() / 1e9 for c in cy]
            log(f"EKE {r['reg_names'][0]} par cycle (GJ) : " + ", ".join(f"cycle {c} = {v:.2f}" for c, v in zip(cy, ev))
                + "  (un premier cycle nettement plus fort = derive de mise en route : augmenter --skip)")
            summary["eke_par_cycle_GJ"] = ev
        ax[0, 0].legend(fontsize=7)
        ax[-1, 0].set_xlabel("phase M2 (deg ; 0 = eta montant, 90 = pleine mer)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_energie_phase.png"), dpi=110); plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        rho = g.rho0
        E = rho * r["acc2"]["eke"].mean(0)
        im = pmap(ax[0, 0], g, E, "magma", 0, np.nanpercentile(E[g.maskC[0]], 99.5), "EKE integree (J/m2)")
        fig.colorbar(im, ax=ax[0, 0], shrink=0.8)
        for axx, key, lab in ((ax[0, 1], "Ph", "P_h cisaillement horizontal"), (ax[1, 0], "Pv", "P_v cisaillement vertical"),
                              (ax[1, 1], "B", "B flottabilite -g<rho'w'>/rho0")):
            F = rho * r["acc2"][key].mean(0)
            lim = np.nanpercentile(np.abs(F[g.maskC[0]]), 99) or 1.0
            im = pmap(axx, g, F, "RdBu_r", -lim, lim, lab + " (W/m2)")
            fig.colorbar(im, ax=axx, shrink=0.8)
        fig.suptitle(f"Moyennes sur la maree (positif = gain d'EKE ; eponges de {g.nsponge} mailles masquees)"); fig.tight_layout()
        fig.savefig(os.path.join(out, "fig4_cartes_energie.png"), dpi=110); plt.close(fig)

        if r["sections"]:
            chk = sd.flux_check(a.run, g, fl, sections, T, skip) if fl else {}
            m, e = r["sec_mean_salt"].mean(1), r["sec_eddy_salt"].mean(1)
            q = r["sec_q"].mean(1)
            fig, axx = plt.subplots(figsize=(7, 3.5))
            xx = np.arange(len(m))
            axx.bar(xx - 0.2, m / 1e3, 0.4, label="moyenne de phase")
            axx.bar(xx + 0.2, e / 1e3, 0.4, label="tourbillonnaire")
            if chk:
                axx.plot(xx, [chk.get(s, np.nan) / 1e3 for s in r["sections"]], "k*", ms=10,
                         label="total modele (USLTMASS)")
            axx.set_xticks(xx); axx.set_xticklabels(r["sections"]); axx.axhline(0, color="k", lw=0.5)
            axx.set_ylabel("flux de sel (10^3 psu m3/s)"); axx.legend(fontsize=8); axx.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_flux_sel.png"), dpi=110); plt.close(fig)
            log("Flux moyens sur la maree aux coupes (sel en psu m3/s, Q en m3/s ; positif vers +x ou +y"
                + ("; epaisseur z* incluse) :" if et else ") :"))
            for i, s in enumerate(r["sections"]):
                log(f"  {s:12s} moyenne-phase {m[i]:+10.1f}  tourbillon {e[i]:+10.1f}  "
                    f"total {m[i]+e[i]:+10.1f}  modele {chk.get(s, float('nan')):+10.1f}  (Q moyen {q[i]:+8.1f} m3/s)")
                summary[f"sec_{s}"] = dict(mean=m[i], eddy=e[i], model=chk.get(s, float("nan")), Q=q[i])

    # ------------------------------------------------------- 3. tourbillons 2D
    if l2:
        ed = sd.eddy_analysis_2d(a.run, g, l2, T, skip, a.level, a.t0, a.min_diam, a.ow, a.ro_min, log=log)
        rows = ed["rows"]
        sd.write_csv(os.path.join(out, "tourbillons.csv"), rows)
        dt2 = ed["dtout"]
        tracks = sd.track_table(rows, dt2)
        sd.write_csv(os.path.join(out, "trajectoires.csv"), tracks)
        nb, ncyc = ed["nb"], ed["ncyc"]
        sign = np.array([x["sign"] for x in rows]) if rows else np.zeros(0)
        bins = np.array([x["phase_bin"] for x in rows]) if rows else np.zeros(0, int)
        lock = np.array([x["locked"] for x in rows]) if rows else np.zeros(0)
        log(f"Tourbillons ({ed['level_depth']:.1f} m) : {len(rows)} detections, {len(tracks)} trajectoires, "
            f"{ncyc} cycles ; cycloniques {int((sign > 0).sum())}, anticycloniques {int((sign < 0).sum())}")
        if rows:
            dur = np.array([t_["duree_s"] for t_ in tracks]) / 3600
            log(f"  duree de vie mediane {np.median(dur):.2f} h (max {dur.max():.1f} h) ; "
                f"r_eq median {np.median([x['r_eq'] for x in rows]):.0f} m ; "
                f"part verrouillee en phase (>0,5) : {np.nanmean(lock > 0.5):.0%}")
            summary["eddies"] = dict(n=len(rows), tracks=len(tracks), lifetime_h_median=float(np.median(dur)),
                                     locked_frac=float(np.nanmean(lock > 0.5)))
        fig, ax = plt.subplots(2, 4, figsize=(13, 5.8))
        sel_b = [int(round(k * nb / 8)) % nb for k in range(8)]
        vals = [ed["last_cycle_ro"][k] for k in sel_b if k in ed["last_cycle_ro"]]
        lim = max(1.0, float(np.nanpercentile(np.abs(np.concatenate([v[np.isfinite(v)] for v in vals])), 98))) if vals else 1
        last = max((x["cycle"] for x in rows), default=0)
        for axx, k in zip(ax.ravel(), sel_b):
            if k not in ed["last_cycle_ro"]:
                continue
            im = pmap(axx, g, ed["last_cycle_ro"][k], "RdBu_r", -lim, lim, f"phase {ed['phase_deg'][k]:.0f} deg")
            for x in rows:
                if x["cycle"] == last and x["phase_bin"] == k:
                    axx.add_patch(plt.Circle((x["x"] / 1e3, x["y"] / 1e3) if g.cartesian else (x["x"], x["y"]),
                                             x["r_eq"] / 1e3 if g.cartesian else x["r_eq"] / 111e3,
                                             fill=False, color="k", lw=0.8))
        fig.colorbar(im, ax=ax, shrink=0.6, label="zeta/f")
        fig.suptitle(f"Vorticite relative / f a {ed['level_depth']:.0f} m, dernier cycle (cercles = detections)")
        fig.savefig(os.path.join(out, "fig2_vorticite_cycle.png"), dpi=110); plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        pd = ed["phase_deg"]
        for s_, c, lab in ((1, "C3", "cycloniques"), (-1, "C0", "anticycloniques")):
            ax[0, 0].plot(pd, np.bincount(bins[sign == s_], minlength=nb) / max(ncyc, 1), c, label=lab)
        ax[0, 0].set_xlabel("phase M2 (deg)"); ax[0, 0].set_ylabel("detections par cycle")
        ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)
        landmap(ax[0, 1], g)
        cs = ax[0, 1].contour(km(g, g.XC[0]), km(g, g.YC[:, 0]), np.where(g.maskC[0], g.Depth, np.nan),
                              levels=[25, 50, 100, 200], colors="0.6", linewidths=0.5)
        t_first = min((x["t"] for x in rows), default=0)
        for t_ in tracks:
            if t_["n"] >= 2 and t_["t0"] > t_first:
                ax[0, 1].plot(km(g, t_["x0"]), km(g, t_["y0"]), "o", ms=3,
                              color="C3" if t_["sign"] > 0 else "C0", alpha=0.6)
        ax[0, 1].set_title("sites de formation (1re detection)", fontsize=9)
        if rows:
            ax[1, 0].hist([x["r_eq"] / 1e3 for x in rows], bins=20, color="0.5")
            ax[1, 1].hist(np.clip(lock[np.isfinite(lock)], -0.5, 1.5), bins=20, color="0.5")
        ax[1, 0].set_xlabel("rayon equivalent (km)")
        ax[1, 1].set_xlabel("fraction verrouillee en phase (zeta moyen de phase / zeta)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_tourbillons.png"), dpi=110); plt.close(fig)

    json.dump(summary, open(os.path.join(out, "resume.json"), "w"), indent=1, default=float)
    log(f"Sorties dans {out}")


if __name__ == "__main__":
    main()
