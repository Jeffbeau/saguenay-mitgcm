"""
Geometrie analytique du cas test synthetique (partagee par le parent et l'enfant).

Coordonnees en m dans le repere du parent : x vers l'est depuis le bord ouest,
y vers le nord depuis le bord sud, profondeurs positives vers le bas.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

LX, LY = 37600.0, 21600.0
HMAX = 270.0

# profil du talweg : noeuds (km, m) ; PCHIP conserve exactement les cols
XK = np.array([0, 3, 5.0, 7, 13, 16.0, 19, 26, 29.0, 31, 40])
HK = np.array([250, 250, 124, 200, 190, 68, 120, 110, 23, 60, 150])
h_axis = PchipInterpolator(XK * 1e3, HK)


def y_axis(xx):
    return 15e3 - 3e3 * (np.clip(xx, 0, 30e3) / 30e3) ** 2


def half_width(xx):
    return np.interp(xx, [0, 20e3, 27e3, 30e3, 40e3], [1000, 1000, 800, 600, 600])


def coord_estuaire(X, Y):
    """s < 0 : estuaire, au sud-est d'une rive nord a 40 deg passant par l'embouchure."""
    th = np.deg2rad(40.0)
    return -np.sin(th) * (X - 30e3) + np.cos(th) * (Y - 12e3)


def niveaux(nr):
    """Epaisseurs dz (m), faces zf et centres zc (positifs vers le bas)."""
    dz = np.geomspace(1.5, 20.0, nr)
    dz *= 280.0 / dz.sum()
    dz = np.round(dz, 3)
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    return dz, zf, 0.5 * (zf[:-1] + zf[1:])


def profondeur(X, Y):
    """Profondeur analytique H(x, y) (m > 0, terre = 0) aux points (X, Y)."""
    d = np.abs(Y - y_axis(X))
    hw = half_width(X)
    H_f = np.where((d < hw) & (X <= 31.5e3),
                   h_axis(X) * np.clip(1 - (d / hw) ** 2, 0, 1) ** 0.4, 0.0)
    s = coord_estuaire(X, Y)
    doff = np.clip(-s, 0, None)
    H_e = np.where(s < 0, 15 + 255 * np.tanh(doff / 3e3) ** 1.5, 0.0)
    H = np.maximum(H_f, H_e)
    return np.where(H < 10.0, 0.0, np.minimum(H, HMAX))


def hfac_c(H, dz, hfacmin, hfacmindr):
    """hFacC comme MITgcm (coordonnee z, cellules partielles)."""
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    hf = np.zeros((len(dz),) + np.shape(H))
    for k in range(len(dz)):
        mn = max(hfacmin, min(hfacmindr / dz[k], 1.0))
        t = np.clip((H - zf[k]) / dz[k], 0.0, 1.0)
        t = np.where(t < mn, np.where(t < 0.5 * mn, 0.0, mn), t)
        hf[k] = t
    return hf


def profils(zc):
    """Profils types (T_f, S_f) du fjord et (T_e, S_e) de l'estuaire."""
    S_f = 30.5 - 22.0 * np.exp(-zc / 4.0)
    T_f = 1.5 + 12.0 * np.exp(-zc / 6.0)
    S_e = 33.8 - 7.5 * np.exp(-zc / 25.0)
    T_e = 4.3 - 3.8 * np.exp(-((zc - 70.0) / 45.0) ** 2) + 2.0 * np.exp(-zc / 10.0)
    return T_f, S_f, T_e, S_e


def ts_initiaux(zc, X, Y):
    """T, S initiaux 3D (nr, ny, nx) : fjord -> estuaire par une transition en tanh."""
    T_f, S_f, T_e, S_e = profils(zc)
    w_e = 0.5 * (1 - np.tanh(coord_estuaire(X, Y) / 1.5e3))
    w_e[X < 27e3] = 0.0
    T3 = (1 - w_e)[None] * T_f[:, None, None] + w_e[None] * T_e[:, None, None]
    S3 = (1 - w_e)[None] * S_f[:, None, None] + w_e[None] * S_e[:, None, None]
    return T3, S3
