"""
config.py — Paramètres uniques du pipeline grille + bathymétrie (Saguenay, MITgcm).

Tout ce qui est « décision » vit ici ; les scripts 01..04 n'ont aucun nombre magique.
Coordonnées horizontales : UTM 19N (EPSG:32619), en mètres. Grilles cartésiennes
(usingCartesianGrid) dont les faces sont alignées sur des multiples de 100 m, pour que
chaque face du parent (100 m) coïncide exactement avec une face de l'enfant (25 m).
"""
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
NONNA_DIR = Path(__import__("os").environ.get(
    "NONNA_DIR", HERE.parent / "bathy" / "Bathymetry"))       # tuiles NONNA10_*.tiff
# Profil de configuration : SAG_PROFILE=full (défaut), lite (parent 200 m entier) ou
# mini (zone des seuils seule, 200 m : tests rapides sur petite machine)
PROFILE = __import__("os").environ.get("SAG_PROFILE", "full")
WORK = HERE / "work"          # intermédiaires partagés (mosaïques NONNA traitées)
CROP = None                    # (lon0, lon1, lat0, lat1) : s01 ne lit que cette emprise (None = tout)
OUT = HERE / ("output" if PROFILE == "full" else f"output_{PROFILE}")   # livrables MITgcm

# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
CRS_SRC = "EPSG:4326"
CRS_MODEL = "EPSG:32619"      # UTM 19N — facteur d'échelle ~0.9997 à 70°O, négligeable

# ---------------------------------------------------------------------------
# Référence verticale
# NONNA est au zéro des cartes (ZC), valeurs négatives = sous ZC, positives = découvrant.
# Le modèle veut la profondeur sous le niveau moyen (NMM). Z0 = hauteur du NMM au-dessus
# du ZC (m). Profondeur modèle H = -z_ZC + Z0.
# Tadoussac (SHC 03425) : NME = 2,40 m / ZC ; PMSGM 5,38 ; BMIGM -0,30 ; PHMA 5,44 ; PBMA -0,40
#   -> niveau extrême au-dessus du NMM +3,04 m, sous le NMM -2,80 m.
# Z0 peut varier le long du fjord (marnage plus fort vers Chicoutimi) : liste (lon, Z0)
# interpolée linéairement en longitude. Une seule station pour l'instant -> Z0 constant.
# ---------------------------------------------------------------------------
# Port-Alfred (Baie des Ha! Ha!) : NME = 2,69 m / ZC. Interpolation linéaire en longitude,
# constante hors de l'intervalle (bras nord / Chicoutimi = 2,69 m faute de station).
Z0_STATIONS = [(-69.72, 2.40), (-70.87, 2.69)]  # (lon, Z0 [m]) : Tadoussac, Port-Alfred

# ---------------------------------------------------------------------------
# Traitement 10 m (avant passage aux grilles)
# ---------------------------------------------------------------------------
# Deux produits NONNA, traités séparément puis combinés cellule par cellule :
#   n10  (≈10 m) prioritaire là où il couvre toute la cellule ; n100 (≈100 m) ailleurs.
#   close   : fermeture morphologique (pixels) pour sceller les micro-trous
#   min_isl : un trou fermé plus petit que ça (pixels) est toujours de l'eau
PRODUCTS = {
    "n10":  dict(glob="NONNA10_*.tif*",  close=2, min_isl=50),
    "n100": dict(glob="NONNA100_*.tif*", close=0, min_isl=3),
}
ISLAND_RING_ZMAX = -1.0        # m (ZC) : un trou fermé dont le pourtour atteint au moins
                               #   cette cote (estran, ~ZC) = ÎLE ; sinon trou de sondage (eau).
                               #   (Ex. Île Saint-Louis : pourtour max 0.0 m ZC -> île)
# NONNA-100 = cote la plus haute de la cellule (biais vers le haut-fond). Correction
# a + b|grad z| ajustée automatiquement sur le recouvrement n10/n100 ; valeurs par défaut
# (ajustement du 2026-09-23, R2 = 0,71) si NONNA-10 absent.
N100_BIAS_DEFAULT = dict(a=0.50, b=53.7, n=0, bias_mean=7.3, rms_before=13.4, rms_after=6.0)
N100_BIAS_CAP = 50.0           # m, correction maximale
EDGE_ERODE = 3                 # pixels de couverture retirés au bord extérieur de chaque mosaïque
N10_FULL_COVER = 0.99          # n10 utilisé si sa couverture de la cellule >= 99 %

# ---------------------------------------------------------------------------
# Grilles horizontales (bornes en UTM 19N, multiples de 100 m)
# PARENT : tout le fjord jusqu à la fin des levés à Chicoutimi (71,08°O : OB O = rivière, débit),
#   Baie des Ha! Ha! incluse
#   + l'estuaire jusqu'à 69,45°O / 48,0°N (limite sud de NONNA-100).
# ENFANT : zone des seuils, ~7 km à l'est du seuil d'entrée -> amont du col de 126 m.
# ---------------------------------------------------------------------------
GRIDS = {
    # bbox = (lon_min, lon_max, lat_min, lat_max) ; None = toute la couverture NONNA
    "parent": dict(dx=100.0, bbox=(-71.080, -69.450, 48.000, 48.500)),
    # Enfant : estuaire à l est du seuil d entrée (~23 m, 69,65°O) -> amont du col de ~126 m
    # (Baie Éternité / Île Saint-Louis). OB est = estuaire, OB ouest = coupe du fjord.
    "child":  dict(dx=25.0, bbox=(-70.060, -69.550, 48.080, 48.280)),
}
# Bords FERMÉS de force (mur). Parent : le bord O coupe la rivière à 71,0°O -> laissé OUVERT
# (entrée du débit Q ~ 1200 m3/s par OBCS). Mettre ["W"] pour un mur.
CLOSED_EDGES = {"parent": [], "child": []}
NEST_MARGIN = 10               # l'enfant doit être à >= 10 cellules parent des bords du parent
NEST_RATIO = 4                 # parent.dx / child.dx
SNAP = 100.0                   # toutes les bornes sont des multiples de SNAP (= dx parent)
PAD_MULT = 4                   # Nx, Ny arrondis au multiple de PAD_MULT (tuiles MPI)

# ---------------------------------------------------------------------------
# Grille verticale (commune parent/enfant, indispensable pour l'OBCS hors ligne)
# NSURF niveaux de DZ_SURF m, puis étirement géométrique de raison r (calculée) jusqu'à HMAX.
# ---------------------------------------------------------------------------
NR = 60
DZ_SURF = 1.0
NSURF = 8
HMAX = None                    # m ; None = auto : profondeur max du parent + 5 m, arrondie à 10 m
                               #   (le chenal Laurentien dans l'estuaire dépasse 300 m)
R_MAX_ALLOWED = 1.10           # contrôle : dz(k+1)/dz(k) <= 1.10

# ---------------------------------------------------------------------------
# Correction de bathy sur la grille modèle
# ---------------------------------------------------------------------------
WET_FRAC_MIN = 0.5             # une cellule est mouillée si >= 50 % de sa surface est de l'eau
H_DRY = 2.0                    # m sous NMM : moins profond que ça -> terre. ~ basse mer moyenne
                               #   (marnage moyen 4 m) : l estran découvrant devient terre (pas de W&D)
H_MIN = 5.0                    # m : profondeur min imposée aux cellules mouillées
                               #   (z* : (H - a)/H > hFacInf avec a ~ 3 m en VE, hFacInf=0.2)
MIN_COMPONENT = 50             # enfant : zone d'eau isolée plus petite (cellules) -> terre
NPX = 1                        # nPx écrit dans SIZE.h (décomposition MPI en x), si Nx divisible
W_OB = "river"                 # frontière ouest : "river" (débit Q_RIVER) ou "fjord" (coupe du fjord :
                               #   flux de marée du fjord amont + rivière)
UPSTREAM_AREA_FROM = None      # profil dont on lit l'aire du fjord en amont de la coupe ouest
OB_MIN_SEG = 4                 # segment de frontière ouverte plus court (cellules) -> fermé
MAX_LAND_NEIGH = 3             # cellule mouillée avec >= 3 voisins terre (4-conn) -> terre (itéré)
HFAC_MIN = 0.2                 # identiques à data/PARM01
HFAC_MIN_DR = 0.5
APPLY_HFAC_ROUNDING = True     # écrire la profondeur « effective » que MITgcm utilisera
SMOOTH = False                 # en coordonnée z + partial cells, pas besoin de lisser (pas de
                               #   sigma). Laisser False sauf test de sensibilité.

# Cohérence parent/enfant aux frontières ouvertes de l'enfant (en cellules PARENT)
NEST_COPY = 2                  # bande où l'enfant = copie exacte du parent (masque + H)
NEST_BLEND = 2                 # bande suivante : mélange linéaire parent -> enfant

# ---------------------------------------------------------------------------
# Repères physiques pour la vérification (Belzile et al. 2016)
# ---------------------------------------------------------------------------
MOUTH_LONLAT = (-69.715, 48.128)    # embouchure à Tadoussac (origine des distances)
EXPECTED_SILLS = [(0.0, 20.0), (18.0, 60.0), (32.0, 115.0)]   # (km depuis Tadoussac, m)
EXPECTED_BASIN = 280.0

# Physique utile pour data (écrit dans les extraits)
F0 = 2 * 7.2921e-5 * np.sin(np.radians(48.2))


# ---------------------------------------------------------------------------
# Forçage et conditions initiales (s05) — approche phénoménologique
# ---------------------------------------------------------------------------
T_M2 = 44640.0                 # s : 12,40 h (M2 vraie 44 714 s ; arrondi pour que T/24 = 1860 s
                               #   soit un multiple entier du pas de temps)
REC_PER_CYCLE = 24             # enregistrements OBCS par cycle (1860 s)
N_CYCLES = 4                   # durée du run (cycles M2), dont RAMP_CYCLES de mise en route
RAMP_CYCLES = 1
TIDE_AMP = 1.6                 # m : amplitude M2 visée dans le domaine (marée moyenne ;
                               #   (PMM - BMM)/2 = (4,06 - 0,79)/2 à Tadoussac). VE : ~2,8 m.
Q_RIVER = 1200.0               # m3/s, Saguenay à Chicoutimi (constant)
T_RIVER, S_RIVER = 8.0, 0.0
DT = {200.0: 30.0, 100.0: 20.0, 50.0: 5.0, 25.0: 5.0}   # s, par résolution (diviseurs de 1860 s ;
                               #   l'enfant forcé par le parent : diviseur de 930 et 360 s)
SPONGE_M = 3000.0              # m : épaisseur de l'éponge OBCS
# Profils types d'été (analytiques, ordres de grandeur ; z positif vers le bas, m)
#   fjord    : couche saumâtre ~5 m (S~8, T~14 °C), halocline à 6 m, eau profonde S 30,8 / T 1,5 °C
#   estuaire : tête du chenal Laurentien, S 26,5 en surface -> 34,4 au fond ; CIL ~0,7 °C vers 60 m ;
#              eau profonde chaude ~5 °C
PROFILE_BLEND_LON = -69.68     # passage fjord -> estuaire (entre la pointe de Tadoussac et le seuil)
PROFILE_BLEND_DLON = 0.015     # largeur de la transition (°, ~1 km)


def profile_fjord(z):
    S = 8.0 + 19.5 * 0.5 * (1 + np.tanh((z - 6.0) / 2.5)) + 3.3 * (1 - np.exp(-z / 40.0))
    T = 1.5 + 12.5 * 0.5 * (1 - np.tanh((z - 6.0) / 3.0)) - 0.8 * np.exp(-((z - 50.0) / 25.0) ** 2)
    return T, S


def profile_estuary(z):
    S = 26.5 + 5.5 * (1 - np.exp(-z / 30.0)) + 2.4 * 0.5 * (1 + np.tanh((z - 150.0) / 50.0))
    T = 0.5 + 6.0 * np.exp(-z / 15.0) + 4.6 * 0.5 * (1 + np.tanh((z - 150.0) / 50.0))
    return T, S


# ---------------------------------------------------------------------------
# Profil « lite » : parent seul à 200 m, 32 niveaux (tests sur ordinateur personnel).
# Même domaine, mêmes corrections ; pas d'enfant.
# ---------------------------------------------------------------------------
if PROFILE == "lite":
    GRIDS = {"parent": dict(dx=200.0, bbox=GRIDS["parent"]["bbox"])}
    NR, NSURF, DZ_SURF = 32, 4, 2.0
    R_MAX_ALLOWED = 1.12
    OB_MIN_SEG = 2             # à 200 m la rivière à Chicoutimi ne fait que 2-3 cellules

# ---------------------------------------------------------------------------
# Profil « mini » : zone des seuils seule (estuaire -> Baie Éternité) à 200 m, 32 niveaux.
# La frontière ouest coupe le fjord : on y impose le flux de marée qui remplit/vide tout le
# fjord en amont (aire lue sur le parent « lite ») + le débit de la rivière.
# ~8x plus léger que « lite » : tient dans < 1 Go, un cycle M2 en quelques minutes.
# ---------------------------------------------------------------------------
if PROFILE == "mini":
    # Enfant 50 m (rapport 4) sur le seuil d'entrée (22,9 m, 69,647°O 48,125°N), pour la Config B
    # non hydrostatique. Emprise en UTM (x0, x1, y0, y1), sur les faces du parent (200 m) :
    #   O 445,2 km : coupe nette du fjord dans le bassin extérieur ;
    #   S 5328,0 km : seuil à ~1,9 km du bord (le bord sud du mini est à 5325,8 km ; l'enfant
    #     reste à >= 10 cellules parent, mais dans l'éponge du parent, qui s'arrête à 5328,8 km) ;
    #   E 455,6 km : aval du seuil vers le chenal Laurentien (éponge du parent à partir de 456,0).
    # 208 x 144 x 32 (0,96 M points) : 2 cycles NH ~ 2-3 h sur 2 cœurs, ~1,1 Go (estimation cas test).
    GRIDS = {"parent": dict(dx=200.0, bbox=(-70.060, -69.550, 48.080, 48.280)),
             "child": dict(dx=50.0, utm=(445200.0, 455600.0, 5328000.0, 5335200.0))}
    NR, NSURF, DZ_SURF = 32, 4, 2.0
    R_MAX_ALLOWED = 1.12
    OB_MIN_SEG = 2
    NPX = 2                    # 2 processus : laisse un cœur libre sur une VM à 3 cœurs
    # s01 découpe les mosaïques à l'emprise du mini (+ marge) : ~10x moins de pixels NONNA-10.
    # Mosaïques découpées dans work_mini/ pour ne pas écraser celles de tout le fjord (work/).
    CROP = (-70.080, -69.530, 48.070, 48.290)
    WORK = HERE / "work_mini"
    W_OB = "fjord"
    UPSTREAM_AREA_FROM = "lite"
