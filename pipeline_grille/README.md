# Pipeline grille + bathymétrie — Saguenay (MITgcm)

Tout refait de zéro. `python run_all.py` (Python ≥ 3.10 : numpy, scipy, rasterio, pyproj,
matplotlib, scikit-image). Tous les paramètres sont dans `config.py`.

| étape | script | rôle |
|---|---|---|
| 1 | `s01_mosaic.py` | NONNA-10 et NONNA-100 : fusion, correction du biais haut-fond de NONNA-100, masque d'eau (trous / îles), comblement, ZC → NMM |
| 2 | `s02_grids.py` | grilles UTM 19N parent 100 m / enfant 25 m alignées (rapport 4), grille verticale commune |
| 3 | `s03_bathy.py` | agrégation par cellule, corrections, cohérence parent→enfant aux OB, arrondi hFac, écriture |
| 4 | `s04_verify.py` | relecture binaire, profondeur des seuils à 10/25/100 m, sections aux OB, figures |
| 5 | `s05_forcing.py` | parent : marée M2 (pompage par l'estuaire), rivière, T/S initiaux types, namelists et code MITgcm → `parent/run/` |

Profil `SAG_PROFILE=mini` : parent 200 m sur la zone des seuils + enfant 50 m sur le seuil d'entrée
(`output_mini/`, emprise dans `config.py`). L'enfant sert à la Config B non hydrostatique ; voir le README
principal (« Enfant 50 m du mini »).

Profils : `SAG_PROFILE=full` (défaut ; parent 100 m + enfant 25 m, 60 niveaux → `output/`) ou
`SAG_PROFILE=lite` (parent 200 m, 32 niveaux, pour tests sur ordinateur personnel → `output_lite/`).

Sorties dans `output/parent/` et `output/child/` : `bathy.bin` (real*8 big-endian, −H, terre = 0),
`SIZE.h` (1 tuile, à redécouper plus tard), `data_PARM04.txt` (delX/delY/delR), `data.obcs_OB.txt`
(indices OB détectés), `grid.npz` (coordonnées, masques), `log.json`. Rapport : `output/verification.md`.

Données attendues dans `../bathy/Bathymetry/` (sous-dossiers acceptés) : `NONNA10_*.tiff` (prioritaires, zone des seuils) et
`NONNA100_*.tiff` (tout le fjord + estuaire). Sans NONNA-10, le pipeline tourne sur NONNA-100 seul
(correction de biais par défaut) mais l'enfant 25 m perd sa résolution réelle.

Z0 = 2,40 m (NME Tadoussac, SHC 03425). À compléter : stations amont (marnage plus fort vers Chicoutimi).
