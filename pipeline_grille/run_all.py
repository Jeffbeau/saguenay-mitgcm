"""run_all.py — enchaîne tout le pipeline : python run_all.py
(NONNA_DIR=... pour pointer ailleurs que ../bathy/Bathymetry ;
 SAG_PROFILE=lite pour le parent 200 m / 32 niveaux de test)"""
import s01_mosaic, s02_grids, s03_bathy, s04_verify, s05_forcing
for m in (s01_mosaic, s02_grids, s03_bathy, s04_verify, s05_forcing):
    m.main()
