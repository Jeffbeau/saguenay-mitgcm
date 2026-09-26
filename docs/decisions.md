# Configuration MITgcm Saguenay : décisions et état (2026-09-24)

Cahier des charges (document Claude) : https://claude.ai/code/artifact/79bedf7f-436f-4b37-a33e-e3fb70c27403

## Façon de travailler (préférences de l'utilisateur)
- **Toujours terminer une réponse de travail en indiquant la prochaine étape.**
- Approche **phénoménologique** : on cherche à identifier le phénomène (dynamique des tourbillons), pas à reproduire le détail. Forçages et conditions initiales approximatifs acceptés.
- Tests d'abord sur l'ordinateur de l'utilisateur. La décomposition MPI et l'étude des coûts viendront plus tard.
- L'utilisateur trouve la mise en place longue : aller droit au but, avec des commandes courtes à copier-coller.

## Environnement de calcul de l'utilisateur
- VirtualBox Linux (hôte « mitgcm », utilisateur jf) : 3 cœurs, 3,8 Go de RAM, pas de swap au départ. MITgcm dans ~/MITgcm (genmake2 demande -rootdir=$HOME/MITgcm ; pkg-config installé pour MPI). gfortran + OpenMPI.
- Dossier partagé VirtualBox : /media/sf_MITGCM (Additions invité installées, groupe vboxsf). Runs copiés dans ~/runs/.
- Le profil lite (5 M points) a été tué par manque de mémoire (OOM) sur 3 processus, d'où la création du profil mini.
- Pas d'Ubuntu/WSL sur le poste du travail.
- Python sur la VM : dans un venv (Ubuntu 24.04 refuse pip dans le Python système ; un NumPy 2 installé par pip à côté de rasterio d'apt casse l'import). setup_mitgcm.sh n'utilise plus pip hors venv.

## Décisions
- Imbrication unidirectionnelle hors ligne via OBCS : un parent à 100 m sur tout le fjord et un bout de l'estuaire, un enfant à 25 m sur la zone des seuils.
- Config A hydrostatique d'abord, puis Config B non hydrostatique sur l'enfant seulement. Config B en surface libre linéaire : checkpoint69k refuse nonlinFreeSurf ≠ 0 avec le solveur 3D (config_check.F).
- z* (nonlinFreeSurf=4, select_rStar=2), staggerTimeStep, GGL90 (KPP en sensibilité).
- Tourbillons définis par rapport à la moyenne de phase de marée (M2).
- MITgcm checkpoint69k.
- 3e seuil : on le garde dans le mini. L'enfant passe à 25 km si l'EKE du 3e seuil est sous 20 % de celle du 2e.
- Extraction parent → enfant : `imbrication/extraire_obcs.py` (voir « Imbrication »). Enfant démarré à un t0 multiple de 44 640 s (phase M2 nulle), enregistrements OBCS toutes les 930 s (les instantanés 3D du parent).
- Config B (non hydrostatique) : GGL90viscMax ≤ 0,2·dz_min²/Δt, car la viscosité verticale est explicite sur w.

## Pipeline grille + bathy + forçage
Emplacement : OneDrive …/Personnel/MITGCM/pipeline_grille/ (config.py, gridlib.py, s01..s05, run_all.py, templates/). Profils SAG_PROFILE : full (output/), lite (output_lite/), mini (output_mini/).
- Sources : NONNA-10 (26 tuiles, prioritaire) et NONNA-100 (4800N07000W + 4800N07100W), au zéro des cartes. NONNA-100 est biaisé vers le haut-fond, corrigé par z - (0,40 + 52,8·|∇z|). ZC -> NMM : Z0 entre 2,40 m (Tadoussac) et 2,69 m (Port-Alfred).
  - NONNA100_4800N07000W manque dans le dossier de l'utilisateur.
- full : parent 100 m 1200×532×60, enfant 25 m 1504×864×60 (grappe de calcul).
- lite : parent 200 m 600×268×32, tout le fjord. Trop lourd pour la VM de 4 Go.
- **mini** (tests sur la VM) : zone des seuils seule, 200 m, 188×108×32 (650 k points), nPx=2 (mpirun -np 2). OB : O 10 (coupe du fjord à 70,06°O), E 75, S 45. L'OB ouest impose Q = Q_riv − A_amont·dη/dt, avec A_amont = 213 km² (lu sur lite). E/S : (A_mini + A_amont)·dη/dt − Q_riv. Vérifié : η = ±1,60 m. Cols conservés (23,3 / 68,0 / 124,2 m). HMAX 270 m.
- **Enfant du mini** (2026-09-25, pour la Config B) : 50 m (rapport 4), centré sur le 2e seuil (col ~68 m au large d'Anse-de-Roche, Sacré-Cœur, 69,856°O 48,178°N) à la demande de l'utilisateur. Emprise UTM x 432,4-440,4 km, y 5331,6-5341,2 km (`utm` dans config.py), 160×192×32 (0,98 M points), col à 3,7-4,9 km des bords, marges avec le parent ≥ 29 cellules. Essai précédent (seuil d'entrée, x 445,2-455,6 km) abandonné. Emprise à vérifier sur fig_child.png. Δt 5 s ; éponge 1 km. Estimation : ~1,1 Go, 2 cycles NH ≈ 2 à 3 h sur 2 cœurs.
- **s01 découpé (mini).** `CROP` dans config.py : s01 ne lit que les tuiles qui touchent l'emprise du mini et découpe la mosaïque (rasterio merge, bounds) ; sortie dans `work_mini/` (les mosaïques de tout le fjord restent dans `work/`). Test synthétique : 2,5x moins de pixels NONNA-10, s01 3x plus rapide.
- **Enfant B réel lancé (2026-09-25, VM).** Parent mini 6 cycles : fin normale, CFL max u 0,29, v 0,24, w 0,71, η moyen du dernier cycle ±1,63 m. Extraction (t0 = 89 280 s, 2 cycles) : OB N 32 points (bras nord-sud, débit moyen +1199 m³/s) et E 24 points (bras est-ouest, −1200 m³/s) ; écart avant correction 7,8 % (N) et 0,9 % (E), correction ≤ 7,1 % de u rms ; bilan de volume 1,9 % du débit max ; surfaces mouillées 20,13 / 19,88 km². Démarrage : CFL w = 3,9 à t = 0 (vitesses interpolées), 0,12 à 1860 s ; CFL u 0,09.
- **Alignement.** s02 aligne l'enfant sur les faces du parent (gridlib.make_grid, `align`) : dans le mini, dx parent = 200 m ≠ SNAP = 100 m. Sans effet sur le profil full (dx parent = SNAP).
- Corrections bathy : voir README.

## Forçage parent (s05_forcing.py) — Config A
- Marée M2 pure, période modèle 44 640 s, amplitude 1,6 m. Pompage par vitesse barotrope normale uniforme aux OB. Rampe d'un cycle, 4 cycles simulés (passer à 6, voir Diagnostics).
- Rivière : 1200 m³/s (T 8 °C, S 0) à l'OB ouest pour full/lite. Pour mini, l'OB ouest reçoit le profil du fjord.
- T/S initiaux : profils analytiques types d'été (fjord et estuaire, transition à 69,68°O).
- useOBCSbalance = FALSE. Éponge de 3 km. Advection 33, viscC2smag 2,2, parois glissantes, frottement de fond 2,5e-3. dt 30 s (200 m).
- DIAGNOSTICS_SIZE : numDiags = 16·Nr (12·Nr était insuffisant : il en fallait 386 pour Nr=32). Avec les sorties recommandées : 26·Nr.
- **Premier run du mini (2026-09-25) : arrêt CALC_R_STAR à 22 320 s.** η moyen suivait la consigne (0,353 m contre 0,35 attendu), mais une poche collée à l'OB E (i = 186, j = 75-76, H 5-23 m, 3-4 voisins mouillés) s'empilait à +3,6 m au flot puis se vidait au jusant (colonne z* trop mince). Correctif s03 : bande de OB_EXTRUDE = 4 cellules le long des OB du parent où masque et H sont recopiés perpendiculairement au bord depuis l'intérieur, H ≥ H_OB_MIN = 10 m (dans l'éponge de 3 km). L'enfant n'est pas touché (sa bande vient du parent, loin des bords du parent).
- **s05 corrigé (2026-09-25).** (1) `periodicExternalForcing = .TRUE.` manquait : sans lui, obcs_fields_load.F ne lit les OB qu'une fois (pas de marée). (2) Enregistrements à (n − ½)·P, N = durée/P + 1, puis l'état à −P/2 (rivière seule) ; externForcingCycle = (N+2)·P ; rampe nulle pour t ≤ 0. L'ancien fichier (N_CYCLES + 1 cycles) rebouclait sur la rampe au-delà et mélangeait fin et début pour t < P/2. (3) data.diagnostics = mini_recommande (state3D 930 s avec WVEL, eta2D 360 s), numDiags = 26·Nr. (4) Options CPP/OBCS : templates/ s'il existe, sinon copie du source MITgcm (NONLIN_FRSURF, OBCS_SPONGE). (5) Aire amont : 213 km² par défaut si output_lite/ est absent. Mini : N_CYCLES = 6 (8928 pas). Test (bathy synthétique, 2 processus MPI) : 1000 pas en 5 min, η suit la rampe.

## Faits vérifiés dans le source checkpoint69k (2026-09-24)
- **OBCS.** L'enregistrement n correspond à t = (n − ½)·externForcingPeriod (get_periodic_interval.F). Pour t < P/2, MITgcm interpole entre le dernier enregistrement et le premier. Avec periodicExternalForcing=.FALSE., les OB sont lues une seule fois ; externForcingCycle=0 est refusé.
- **Marée OBCS native.** #define ALLOW_OBCS_TIDES ; useOBCStides, OBCS_tidalPeriod(1) ; fichiers OB[N,S,E,W]_[u,v]Tid[Am,Ph]File (amplitude en m/s, phase en s, u = A cos ω(t − Ph)). Il n'y a pas de rampe : on garde les fichiers prescrits.
- **Éponge OBCS.** Elle relaxe U, V, T et S. Urelaxobcs* s'applique aux OB E/O, Vrelaxobcs* aux OB N/S.
- **Diagnostics.** Les instantanés (frequency < 0) sont décalés par défaut de |freq|/2 : mettre timePhase(n) = 0. Une liste avec levels() réserve quand même Nr niveaux par champ dans numDiags.
- **Namelists.** Chaque ligne doit commencer par une espace, sinon le terminateur « & » n'est pas reconnu. Une ligne de plus de ~200 caractères est tronquée sans avertissement : utiliser N*valeur.
- **Faces OB.** La vitesse normale est imposée à u(OB_Iw+1) à l'ouest, u(OB_Ie) à l'est, v(OB_Js+1) au sud, v(OB_Jn) au nord ; T, S et w dans la cellule OB. Au coin de deux frontières, une face normale peut déboucher sur une cellule OB voisine : son débit n'entre pas dans l'intérieur.
- **η aux OB en z*.** Sans fichier OB*eta, obcs_calc.F met OB*eta = 0 et update_etah.F l'applique : η = 0 dans les cellules OB, facteur z* = 1 aux faces OB. Le débit imposé vaut donc exactement u·A (hypothèse de gen_testcase.py et s05). OB*etaFile n'est permis qu'avec nonlinFreeSurf ≠ 0 (obcs_check.F).
- **Bogue checkpoint69k.** obcs_apply_r_star.F, OB N et S : le facteur z* lit OBNeta(j) et OBSeta(j) (j = OB_Jn, OB_Js+1) au lieu de l'indice i. Avec des fichiers OB*eta, le facteur utilise la valeur d'un autre point (0 dans nos fichiers) : l'enfant A du cas test perdait 115 m³/s à l'OB sud (−5 cm/cycle). obcs_apply_surf_dr.F est correct. Correctif : copie corrigée dans code/ (gen_enfant.py le fait).
- **Non hydrostatique.** OB*wFile n'est permis qu'avec nonHydrostatic (obcs_check.F) ; w est imposé dans la cellule OB (obcs_apply_w.F). nonlinFreeSurf ≠ 0 est refusé avec le solveur 3D. Les facteurs implicitNHPress, implicSurfPress, implicDiv2DFlow doivent être non nuls (défaut 1). implicitViscosity ne s'applique pas à w (calc_gw.F) : la viscosité verticale de w, GGL90 compris, est explicite.
- **WVEL en z*.** C'est la vitesse à travers les surfaces r*. Vitesse vraie : w = w*(1 + η/H) + (1 − z/H)·∂η/∂t (z profondeur de l'interface). L'écart, de l'ordre de Aω ≈ 2e-4 m/s, n'est pas négligeable devant w.

## Sorties et diagnostics (décidé 2026-09-24)
- **Fréquences.** 3D toutes les 930 s (48 par cycle), 2D toutes les 360 s (124 par cycle), timePhase = 0. Règle : un multiple de Δt qui divise 44 640 s. Si Δt doit baisser, 15 s garde l'alignement (pas 20 s).
- **Fichier recommandé.** data.diagnostics.mini_recommande est dans le paquet saguenay_diag.
- **Source de l'EKE.** EKE, tensions et conversions viennent des instantanés 3D, par écart à la moyenne de phase. Les moyennes par cycle (UVELSQ…) mélangent marée et tourbillons ; elles servent au résiduel et au contrôle.
- **Champs obligatoires.** RHOAnoma pour la conversion barocline, ETAN pour l'épaisseur z* dans les flux.
- **Boîte à outils sagdiag** (numpy/scipy/matplotlib). Elle calcule, en 2 passes avec memmap :
  - EKE, MKE, P_h, P_v, B ;
  - les flux de sel moyens et tourbillonnaires aux coupes ;
  - la détection Okubo-Weiss avec suivi et fraction verrouillée en phase.

  Elle applique trois corrections : épaisseur z*, biais N/(N−1), bandes d'éponge exclues.
- **Durée d'analyse.** Simuler au moins 6 cycles et analyser avec --skip 2. Dans le cas test, le 1er cycle après la rampe a 2 à 3 fois plus d'EKE que les suivants.

## Cas test synthétique (2026-09-24)
- **Configuration.** Grille 94×54×32 à 400 m, même emprise et mêmes choix numériques que le mini, bathymétrie idéalisée (cols 124 / 68 / 23 m). Les 4 cycles tournent en une dizaine de minutes sur un cœur.
- **Validation.**
  - CFL max : u 0,29, w 0,39. Marée : M2 de η moyen = 1,55 m, phase +0,1° (convention OBCS validée).
  - Débit moyen aux trois seuils : 1186 à 1198 m³/s pour une rivière de 1200.
  - Flux de sel reconstruit : −1,6 à +0,3 % du total du modèle (USLTMASS).
  - La vorticité reproduit momVort3 (écart relatif 2e-7).
- **Résultats (idéalisés).** EKE/MKE = 0,06 sur le domaine et 0,34 dans l'estuaire. P_h = +0,79 MW, B = +1,2 MW. 28 tourbillons à 400 m, tous verrouillés en phase.
- **Alerte pour le mini à 200 m.** Les vitesses au seuil d'entrée atteignent ~3,8 m/s dans le cas test, donc le CFL u pourrait monter vers 0,57. Surveiller advcfl ; au besoin, passer à l'advection verticale implicite ou à Δt = 15 s.

## Imbrication parent → enfant (2026-09-24)
- **Outils.** `imbrication/extraire_obcs.py` (OB*.bin et conditions initiales), `imbrication/comparer.py` (cohérence), `imbrication/config_enfant.py` (namelists, code/, enfant.json : commun au cas test et au vrai enfant), `imbrication/enfant_depuis_pipeline.py` (config complète du vrai enfant depuis s02/s03 ; physique et tRef/sRef lus dans le data du parent), `scripts/run_imbrication.sh` (cas test de bout en bout).
- **Tests du chemin pipeline (nuage, sans NONNA).** s02 → s04 en profil mini sur une bathymétrie synthétique au format de s01 : col d'entrée 23,4 m à 50 m (23,1 m sur la grille 10 m), sections OB de l'enfant identiques à celles du parent. `enfant_depuis_pipeline.py` sur le cas test : même config que `gen_enfant.py --nh` ; enfant NH compilé en MPI (`genmake2 -mpi`, nPx = 2) et lancé avec `mpirun -np 2`.
- **Hypothèses.** Grilles cartésiennes uniformes, faces de l'enfant alignées sur celles du parent, même grille verticale. Parent lu par bandes (memmap) ; conditions initiales écrites niveau par niveau.
- **Méthode.** Interpolation bilinéaire (points secs remplis par le plus proche mouillé), linéaire en temps. Correction de flux par frontière et par enregistrement : vitesse uniforme ajoutée pour que le débit entrant égale celui du parent à travers la même ligne de faces, moins le remplissage des cellules OB. Seules les faces qui alimentent une cellule intérieure sont comptées et corrigées. En z*, fichiers OB*eta et facteur (1 + η_OB/H). Enfant NH : OB*w en vitesse vraie.
- **Démarrage.** T, S, U, V, η du parent à t0. Les vitesses interpolées ne sont pas à divergence nulle : en NH, CFL w = 2,1 au premier pas seulement, puis comme A.
- **Cas test (parent 400 m → enfant 200 m, 118×76×32, Δt 15 s, 2 cycles depuis t0 = 44 640 s, fenêtre = 2e cycle).** Extraction : correction ≤ 1,6 % de u rms, bilan de volume à 0,6 % du débit max, surfaces mouillées 108,88 / 108,96 km².

  | Critère | A (z*, hydrostatique) | B (NH, SL linéaire) |
  | --- | --- | --- |
  | Amplitude M2 de η | −0,20 % (phase −0,03°) | −0,23 % (phase −0,12°) |
  | Niveau moyen enfant − parent | −0,0 cm (−7,2 cm sans le correctif du bogue z*) | +0,1 cm |
  | Débit M2 seuil 2 / entrée | −0,25 % / −0,26 % | −0,26 % / −0,30 % |
  | Débit moyen seuil 2 / entrée (parent 1192 / 1182 m³/s) | 1194 / 1187 m³/s | 1195 / 1190 m³/s |
  | Salinité moyenne (rms / étendue) | 2,0 % | 2,0 % |
  | Coût (1 cœur, 2 cycles) | ~15 min | ~20 min |

- **Effet NH au seuil : `imbrication/seuil_nh.py`.** Talweg (chemin le plus profond entre les deux OB les plus éloignées), col = point le moins profond ; coupes w + isohalines à 8 phases, w rms et |w| max à ±1 km du col, rapport B/A. Cas test 200 m (col 23 m, dernier cycle) : w rms 10,4 (A) / 9,6 mm/s (B), |w| max 39 / 52 mm/s (rapport 1,33) : effet faible à 200 m, comme attendu.
- **Limites.** Le cas test à 200 m ne dit rien de la physique non hydrostatique (δ = H/L trop petit) : il valide la chaîne. Parent en z* et enfant B en surface libre linéaire : T, S sont passés niveau par niveau sans remappage vertical (décalage ≤ |η| près de la surface).
- **Vrai parent 100 m.** Les instantanés 3D complets à 930 s pèsent ~150 Mo par champ ; l'extraction ne lit que des bandes, mais il faudra les stocker (grappe) ou réduire la période de sortie à la durée de l'enfant.

## Constats bathy
- Cols (sous NMM) : 23 m à 69,647°O (seuil d'entrée) ; 68 m à 18,3 km ; 126 m à 32,4 km. Bassin intérieur ~270 m.

## Faits physiques (Belzile et al. 2016, JGR)
Longueur 110 km, largeur moyenne 2 km (1,1 km à l'embouchure). Seuils : 20 / 60 / 115 m. Bassin intérieur de 280 m. Marnage 4 m (6 m en vives-eaux). Q ≈ 1200 m³/s. Flux d'énergie de marée ~56 MW.
