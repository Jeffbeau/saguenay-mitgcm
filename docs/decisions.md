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

## Décisions
- Imbrication unidirectionnelle hors ligne via OBCS : un parent à 100 m sur tout le fjord et un bout de l'estuaire, un enfant à 25 m sur la zone des seuils.
- Config A hydrostatique d'abord, puis Config B non hydrostatique sur l'enfant seulement. Config B en surface libre linéaire : checkpoint69k refuse nonlinFreeSurf ≠ 0 avec le solveur 3D (config_check.F).
- z* (nonlinFreeSurf=4, select_rStar=2), staggerTimeStep, GGL90 (KPP en sensibilité).
- Tourbillons définis par rapport à la moyenne de phase de marée (M2).
- MITgcm checkpoint69k.
- 3e seuil : on le garde dans le mini. L'enfant passe à 25 km si l'EKE du 3e seuil est sous 20 % de celle du 2e.

## Pipeline grille + bathy + forçage
Emplacement : OneDrive …/Personnel/MITGCM/pipeline_grille/ (config.py, gridlib.py, s01..s05, run_all.py, templates/). Profils SAG_PROFILE : full (output/), lite (output_lite/), mini (output_mini/).
- Sources : NONNA-10 (26 tuiles, prioritaire) et NONNA-100 (4800N07000W + 4800N07100W), au zéro des cartes. NONNA-100 est biaisé vers le haut-fond, corrigé par z - (0,40 + 52,8·|∇z|). ZC -> NMM : Z0 entre 2,40 m (Tadoussac) et 2,69 m (Port-Alfred).
  - NONNA100_4800N07000W manque dans le dossier de l'utilisateur.
- full : parent 100 m 1200×532×60, enfant 25 m 1504×864×60 (grappe de calcul).
- lite : parent 200 m 600×268×32, tout le fjord. Trop lourd pour la VM de 4 Go.
- **mini** (tests sur la VM) : zone des seuils seule, 200 m, 188×108×32 (650 k points), nPx=2 (mpirun -np 2). OB : O 10 (coupe du fjord à 70,06°O), E 75, S 45. L'OB ouest impose Q = Q_riv − A_amont·dη/dt, avec A_amont = 213 km² (lu sur lite). E/S : (A_mini + A_amont)·dη/dt − Q_riv. Vérifié : η = ±1,60 m. Cols conservés (23,3 / 68,0 / 124,2 m). HMAX 270 m.
- Corrections bathy : voir README.

## Forçage parent (s05_forcing.py) — Config A
- Marée M2 pure, période modèle 44 640 s, amplitude 1,6 m. Pompage par vitesse barotrope normale uniforme aux OB. Rampe d'un cycle, 4 cycles simulés (passer à 6, voir Diagnostics).
- Rivière : 1200 m³/s (T 8 °C, S 0) à l'OB ouest pour full/lite. Pour mini, l'OB ouest reçoit le profil du fjord.
- T/S initiaux : profils analytiques types d'été (fjord et estuaire, transition à 69,68°O).
- useOBCSbalance = FALSE. Éponge de 3 km. Advection 33, viscC2smag 2,2, parois glissantes, frottement de fond 2,5e-3. dt 30 s (200 m).
- DIAGNOSTICS_SIZE : numDiags = 16·Nr (12·Nr était insuffisant : il en fallait 386 pour Nr=32). Avec les sorties recommandées : 26·Nr.
- **À vérifier dans s05** : l'enregistrement n doit tomber à t = (n − ½)·P. Il faut 2 enregistrements finaux, la suite logique puis l'état avant départ (rivière seule), et externForcingCycle = (N+2)·P, au moins la durée du run + P/2.

## Faits vérifiés dans le source checkpoint69k (2026-09-24)
- **OBCS.** L'enregistrement n correspond à t = (n − ½)·externForcingPeriod (get_periodic_interval.F). Pour t < P/2, MITgcm interpole entre le dernier enregistrement et le premier. Avec periodicExternalForcing=.FALSE., les OB sont lues une seule fois ; externForcingCycle=0 est refusé.
- **Marée OBCS native.** #define ALLOW_OBCS_TIDES ; useOBCStides, OBCS_tidalPeriod(1) ; fichiers OB[N,S,E,W]_[u,v]Tid[Am,Ph]File (amplitude en m/s, phase en s, u = A cos ω(t − Ph)). Il n'y a pas de rampe : on garde les fichiers prescrits.
- **Éponge OBCS.** Elle relaxe U, V, T et S. Urelaxobcs* s'applique aux OB E/O, Vrelaxobcs* aux OB N/S.
- **Diagnostics.** Les instantanés (frequency < 0) sont décalés par défaut de |freq|/2 : mettre timePhase(n) = 0. Une liste avec levels() réserve quand même Nr niveaux par champ dans numDiags.
- **Namelists.** Chaque ligne doit commencer par une espace, sinon le terminateur « & » n'est pas reconnu. Une ligne de plus de ~200 caractères est tronquée sans avertissement : utiliser N*valeur.

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

## Constats bathy
- Cols (sous NMM) : 23 m à 69,647°O (seuil d'entrée) ; 68 m à 18,3 km ; 126 m à 32,4 km. Bassin intérieur ~270 m.

## Faits physiques (Belzile et al. 2016, JGR)
Longueur 110 km, largeur moyenne 2 km (1,1 km à l'embouchure). Seuils : 20 / 60 / 115 m. Bassin intérieur de 280 m. Marnage 4 m (6 m en vives-eaux). Q ≈ 1200 m³/s. Flux d'énergie de marée ~56 MW.
