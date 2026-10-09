# Projet MITgcm Saguenay

Configuration MITgcm (checkpoint69k) du fjord du Saguenay et diagnostics tourbillonnaires. Les règles de travail sont dans `CLAUDE.md`, les décisions dans `docs/decisions.md`.

## Contenu

| Chemin | Rôle |
| --- | --- |
| `CLAUDE.md` | Contexte et règles pour Claude Code |
| `docs/decisions.md` | Décisions, état, faits vérifiés (source de vérité) |
| `scripts/` | `setup_mitgcm.sh` (MITgcm 69k + gfortran), `run_cas_test.sh` (cas test de bout en bout) |
| `pipeline_grille/` | Pipeline grille + bathy + forçage (code seulement, pas de données) |
| `sagdiag/sagdiag.py` | Bibliothèque : lecture MDS (fichiers globaux ou par tuile), moyenne de phase, énergétique, tourbillons |
| `sagdiag/sagdiag_run.py` | Chaîne complète : contrôle du run, analyses, figures, `resume.txt` |
| `sagdiag/cartes_coupes.py` | Figures de présentation : courants de surface, zoom sur le cap, coupes zoomées sur la pycnocline |
| `sagdiag/animations.py` | Animations (w le long du talweg et en plan, vorticité de surface), taille des structures de w, coût et extrapolation |
| `sagdiag/coupes_test.json` | Régions et coupes du cas test |
| `sagdiag/coupes_mini_modele.json` | Modèle à compléter pour ton mini (lon/lat) |
| `sagdiag/data.diagnostics.mini_recommande` | Sorties recommandées pour le mini |
| `cas_test/` | Générateur `gen_testcase.py`, `code/`, namelists |
| `resultats_cas_test/` | Figures, tableaux et résumé du cas test |
| `imbrication/` | Extraction parent → enfant (OBCS hors ligne, conditions initiales), contrôle de cohérence, analyses au seuil (`seuil_nh.py`) et au tourbillon de l'Anse-de-Roche (`tourbillon_anse.py`) |
| `cas_test/gen_enfant.py` | Enfant du cas test : A hydrostatique z*, B non hydrostatique (`--nh`) |

Seuls numpy, scipy et matplotlib sont requis. Aucune dépendance à MITgcmutils.

## Lancer les diagnostics sur ton run

```bash
cd ~/saguenay-mitgcm
python3 sagdiag/sagdiag_run.py ~/runs/mini --config sagdiag/coupes_mini_modele.json
```

Les résultats vont dans `~/runs/mini/diag/`. Les préfixes de fichiers sont détectés automatiquement : 3D instantané avec UVEL et VVEL sur Nr niveaux, liste 2D à quelques niveaux, ETAN. Pour les forcer : `--state3d`, `--lev2d`, `--eta`, `--flux`.

Options utiles :

| Option | Défaut | Effet |
| --- | --- | --- |
| `--skip` | 1 | Cycles ignorés au début (rampe) |
| `--min-diam` | 8 | Diamètre minimal d'un tourbillon, en Δx (critère du cahier des charges) |
| `--ow` | 0.2 | Seuil Okubo-Weiss : W < −0,2 σ_W |
| `--ro-min` | 0.2 | Nombre de Rossby minimal \|ζ/f\| |
| `--level` | 1 | Niveau de la liste 2D utilisé pour la détection (0, 1, 2) |

Pour trouver les indices des niveaux à ~1, 10 et 30 m de ta grille (à reporter dans `data.diagnostics`) :

```bash
python3 -c "import sys; sys.path.insert(0,'sagdiag'); import sagdiag as s; s.print_levels('$HOME/runs/mini')"
```

## Figures de présentation (cartes et coupes)

`sagdiag/cartes_coupes.py` trace, pour le dernier cycle M2 complet d'un run, les courants de surface et des coupes verticales zoomées sur la pycnocline. Il prend 4 moments du cycle, repérés sur η moyen : flot (dη/dt max), pleine mer, jusant (dη/dt min) et basse mer. Pour l'enfant 50 m v3 :

```bash
cd ~/saguenay-mitgcm
python3 sagdiag/cartes_coupes.py ~/runs/enfantA_v3/run --origine-utm 430800 5330800 --gif
```

Sorties dans `RUN/diag/figures/` (quelques minutes, un champ 3D à la fois) :

| Fichier | Contenu |
| --- | --- |
| `fig_courants_surface.png` | Vitesse à ~1 m (couleur) et flèches, 4 phases |
| `fig_vorticite_surface.png` | ζ/f à ~1 m et flèches, 4 phases |
| `fig_cap_zoom.png` | Lignes de courant sur ζ/f autour du cap de la Pointe-aux-Crêpes, 8 instants du cycle |
| `fig_carte_coupes.png` | Bathymétrie, talweg, coupes transversales au col et au cap |
| `fig_profils.png` | Profils moyens S, T, N² : profondeur et épaisseur de la pycnocline |
| `fig_coupe_talweg.png`, `fig_coupe_col.png`, `fig_coupe_cap.png` | σ et isopycnes, vitesse vers l'aval, 4 phases, de la surface à `--zmax` |
| `courants_surface.gif` | Animation des courants de surface sur le cycle (`--gif`) |
| `cartes_coupes.txt` | Phases, pycnocline, vitesses, ζ/f au cap |

Les coupes sont tracées en profondeur vraie, z = η + r*(1 + η/H) en z*, avec la surface libre. Le tireté marque la pycnocline (N² max de chaque colonne). Le zoom vaut par défaut 2 fois la base de la pycnocline (N² > 10 % du max) ; `--zmax 0` montre toute la colonne. Axes en km UTM 19N si l'origine est connue : `--origine-utm X0 Y0` (coin sud-ouest du run), ou `grids.npz` de `--pipeline` avec `enfant.json` à côté du run. Le cap est alors placé d'office ; sinon `--cap X Y` (m, repère du run). `--cycle k` choisit le cycle [k·T, (k+1)·T).

### Animations et arguments pour aller plus loin

```bash
python3 sagdiag/animations.py ~/runs/enfantA_v3/run --origine-utm 430800 5330800
```

| Fichier | Contenu |
| --- | --- |
| `anim_w_talweg.gif` | w vraie le long du talweg (couleur), isopycnes et surface libre, sur le cycle |
| `anim_w_carte.gif` | w vraie en plan à 10 et 40 m (`--profondeurs`) |
| `anim_vorticite.gif` | ζ/f et flèches à ~1 m |
| `fig_echelles_w.png` | Longueur d'onde des structures de w le long du talweg (passages par zéro), comparée à 8 Δx et au tourbillon observé (D ≈ 230 m) |
| `animations.txt` | Échelles, coût du run (section ALL de `STDOUT.0000` ou `output.txt`) et extrapolation aux configurations visées |

En z*, w vraie = w*(1 + η/H) + (1 − d/H)·∂η/∂t (d : profondeur r*). L'extrapolation suppose le même code hydrostatique, Δt ∝ Δx et un coût proportionnel au nombre de points × pas ; la mémoire est une borne basse (le profil lite, 5 M points, a dépassé 3,8 Go).

## Méthode

Tout champ q est décomposé en q = ⟨q⟩_φ + q′, où ⟨q⟩_φ est la moyenne de phase M2 : la moyenne des instantanés à phase égale sur les cycles complets après la rampe.

Le calcul se fait hors ligne en deux passes, instantané par instantané. La passe 1 écrit les sommes par phase sur disque (memmap float32). La passe 2 calcule les écarts q′ et accumule les statistiques. La mémoire vive reste de l'ordre de quelques champs 3D, ce qui tient dans la VM de 3,8 Go.

| Diagnostic | Formule (aux centres de maille) |
| --- | --- |
| EKE | ½⟨u′² + v′²⟩ |
| MKE de phase | ½(⟨u⟩_φ² + ⟨v⟩_φ²) |
| P_h, cisaillement horizontal | −⟨u′u′⟩∂ₓU − ⟨u′v′⟩(∂ᵧU + ∂ₓV) − ⟨v′v′⟩∂ᵧV |
| P_v, cisaillement vertical | −⟨u′w′⟩∂_zU − ⟨v′w′⟩∂_zV |
| B, flottabilité | −(g/ρ₀)⟨ρ′w′⟩ (avec RHOAnoma) |
| Flux de sel aux coupes | ∫⟨U⟩_φ⟨S⟩_φ dA (moyenne de phase) + ∫⟨u′S′⟩ dA (tourbillons) |

Les conversions positives signifient un gain d'EKE. Elles sont intégrées en W et l'énergie en J, avec ρ₀ lu dans `data`.

La détection des tourbillons se fait sur le champ complet au niveau 2D choisi. Un tourbillon est une région connexe où W = s_n² + s_s² − ζ² < −0,2 σ_W, avec un diamètre équivalent d'au moins `--min-diam`·Δx et \|Ro\| ≥ `--ro-min`. Le suivi apparie par plus proche voisin de même polarité.

Pour chaque tourbillon, la fraction verrouillée en phase vaut ζ(moyenne de phase)/ζ sur son cœur. Près de 1, le tourbillon est reproduit à chaque marée au même endroit ; il fait alors partie de ⟨u⟩_φ. Près de 0, il est aléatoire et porté par u′.

La vorticité est calculée comme MITgcm. Elle reproduit `momVort3` à la précision float32 (écart relatif ~2e-7).

## Sorties

| Fichier | Contenu |
| --- | --- |
| `resume.txt`, `resume.json` | Chiffres clés : CFL, marée M2/M4, EKE/MKE, conversions (MW), flux de sel, statistiques de tourbillons |
| `fig1_controle.png` | η, CFL et KE au cours du run |
| `fig2_vorticite_cycle.png` | ζ/f sur 8 phases du dernier cycle, avec les détections |
| `fig3_energie_phase.png` | EKE et MKE par région selon la phase |
| `fig4_cartes_energie.png` | EKE, P_h, P_v, B intégrés sur la verticale |
| `fig5_flux_sel.png` | Flux de sel moyen et tourbillonnaire aux coupes, avec le total du modèle (USLTMASS) comme contrôle |
| `fig6_tourbillons.png` | Détections par phase, sites de formation, rayons, fraction verrouillée |
| `tourbillons.csv`, `trajectoires.csv` | Détections et trajectoires |
| `energetique.npz` | Champs par phase et moyennes 3D |
| `phasemean/` | Moyennes de phase 3D (float32, à effacer au besoin) |

## Cas test synthétique

Le cas test a la même emprise que le mini (37,6 × 21,6 km), 3 seuils (124 / 68 / 23 m) et un estuaire. Il utilise les mêmes OB : pompage W/E/S, rivière de 1200 m³/s et rampe d'un cycle.

Il reprend aussi tes choix numériques : z* (`nonlinFreeSurf=4`, `select_rStar=2`), `staggerTimeStep`, GGL90, éponge 3 km, advection 33, Smagorinsky 2,2, parois glissantes, Cd 2,5e-3 et Δt 30 s. La grille est à 400 m (94×54×32) pour tourner sur un cœur.

```bash
cd ~/saguenay-mitgcm/cas_test
python3 gen_testcase.py --mitgcm ~/MITgcm
mkdir -p build run && cd build
~/MITgcm/tools/genmake2 -rootdir=$HOME/MITgcm -mods=../code && make depend && make -j3
cd ../run && ln -s ../input/* . && ln -s ../build/mitgcmuv . && ./mitgcmuv > output.txt
python3 ../../sagdiag/sagdiag_run.py . --config ../../sagdiag/coupes_test.json --min-diam 3
```

Le générateur écrit les enregistrements OBCS selon la convention de MITgcm : l'enregistrement n tombe à t = (n − ½)·P. Il ajoute aussi 2 enregistrements de fin : la suite logique, puis l'état avant le départ. C'est le modèle à suivre pour `s05_forcing.py`.

## Points de méthode

- **Épaisseur z\*.** Les flux aux coupes utilisent h = h₀(H + η)/H, avec η interpolé entre les instantanés ETAN. Sans cette correction, le débit moyen aux seuils du cas test était sous-estimé de 4 à 6 %.
- **Biais d'échantillonnage.** Avec N cycles, la variance autour de la moyenne de phase est sous-estimée d'un facteur (N−1)/N. L'EKE, les conversions et les flux tourbillonnaires sont donc multipliés par N/(N−1), soit ×1,5 pour 3 cycles.
- **Éponges.** Une bande de `spongeThickness` mailles le long des 4 bords est exclue des intégrales et masquée sur les cartes (lu dans `data.obcs`).
- **Mise en route.** `resume.txt` donne l'EKE par cycle. Si le premier cycle domine, la dérive d'ajustement contamine les écarts : augmente `--skip` et allonge le run.

## Imbrication parent → enfant

L'enfant est forcé hors ligne par les sorties du parent : OBCS aux frontières et conditions initiales à t0. Il faut des grilles cartésiennes uniformes, des faces de l'enfant alignées sur celles du parent et la même grille verticale.

```bash
bash scripts/run_cas_test.sh            # parent (cas test, 400 m)
bash scripts/run_imbrication.sh A B     # enfants à 200 m, ~20 min en parallèle
```

Étapes, pour un autre couple parent/enfant :

```bash
python3 imbrication/extraire_obcs.py PARENT_RUN ENFANT_DIR   # OB*.bin, *_ini.bin, obcs_rapport.txt
# ... compiler et lancer l'enfant dans ENFANT_DIR/run ...
python3 imbrication/comparer.py PARENT_RUN ENFANT_DIR        # ENFANT_DIR/run/diag/imbrication.txt
```

`ENFANT_DIR/enfant.json` décrit la grille de l'enfant dans le repère du parent, ses frontières ouvertes, l'instant de départ t0 (multiple de 44 640 s), la durée, l'espacement des enregistrements et le type de surface libre. `cas_test/gen_enfant.py` l'écrit pour le cas test ; `imbrication/enfant_depuis_pipeline.py` l'écrit à partir des sorties du pipeline grille.

| Étape | Méthode |
| --- | --- |
| Interpolation | Bilinéaire horizontale, points secs du parent remplis par le plus proche voisin mouillé ; linéaire en temps entre instantanés |
| Enregistrements | n à t = (n − ½)·P, N + 1 enregistrements puis l'état à −P/2 ; externForcingCycle = (N + 2)·P |
| Correction de flux | Par frontière et par enregistrement : vitesse uniforme ajoutée pour que le débit entrant égale celui du parent à travers la même ligne de faces (tout le côté de l'enfant, faces du rivage comprises), moins le remplissage des cellules OB. Seules les faces qui alimentent une cellule intérieure comptent (coins). Un côté sans OB où le parent a du débit est signalé |
| z* | Débit de l'enfant avec le facteur (1 + η_OB/H) ; fichiers OB*eta fournis |
| Enfant non hydrostatique | Surface libre linéaire, fichiers OB*w ; w vraie = w*(1 + η/H) + (1 − z/H)·∂η/∂t à partir d'un parent en z* |

Le contrôle compare, hors éponge de l'enfant, l'amplitude et la phase M2 de η, le niveau moyen, les débits aux coupes du fichier de coupes et la salinité moyenne. Le critère est un écart < 5 %.

**Bogue de checkpoint69k.** Dans `pkg/obcs/obcs_apply_r_star.F`, aux frontières N et S, le facteur z* lit `OBNeta(j)`/`OBSeta(j)` au lieu de l'indice i. `gen_enfant.py` place une copie corrigée dans `code/` pour la config A. Tout enfant en z* avec une OB N ou S et des fichiers OB*eta doit l'embarquer.

## Enfant 50 m du mini (Config B non hydrostatique, sur la VM)

Le mini à 200 m sert de parent. Il doit avoir tourné 6 cycles avec `sagdiag/data.diagnostics.mini_recommande` (state3D à 930 s avec WVEL, eta2D à 360 s). L'enfant part de t0 = 89 280 s (2 cycles) et dure 2 cycles.

```bash
cd ~/saguenay-mitgcm/pipeline_grille
for s in s02_grids s03_bathy s04_verify; do SAG_PROFILE=mini python3 $s.py; done
# regarder output_mini/fig_child.png ; au besoin, ajuster l'emprise de "child" dans config.py et relancer
cd ~/saguenay-mitgcm
python3 imbrication/enfant_depuis_pipeline.py pipeline_grille/output_mini ~/runs/enfantB \
    --parent ~/runs/mini --t0 89280 --cycles 2 --nh --npx 2
python3 imbrication/extraire_obcs.py ~/runs/mini ~/runs/enfantB
cd ~/runs/enfantB && mkdir -p build run && cd build
~/MITgcm/tools/genmake2 -mpi -rootdir=$HOME/MITgcm -mods=../code && make depend && make -j3
cd ../run && ln -sf ../input/* . && ln -sf ../build/mitgcmuv . && mpirun -np 2 ./mitgcmuv > output.txt
cd ~/saguenay-mitgcm && python3 imbrication/comparer.py ~/runs/mini ~/runs/enfantB   # --config coupes.json pour les débits
```

Pour la Config A (hydrostatique, même grille), reprendre sans `--nh` dans `~/runs/enfantA`. Comparer A et B au seuil isole l'effet non hydrostatique :

```bash
python3 imbrication/seuil_nh.py ~/runs/enfantA/run ~/runs/enfantB/run      # un seul run accepté aussi
```

Le script étudie ensemble le seuil et le tourbillon de la pointe. Il trace le talweg entre les deux frontières les plus éloignées, repère le col et produit, dans `enfantB/run/diag/` :

| Fichier | Contenu |
| --- | --- |
| `fig_seuil_coupes.png` | w et isohalines le long du talweg à 8 phases du dernier cycle, A et B côte à côte |
| `fig_seuil_w.png` | w rms et \|w\| max près du col sur le cycle |
| `fig_pointe_vorticite.png` | ζ/f (~10 m) autour du col et de la pointe à 8 phases, cœur Okubo-Weiss du tourbillon |
| `fig_seuil_pointe.png` | Froude interne au col, w au col et circulation du tourbillon sur le cycle |
| `seuil_nh.txt` | Rapports B/A, Froude (max, part du cycle supercritique), tourbillon (circulation, aire, Ro), déphasage Froude → tourbillon |

Froude interne = \|U moyen\| / c₁, avec c₁ = (1/π)∫N dz (onde interne du mode 1). La pointe est le maximum de \|ζ\| moyen hors de la zone du col ; `--pointe X Y` (m, repère du run) ou `--pointe-lonlat LON LAT` la fixe, `--rayon-pointe` règle la zone (1 500 m). En z*, WVEL de A est la vitesse r* ; l'écart avec w vraie (~Aω ≈ 2e-4 m/s) est petit devant w au seuil.

À surveiller : `obcs_rapport.txt` (correction de flux de quelques %, bilan de volume < 1 % : l'erreur sur la marée de l'enfant est du même ordre ; au-delà de 3 %, ne pas lancer), le CFL dans `STDOUT.0000` (au premier pas, les vitesses interpolées ne sont pas à divergence nulle), puis `run/diag/imbrication.txt`. `comparer.py` ne compare les débits qu'aux coupes du fichier `--config` situées dans l'enfant, en mètres dans le repère du modèle.

## Enfants 25 m (profil `mini25`)

Deux zones, imbriquées dans l'enfant 50 m v3 (rapport 2) : `seuil` (ressaut de jusant, aval du 2e seuil) et `cap` (tourbillon de surface au cap de la Pointe-aux-Crêpes). Le parent est le run 50 m v3 (`~/runs/enfantA_v3`, 3,1 cycles) ; l'enfant 25 m part de t0 = 44 640 s dans le temps de ce run et dure 2 cycles. A et B sont forcés par le même parent : seule la physique diffère.

```bash
Z=cap                                         # ou seuil
cd ~/saguenay-mitgcm/pipeline_grille
for s in s02_grids s03_bathy s04_verify; do SAG_PROFILE=mini25 SAG_ZONE=$Z python3 $s.py; done
cd ~/saguenay-mitgcm
P=~/runs/enfantA_v3/run
for c in B A; do
  opt=""; [ $c = B ] && opt="--nh"
  python3 imbrication/enfant_depuis_pipeline.py pipeline_grille/output_mini25_$Z ~/runs/p25_${Z}_$c \
      --parent $P --t0 44640 --cycles 2 --npx 2 --eponge 500 $opt
  python3 imbrication/extraire_obcs.py $P ~/runs/p25_${Z}_$c | tail -4
done
```

Puis, pour chaque run, compilation `genmake2 -mpi`, essai court et `nohup mpirun -np 2` comme pour l'enfant 50 m. Analyse : `comparer.py` (parent = run 50 m) et `seuil_nh.py` A contre B, `--niveau 0` pour la surface au cap.

Au cap, `seuil_nh.py` demande `--pointe-lonlat -69.8947 48.2166 --pipeline pipeline_grille/output_mini25_cap --rayon-pointe 600` : sans `--pipeline`, l'origine est celle du mini et le point tombe hors du domaine. Détection : `sagdiag_run.py RUN --skip 0 --level 0 --min-diam 4 --sans-3d` (un seul cycle exploitable : pas d'énergétique).

Disque (cap, 160×200×32, sorties float32) : state3D à 930 s ≈ 2,4 Go par run de 2 cycles, le reste ≈ 0,7 Go. Les analyses n'utilisent que le 2e cycle de state3D : celui du 1er cycle peut être effacé après le run. Avec peu de place, mettre `frequency(3) = -1860.` dans `input/data.diagnostics` avant le lancement (1,2 Go ; `seuil_nh.py` prend les instantanés communs aux deux runs).

## Tourbillon de l'Anse-de-Roche et mouillage ADCP (`tourbillon_anse.py`)

Compare un run aux observations de l'utilisateur : cyclone de fin de jusant au sud de l'Anse-de-Roche (dérive de glace, mars 2019 et 2021) et trains d'ondes internes au mouillage ADCP du 4 juillet 2018. Le tourbillon (~1 km) est résolu à 50 m : le run `enfantA_v3` suffit, sans nouveau calcul.

```bash
python3 imbrication/tourbillon_anse.py ~/runs/enfantA_v3/run --noms A --origine-utm 430800 5330800
```

Par défaut : fenêtre des figures d'observation (69,917-69,850 O, 48,167-48,222 N), cible 48,200 N 69,882 O (rayon de recherche 1 200 m), mouillage 48,1980 N 69,87735 O, niveau 0 de `lev2D` (surface), `--skip 1` (le 1er cycle complet est ignoré). La basse mer (BM) est prise au mouillage dans `eta2D`, cycle par cycle. Deux runs A et B sur la même grille : `RUN_A RUN_B --noms A B`. Origine du run : `--origine-utm` (coin sud-ouest de l'enfant en UTM 19N), sinon `grids.npz` de `--pipeline` et `enfant.json`. La conversion lon/lat ↔ UTM est intégrée (pas de pyproj). Sinon : `--fenetre-xy`, `--cible-xy`, `--mouillage-xy` (m, repère du run). Dans `DERNIER_RUN/diag/` :

| Fichier | Contenu |
| --- | --- |
| `fig_anse_vorticite_<nom>.png` | ζ/f de surface (±20, comme les observations) et flèches, de BM −4 h à BM +1,5 h toutes les 30 min (`--heures`, `--pas`) ; contour noir : cœur cyclonique détecté ; tirets : zone ζ > 5 f qui le contient (`--seuil-patch`) |
| `fig_anse_divergence_<nom>.png` | divergence/f de surface aux mêmes heures (bandes des ondes internes, si résolues) |
| `fig_anse_series.png` | circulation, diamètre, ζ/f max du cyclone et divergence rms dans la fenêtre, en fonction de l'heure par rapport à la BM, un trait par cycle |
| `fig_anse_mouillage.png` | η et diagramme temps-profondeur de w* et des isohalines au mouillage (instantanés 3D) |
| `anse.txt` | par cycle : Γ max et son heure, diamètre (et % de la largeur du fjord), ζ/f max, V ~ Γ/(πD), durée de vie, centre en lon/lat, anticyclone le plus fort au même instant ; écart entre cycles des heures de maximum (verrouillage de phase) ; pycnocline et c₁ au mouillage |
| `anse_detection.csv` | la détection à chaque instantané 2D |

Détection : cœur Okubo-Weiss (W < −0,2 σ_W, ζ > 0,2 f) dont le centre (pondéré par ζ) est à moins de `--rayon-cible` de la cible. Mémoire : un instantané à la fois. Test sur un faux run (cyclone de Rankine R = 450 m, V = 0,6 m/s, maximum imposé à BM −0,5 h) : ζ/f 24,7 (théorie 24,7), D 894 m, Γ 1,59e3 m²/s (théorie 1,70e3), maximum retrouvé à BM −0,5 h ; divergence nulle à 1e-5 f près sur un champ non divergent en grille C. À 930 s, le mouillage virtuel ne voit pas les ondes de 5-6 min : il situe la pycnocline et la phase de marée.

## Hydraulique du seuil et fronts internes (`ondes_seuil.py`)

Le long du talweg, du rétrécissement cap / Anse-de-Roche jusqu'au bassin aval du col, sur un cycle, en heures depuis la basse mer au mouillage : profondeur de l'interface (isohaline S*), vitesse de la couche de surface, Froude composite à deux couches G (G = 1 : contrôle hydraulique), w* moyen entre 5 et 30 m. Le script ajoute aussi une coupe en travers du fjord par la cible (bombement de l'interface sous le cyclone).

```bash
python3 imbrication/ondes_seuil.py ~/runs/enfantA_v3/run ~/runs/v3_noslip/run --noms glissant noslip \
    --origine-utm 430800 5330800 --detection ~/runs/v3_noslip/run/diag/anse_detection.csv
```

| Fichier | Contenu |
| --- | --- |
| `fig_ondes_talweg.png` | Diagrammes temps-distance (un run par colonne). Un front ou un ressaut interne apparaît comme une marche qui se déplace ; la pente donne sa vitesse et son sens. Tirets : col ; pointillé : mouillage ; cercles : cyclone (`--detection`). |
| `fig_ondes_coupe_<nom>.png` | Coupe transverse par la cible à BM −3, −1,5, −0,5, 0, +0,5 et +1,5 h : vitesse le long du fjord, isohalines, interface (magenta) |
| `fig_ondes_bombement.png` | Bombement de l'interface à la cible (< 0 : remontée) et profondeur de l'interface au mouillage |
| `ondes_seuil.txt` | S*, Froude au col (max, part du cycle critique), points critiques en amont et en aval, interface au mouillage, bombement, position du cyclone |

S* vaut par défaut la salinité au maximum de N² au col (`--sstar` pour la fixer). Les profondeurs sont vraies en z* : z = η + r*(1 + η/H). Les instantanés 3D (930 ou 1860 s) suivent les fronts (~0,5 m/s), pas les ondes solitaires de 5-6 min : il faut pour ça le run non hydrostatique à 25 m avec `--hf`. Test sur un faux run (seuil de 15 m, front imposé à 0,5 m/s, dôme sous un cyclone) : le front et le dôme sont retrouvés.

## Enfant 25 m « anse », Config A non glissante (profil `mini25`, `SAG_ZONE=anse`)

La zone couvre le rétrécissement cap / Anse-de-Roche, la rive est où la couche limite décolle, le cyclone, le mouillage et le côté amont du col : x 432,0-437,6 km, y 5335,6-5341,6 km, 224×240×32. Le parent est l'enfant 50 m v3 (glissant, 930 s). L'enfant part de t0 = 89 280 s (cycle 2 du v3, celui de `v3_noslip`) et dure 1 cycle, avec des parois non glissantes et une sortie `hf2D` (u, v, w à ~1, 5 et 9 m) toutes les 120 s.

```bash
cd ~/saguenay-mitgcm && git pull
cd pipeline_grille
for s in s02_grids s03_bathy s04_verify; do SAG_PROFILE=mini25 SAG_ZONE=anse python3 $s.py; done
# regarder output_mini25_anse/fig_child.png
cd ..
python3 imbrication/enfant_depuis_pipeline.py pipeline_grille/output_mini25_anse ~/runs/p25_anse_A \
    --parent ~/runs/enfantA_v3/run --t0 89280 --cycles 1 --npx 2 --eponge 500 --no-slip --hf 120
python3 imbrication/extraire_obcs.py ~/runs/enfantA_v3/run ~/runs/p25_anse_A | tail -6
grep -i "bilan\|hors des points OB" ~/runs/p25_anse_A/obcs_rapport.txt     # bilan < 3 %, sinon ne pas lancer
cd ~/runs/p25_anse_A && mkdir -p build run && cd build
~/MITgcm/tools/genmake2 -mpi -rootdir=$HOME/MITgcm -mods=../code && make depend && make -j2
cd ../run && ln -sf ../input/* . && ln -sf ../build/mitgcmuv .
```

Essai court (60 pas), puis le vrai run :

```bash
sed 's/nTimeSteps = 8928,/nTimeSteps = 60,/' ../input/data > data && mpirun -np 2 ./mitgcmuv > output.txt
grep -i "nan\|error" STDOUT.0000 | head ; grep advcfl STDOUT.0000 | tail -4
rm -f data STDOUT.* STDERR.* eta2D.* lev2D.* hf2D.* state3D.* mean3D.* flux3D.* && ln -sf ../input/data .
nohup mpirun -np 2 ./mitgcmuv > output.txt &
```

Coût : 8 928 pas × 1,72 M points × ~1e-5 cœur.s ≈ 21 h sur 2 cœurs. Disque ≈ 3 Go (state3D à 930 s ≈ 2 Go, hf2D ≈ 0,7 Go). Analyses : `comparer.py` (parent v3), `tourbillon_anse.py` et `ondes_seuil.py` avec `--origine-utm 432000 5335600`. Le parent glissant force un enfant non glissant : les frontières sont à ≥ 1,2 km du décollement, mais il faut l'avoir en tête.
