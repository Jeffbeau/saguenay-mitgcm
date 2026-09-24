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
| `sagdiag/coupes_test.json` | Régions et coupes du cas test |
| `sagdiag/coupes_mini_modele.json` | Modèle à compléter pour ton mini (lon/lat) |
| `sagdiag/data.diagnostics.mini_recommande` | Sorties recommandées pour le mini |
| `cas_test/` | Générateur `gen_testcase.py`, `code/`, namelists |
| `resultats_cas_test/` | Figures, tableaux et résumé du cas test |

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
