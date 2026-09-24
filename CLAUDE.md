# CLAUDE.md — Projet MITgcm Saguenay

Contexte et règles pour Claude Code. À lire au début de chaque session.

## Objectif

Configuration MITgcm (checkpoint69k) haute résolution du fjord du Saguenay : simuler des cycles de marée M2 en 3D et diagnostiquer la dynamique des tourbillons. L'approche est phénoménologique : on cherche à identifier le phénomène, pas à reproduire le détail. Des forçages et conditions initiales approximatifs sont acceptés.

## Règles de travail

- Répondre en français. Aller droit au but, avec des commandes courtes à copier-coller.
- Toujours terminer un compte rendu par la prochaine étape.
- Les runs réels tournent d'abord sur la VM de l'utilisateur : 3 cœurs, 3,8 Go de RAM, gfortran + OpenMPI, MITgcm dans `~/MITgcm` (`genmake2 -rootdir=$HOME/MITgcm`). Tout code d'analyse doit tenir dans cette mémoire.
- `docs/decisions.md` est la source de vérité. Le mettre à jour dans le même commit que toute nouvelle décision.
- Ne jamais commiter de données ni de sorties (voir `.gitignore`) : bathymétrie NONNA, `*.bin`, `*.data`/`*.meta`, runs, build.
- Vérifier les faits MITgcm dans le source checkpoint69k plutôt que de mémoire (`scripts/setup_mitgcm.sh` le clone).

## Structure

| Chemin | Contenu |
| --- | --- |
| `docs/decisions.md` | Décisions, état, faits vérifiés |
| `sagdiag/` | Diagnostics tourbillonnaires (numpy, scipy, matplotlib) |
| `cas_test/` | Cas test synthétique « mini » : générateur, `code/`, namelists |
| `resultats_cas_test/` | Résultats de référence du cas test |
| `pipeline_grille/` | Pipeline grille + bathy + forçage de l'utilisateur (s01..s05) |
| `scripts/` | Installation de MITgcm, cas test de bout en bout |

## Dans le nuage

- Les données NONNA ne sont pas dans le dépôt : les étapes qui en dépendent tournent sur la VM.
- `bash scripts/setup_mitgcm.sh`, puis `bash scripts/run_cas_test.sh` (≈ 10 min sur 1 cœur). Le second régénère, compile, lance et diagnostique le cas test.
- Test de non-régression : comparer `cas_test/run/diag/resume.txt` à `resultats_cas_test/resume.txt`. Valeurs de référence :
  - η M2 = 1,553 m, phase +0,1° ;
  - débit moyen aux seuils 1186 à 1198 m³/s (rivière 1200) ;
  - flux de sel à ±2 % de USLTMASS ;
  - CFL max u 0,29, w 0,39 ;
  - vorticité = momVort3 à 2e-7 près.

## Plan (cahier des charges)

Cahier des charges : https://claude.ai/code/artifact/79bedf7f-436f-4b37-a33e-e3fb70c27403 (non accessible depuis le nuage ; l'essentiel est ici et dans `docs/decisions.md`).

0. Remise à plat : dépôt git (fait), pipeline grille, correction bathy, banc d'essai MPI.
1. Parent A à 100 m (marée OBCS, rivière, z*, GGL90), run de 6 jours ; critères de marée et de conservation.
2. Enfant A à 25 m par extraction OBCS depuis le parent, 2 jours ; cohérence de l'imbrication < 5 %.
3. Analyses : décomposition de phase, détection, énergétique, flux aux seuils ; robustesse 25 m contre 50 m.
4. Sensibilités, dont KPP.
5. Enfant B non hydrostatique, en surface libre linéaire (obligatoire).

## Pièges vérifiés dans le source 69k

- **OBCS.** L'enregistrement n tombe à t = (n − ½)·externForcingPeriod. Pour t < P/2, MITgcm interpole entre le dernier enregistrement et le premier. Avec une rampe, ajouter 2 enregistrements finaux (suite logique, puis état avant départ) et poser externForcingCycle = (N+2)·P.
- **Instantanés.** Les diagnostics instantanés sont décalés de |freq|/2 par défaut : mettre `timePhase(n) = 0`. Une liste avec `levels()` réserve quand même Nr niveaux dans numDiags.
- **Fréquences.** Une fréquence de sortie doit être un multiple de Δt qui divise 44 640 s : 3D à 930 s, 2D à 360 s.
- **Namelists.** Chaque ligne doit commencer par une espace. Une ligne de plus de ~200 caractères est tronquée sans avertissement : utiliser `N*valeur`.
- **Non hydrostatique.** nonlinFreeSurf ≠ 0 est refusé avec le solveur 3D.
- **Éponge OBCS.** Elle relaxe U, V, T et S. Urelax s'applique aux OB E/O, Vrelax aux OB N/S.

## Prochaines tâches

- Script d'extraction parent → enfant (OBCS hors ligne, avec correction de flux par frontière).
- Adapter `sagdiag/coupes_mini_modele.json` aux vrais seuils du mini.
- Relancer le mini sur 6 cycles avec `sagdiag/data.diagnostics.mini_recommande`, puis analyser avec `--skip 2`.
