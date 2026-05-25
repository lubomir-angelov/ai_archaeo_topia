# CVAT Implementation Plan

## Decisions

| Decision | Value |
|---|---|
| Auto-annotation model | SAM2 (Meta's Segment Anything Model 2) |
| CVAT UI port | 9090 |
| CVAT PostgreSQL host port | 5431 |
| GPU support | CPU only for Phase 1; GPU targets included but disabled by default |
| User accounts | Single superuser initially; easily editable later |

## Architecture

CVAT runs as a Docker Compose application. The UI is for project/task creation,
annotation, review, and visual inspection. Auto-annotation uses CVAT's serverless
stack (Nuclio) to run SAM2 as a serverless function.

```
Makefile:
  install / start / stop / logs / health / backup / deploy SAM2

CVAT UI (localhost:9090):
  create project
  define labels and attributes
  create annotation tasks
  annotate and review

Later:
  CVAT CLI / SDK:
    create projects/tasks
    export COCO
    import model predictions
    run auto-annotation
```

## Directory layout

```
ai_archaeo_topia/
  .env.cvat              # CVAT environment vars (gitignored, copy from .env.cvat.example)
  .env.cvat.example      # template with safe defaults (committed)
  .gitignore             # updated with CVAT entries
  Makefile               # expanded with cvat-* targets
  data/
    annotation/          # existing — manual annotations
    cvat_exports/        # new — COCO exports from CVAT
    cvat_tasks/          # new — raw map sheet crops for upload to CVAT
  docs/
    CVAT_IMPLEMENTATION.md  # this file
    CVAT_GUIDELINES.md      # existing — annotation workflow
  tools/
    cvat/                # new — cloned CVAT repo (gitignored)
```

## CVAT repo

CVAT is an external dependency cloned into `tools/cvat/`. The repo is
`openvinotoolkit/cvat` (not `cvat-ai/cvat`, which is the legacy org).

CVAT is **not** vendored as a submodule — it is cloned on demand via
`make cvat-clone`. This avoids submodule complexity and keeps the CVAT
release cycle independent.

## Port mapping

| Service | Host port | Container port |
|---|---|---|
| CVAT UI | 9090 | 8080 |
| CVAT share | 7777 | 7777 |
| PostgreSQL | 5431 | 5432 |

No conflict with the existing app PostgreSQL on host port 5433.

## Phases

### Phase 1: Infrastructure (Makefile)

- `cvat-clone` — clone repo into `tools/cvat/`
- `cvat-init-env` — generate `.env.cvat` from `.env.cvat.example`
- `cvat-up` — standard CVAT (no serverless, for manual annotation)
- `cvat-up-serverless` — CVAT + Nuclio for SAM2 auto-annotation
- `cvat-down` — stop all CVAT containers
- `cvat-down-volumes` — stop + remove volumes (destructive, opt-in)
- `cvat-health` — wait until CVAT server responds
- `cvat-ps` — docker compose ps
- `cvat-logs` — docker compose logs -f
- `cvat-superuser` — createsuperuser (non-interactive)
- `cvat-reset` — clean volumes and restart (destructive)
- `cvat-install-nuctl` — download nuctl matching CVAT's Nuclio version
- `cvat-deploy-sam2-cpu` — deploy SAM2 CPU function
- `cvat-deploy-sam2-gpu` — deploy SAM2 GPU function (disabled by default)
- `cvat-functions` — list deployed Nuclio functions
- `cvat-undeploy-sam2` — remove SAM2 function

### Phase 2: UI workflow

After `make cvat-up-serverless` and `make cvat-deploy-sam2-cpu`:

1. Open `http://localhost:9090`
2. Log in as superuser
3. Create project: `archeo_mound_detection_v0_1`
4. Define labels: `mound`, `hard_negative_symbol`, `uncertain_ignore`
5. Define attributes (see `docs/CVAT_GUIDELINES.md`)
6. Create tasks grouped by map sheet
7. Upload image crops from `data/cvat_tasks/`
8. Annotate and review
9. Export COCO to `data/cvat_exports/`

### Phase 3: CLI automation (later)

When the annotation workflow stabilizes:

- COCO/YOLO export targets
- Task backup targets
- Prediction import targets
- SAM2 result visualization targets

## SAM2 notes

CVAT's official serverless functions ship SAM2 (not SAM1). SAM2 has:
- Better accuracy on historical map features
- Interactive tracing support
- Different prompt API than SAM1

If SAM1 compatibility is needed later, a custom Nuclio function can be
written. SAM2 deployment is the default path.

## Troubleshooting

- **Port conflicts**: Verify 9090, 7777, 5431 are free before `cvat-up`
- **Volume issues**: `make cvat-reset` clears all CVAT data (annotations, tasks, DB)
- **Nuclio crashes**: Check `make cvat-logs` for nuclio container errors
- **nuctl version mismatch**: Always run `make cvat-install-nuctl` after cloning
- **GPU out of memory**: Limit to 1 worker; SAM2 GPU deployment is opt-in
