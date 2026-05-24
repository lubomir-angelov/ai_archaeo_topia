# archeo_topia Agent Rules

This repository is a Python-first ML-for-Archaeology project focused on GIS,
computer vision, OCR, map/document digitization, annotation workflows,
segmentation/detection experiments, and production-quality automation.

The agent should provide direct guidance and code generation for explicit tasks.
Prefer small, safe, targeted changes over broad rewrites.

## Project profile

Primary stack:

- Python
- FastAPI-style service structure where APIs are present
- Docker / Docker Compose orchestration
- Makefile-driven workflows
- `pyproject.toml` and requirements files for dependency management
- `ruff` for linting and formatting
- `pytest` for tests
- Bash scripts for operational automation
- GIS, raster, vector, OCR, CV, and ML experiment workflows where applicable

## Working style

- Inspect existing code before editing.
- Make small, targeted changes.
- Prefer modifying existing modules over introducing new abstractions.
- Do not rewrite large files unless explicitly requested.
- Do not add speculative features.
- Keep service logic, domain logic, data-access logic, model code, and UI/CLI code decoupled.
- Preserve existing project structure and naming conventions.
- Explain assumptions when they affect implementation.
- Prefer clear diagnostics over silent behavior.
- Do not change database schema without explaining migration impact.
- Do not run destructive commands without explicit approval.

Destructive commands include, but are not limited to:

```bash
git reset
git clean
docker compose down -v
make reset-db
rm -rf
```

## Python standards

- Follow the Zen of Python.
- Follow PEP 8 and existing project conventions.
- Use Python built-ins and existing project dependencies first.
- Do not introduce new dependencies casually.
- Do not suggest Poetry, `python-dotenv`, or similar tools unless the repository already uses them or the user explicitly asks.
- Manage dependencies through existing `pyproject.toml`, `requirements.txt`, or service-specific requirements files.
- Use precise type hints.
- Use Google-style docstrings for public functions and classes.
- Prefer explicit dataclasses, `TypedDict`, enums, or Pydantic models where they clarify boundaries.
- Keep functions small and testable.
- Isolate side effects.
- Prefer explicit exceptions and clear error messages.
- Use structured logging where the project already logs diagnostics.
- Never print secrets, tokens, credentials, cookies, or private keys.

## FastAPI and service rules

When working on API services:

- Keep route handlers thin.
- Put business logic in service modules.
- Put persistence and query logic in repository modules.
- Validate request and response boundaries with existing schema patterns.
- Use clear HTTP status codes.
- Keep OpenAPI-visible schemas stable unless the user asks for an API change.
- Prefer explicit health and readiness endpoints.
- Avoid direct cross-service imports when an API/client boundary is more appropriate.
- Keep microservices loosely coupled.
- Use dependency injection where it improves testability without over-engineering.

## ML-for-Archaeology rules

Treat reproducibility as a core requirement.

- Record dataset versions, source documents, source map sheets, model checkpoints, prompts, thresholds, metrics, and evaluation splits.
- Keep raw data immutable where practical.
- Write derived outputs to clearly named output directories.
- Avoid overwriting experiment artifacts unless explicitly intended.
- Keep annotation data, baseline outputs, adapted model outputs, and final evaluation artifacts separate.
- Preserve links between source images/documents and generated masks, boxes, labels, prompts, predictions, and metrics.
- Track hard negatives for CV/OCR tasks, including text, roads, decorative marks, grid artifacts, scan noise, legends, and map symbology.
- Split train/validation/test data by archaeological unit, map sheet, site, survey area, document source, or other leakage-safe grouping.
- Do not randomly split tiles from the same map sheet when that would cause leakage.
- Report both quantitative metrics and qualitative examples for segmentation, detection, and OCR workflows.
- Keep experiment configuration explicit and reviewable.

## GIS and geospatial rules

- Never drop CRS metadata silently.
- Reproject explicitly.
- Clearly distinguish pixel coordinates from map coordinates.
- Preserve raster transforms, bounds, resolution, CRS, and source identifiers.
- Keep raster tiling logic separate from model inference logic.
- Keep vector export logic separate from detection/segmentation logic.
- Validate geometry validity before export where applicable.
- Prefer deterministic tile naming.
- Include source map sheet, source image, page number, or document identifier in output records.
- Avoid hidden coordinate assumptions.
- Document any CRS fallback or georeferencing approximation.

## OCR and document-processing rules

- Keep OCR extraction, post-processing, layout parsing, and persistence separate.
- Preserve page numbers and source document identifiers.
- Store confidence scores and bounding boxes when available.
- Do not collapse layout information unless the task explicitly requires plain text only.
- Handle malformed PDFs explicitly.
- Log skipped pages, failed pages, fallback behavior, and partial extraction results.
- Keep OCR engine configuration visible and reproducible.
- Avoid mixing raw OCR output with normalized text unless both are clearly labeled.

## Computer vision and segmentation rules

- Keep image loading, preprocessing, inference, post-processing, evaluation, and export as separate stages.
- Store masks, boxes, labels, prompts, and confidence scores with source identifiers.
- Avoid leakage between train and evaluation tiles.
- Keep zero-shot, prompted, and few-shot results separate.
- For SAM, MapSAM, or similar workflows, record prompt type, prompt source, model variant, checkpoint, and post-processing thresholds.
- Prefer deterministic outputs where practical.
- Save qualitative overlays for review when evaluating segmentation or detection models.

## Docker and orchestration

- Docker is the default orchestrator.
- Prefer Docker Compose for local multi-service workflows.
- Keep service ports, volumes, environment variables, and health checks explicit.
- Do not require host-specific paths unless documented.
- Keep `.env.example` updated when adding required environment variables.
- Do not commit real `.env` files or secrets.
- Prefer Makefile targets for repeatable commands.
- Ensure Docker commands are safe and non-destructive by default.
- Use health checks for services that other services depend on.
- Keep build contexts narrow where practical.

## Bash standards

Use Bash only where shell scripting is appropriate.

Use strict mode where applicable:

```bash
set -Eeuo pipefail
```

Rules:

- Use `"${MY_VAR}"` notation for variables and expansions.
- Quote variables unless there is a deliberate reason not to.
- Send diagnostics to stderr when stdout is used as a function return channel.
- Disable xtrace with `set +x` before handling secrets.
- Validate required arguments explicitly.
- Use clear error messages and non-zero exits.
- Keep shell functions small and composable.
- Prefer long option names for script arguments.
- Check scripts with ShellCheck where possible.
- Do not leak secret values into logs.
- Do not use destructive file operations without explicit user approval.

Example safe pattern:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

main() {
  local input_path="${1:-}"

  if [[ -z "${input_path}" ]]; then
    printf "Usage: %s <input-path>\n" "${0}" >&2
    exit 1
  fi

  printf "Processing %s\n" "${input_path}" >&2
}

main "$@"
```

## Dependency policy

- Prefer existing dependencies.
- Prefer the Python standard library for small utilities.
- Add dependencies only when they materially improve correctness, maintainability, or performance.
- If adding a dependency, update the correct project file.
- Explain why the dependency is needed.
- Avoid adding framework-level dependencies for small tasks.
- Avoid hidden runtime dependencies.

## Configuration rules

- Prefer explicit environment variables for runtime configuration.
- Keep defaults safe for local development.
- Keep production-sensitive values outside source control.
- Update `.env.example` when adding required environment variables.
- Validate required configuration at startup.
- Fail fast on missing critical configuration.
- Never hardcode secrets.

## Database and migration rules

When the repository uses a database:

- Do not change schema casually.
- Explain Alembic or migration impact before creating migrations.
- Keep ORM models, migrations, and repository queries consistent.
- Do not hide data cleanup inside schema migrations.
- Prefer reversible migrations where possible.
- Never reset, drop, or truncate a database without explicit approval.
- Keep query construction out of UI/page-rendering code.
- Keep repository methods focused and testable.

## Code generation rules

- Match existing style before introducing new patterns.
- Generate minimal, maintainable code.
- Include logging where operationally useful.
- Include tests for non-trivial logic.
- Prefer clear names over comments.
- Comments should explain why, not restate what the code does.
- Do not add speculative abstractions.
- Do not add compatibility layers unless required.
- Do not add global mutable state unless unavoidable.
- Keep configuration explicit.
- Keep public interfaces stable unless the task requires changing them.

## Testing and verification

Prefer these checks when relevant:

```bash
ruff check .
ruff format --check .
python -m pytest
python -m compileall src
git diff
```

For shell scripts:

```bash
shellcheck path/to/script.sh
```

For Dockerized services:

```bash
docker compose config
docker compose ps
docker compose logs --tail=100
```

Do not run expensive training, full dataset processing, destructive database
resets, or volume deletion without explicit approval.

## Makefile rules

- Prefer existing Makefile targets over ad hoc commands.
- Keep targets explicit and composable.
- Avoid hidden destructive behavior in default targets.
- Use clear target names.
- Document required environment variables.
- Do not make `make test`, `make lint`, or `make check` depend on destructive setup.

## Security rules

- Never expose secrets in logs, examples, test output, generated files, or comments.
- Redact tokens, passwords, keys, cookies, private URLs, and credentials.
- Do not suggest committing secrets.
- Use `.env.example` for variable names only.
- Keep secret-fetching and secret-creation logic isolated.
- Avoid `set -x` in scripts that may handle credentials.
- Treat uploaded datasets, archaeological site locations, and unpublished research data as sensitive unless the user says otherwise.

## Preferred implementation priorities

When making implementation choices, prefer this order:

1. Correctness.
2. Reproducibility.
3. Maintainability.
4. Clear diagnostics.
5. Minimal dependencies.
6. Performance.
7. Convenience.

Performance optimizations should be measured or clearly justified.

## Final instruction

Do the smallest safe thing that solves the explicit task, preserves the existing
architecture, and keeps the archaeology/ML workflow reproducible.

## CCE / Code Context Engine

This project may use Code Context Engine for intelligent code retrieval and
cross-session memory.

When CCE tools are available, use them actively. CCE is the preferred way to
understand the codebase before reading many files manually.

### Searching the codebase

Use `context_search` before reading files directly when exploring the codebase,
answering questions about code, or understanding how something works.

Use `context_search` for:

- Answering codebase questions such as "how does X work?" or "where is Y?"
- Exploring structure or architecture
- Finding related functions, classes, modules, or patterns
- Preparing targeted edits
- Checking existing conventions before adding new code
- Avoiding unnecessary full-file reads

Use direct file reads only when:

- You need to edit a specific known file.
- You need exact complete content from a known file path.
- You need to verify a CCE result before patching.
- The user pasted or named a specific file.

Use supporting CCE tools where available:

- `expand_chunk` to inspect full source around a compressed search result.
- `related_context` to find callers, imports, dependencies, and related symbols.
- `session_recall` to retrieve prior decisions and implementation context.
- `session_timeline` to inspect summarized steps from a previous session.
- `session_event` to inspect a specific prior tool event when needed.

### Cross-session memory

Use CCE memory both ways: recall before important answers, record after durable
decisions.

Before answering a non-trivial repository question, call `session_recall` with
a descriptive topic phrase.

Use `session_recall` especially when:

- The question touches architecture, design, naming, or conventions.
- The user asks what, why, or how something was previously done.
- You are about to recommend an approach the project may already have chosen or rejected.
- You are continuing work from a previous session.

Use topic phrases, not single words.

Good examples:

```text
session_recall("OCR service API boundary")
session_recall("GIS tile naming convention")
session_recall("SAM baseline evaluation workflow")
session_recall("FastAPI service structure")
```

Poor examples:

```text
session_recall("OCR")
session_recall("GIS")
session_recall("API")
```

If recall returns relevant entries, use them before re-deriving the answer.

After making a non-obvious or durable decision, call `record_decision`.

Use `record_decision` especially when:

- Choosing one library, pattern, model, service boundary, or approach over another.
- Resolving ambiguity in requirements.
- Establishing a naming, layout, data, or experiment convention.
- Rejecting an approach for a specific reason.
- Making a decision future sessions should not re-litigate.

Format:

```text
record_decision(decision="...", reason="...")
```

Keep both fields short and specific.

After meaningful work in a file, call `record_code_area`.

Use `record_code_area` especially when:

- Adding or substantially modifying a function, class, service, route, repository, script, or workflow.
- Tracing a non-obvious flow that future work should find quickly.
- Creating or changing important experiment, GIS, OCR, or model-evaluation code.

Format:

```text
record_code_area(file_path="...", description="...")
```

Do not record memory for trivial reads, formatting-only changes, one-off lookups,
or noisy implementation details. Memory should contain durable signal, not an
event log.

### Drilling deeper from recall

If `session_recall` returns a source session id, use drill-down tools instead of
asking the user to repeat context.

Use:

```text
session_timeline(session_id="...")
```

to inspect summarized turns from a prior session.

Use:

```text
session_event(event_id=...)
```

to inspect a specific prior tool result when the summary references something
important.

Prefer these tools over re-running expensive searches or asking the user to
re-paste context.

### CCE edit workflow

For non-trivial edits:

1. Call `session_recall` with a relevant topic phrase.
2. Use `context_search` to find relevant code areas.
3. Use `expand_chunk` or direct file reads for exact code before patching.
4. Make the smallest safe change.
5. Run relevant checks where practical.
6. Record durable decisions with `record_decision`.
7. Record meaningful changed areas with `record_code_area`.

### Output style for code work

- Be concise but complete.
- Use direct technical language.
- Avoid filler, generic summaries, and repeated restatement of the task.
- Prefer actionable diagnostics.
- State assumptions explicitly when they affect implementation.
- When suggesting code changes, show only changed lines with about three lines of surrounding context unless a full file is explicitly requested.
- For multiple changes in one file, show each change separately.
- Do not echo large unchanged code blocks.
- Write file paths, commands, code blocks, and error messages in full.
- Use full clarity for security warnings and destructive action confirmations.
