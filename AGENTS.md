# Repository Guidelines

## Project Structure & Module Organization
- Consolidate architecture notes, runbooks, and decision records in `docs/`. Before major refactors, review and extend these documents to keep system knowledge current.

## Reference Materials
- A `reference/` folder may be present containing curated documentation and guides for common integration patterns. This folder is optional and may not exist in all clones of this repository.
- **If the folder exists, start by reading `reference/CONTENTS.md`** — it summarizes all available reference materials so you can quickly find what you need.
- Do not assume any specific documents or directories are present; always check `CONTENTS.md` first to see what materials are actually available.

## Build, Test, and Development Commands
- Use PDM to capture application and dev-only dependencies. Document the canonical install command (e.g., `pdm install --group dev`).
- Target runtimes: Python 3.14+ always.
- Enforce version policy in `pyproject.toml`: set `requires-python = ">=3.14"`.
- Expose a single test runner command (`pdm run pytest`) that covers any supporting packages.

## Coding Style & Naming Conventions
- Use 4-space indentation, type-annotate every function, and prefer built-in generics (`list[str]`, `dict[str, Any]`) with `| None` for optionals on Python 3.14+.
- Keep Django apps modular: new views, forms, services, and tasks should live under the app that owns the corresponding data or workflow.
- Treat the linter as non-optional. Run it locally before committing; unresolved linting errors should block CI.
- Write docstrings and comments in American English; focus on clarifying intent rather than restating code.

## Testing Guidelines
- Keep tests in the top-level `tests/` folder (sibling to `src/`). Name modules `test_*.py`, classes `Test*`, and functions `test_*` for automatic discovery.
- Cover asynchronous tasks, external service adapters, and LLM helpers with deterministic fixtures. Mock network-bound APIs so the suite stays offline and fast.
- Make `pdm run pytest` (or the equivalent) the default validation step before pushing changes.

## Commit & Pull Request Guidelines
- Favor concise, sentence-case commit messages that describe both the change and its intent (e.g., `Add credit balance tracking to user profiles`).
- Keep commits scoped to a single concern. Mention the affected app or feature area when useful for reviewers.
- Pull requests should summarize the change set, call out new migrations, list manual or automated test results, and attach UI screenshots or logs for behavioral updates.
