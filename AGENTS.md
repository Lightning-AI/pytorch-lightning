# AGENTS.md

This document identifies and describes all agent and agent-like components in the [Lightning-AI/utilities](https://github.com/Lightning-AI/utilities) repository.
Agents are defined as independently-triggerable components responsible for automating tasks, orchestrating workflows, or executing delegated utility operations.
For each agent, its **name**, **purpose**, **functionality**, and **relative file location** are given.

______________________________________________________________________

## 1. GitHub Workflow Agents

### Agent: check-schema

- **Purpose**: Validates configuration or data schema on code pushes.
- **Functionality**: Automatically runs schema checks as a job for every push, using centralized logic.
- **Relative Location**: `.github/workflows/check-schema.yml`

### Agent: check-docs

- **Purpose**: Validates documentation build and style.
- **Functionality**: Builds docs and runs documentation-specific checks to keep documentation healthy.
- **Relative Location**: `.github/workflows/check-docs.yml`

### Agent: check-typing

- **Purpose**: Ensures static typing quality.
- **Functionality**: Runs type checking to maintain type correctness.
- **Relative Location**: `.github/workflows/check-typing.yml`

### Agent: check-package

- **Purpose**: Validates packaging.
- **Functionality**: Builds the distribution and verifies artifacts integrity and metadata.
- **Relative Location**: `.github/workflows/check-package.yml`

### Agent: check-md-links

- **Purpose**: Verifies Markdown links.
- **Functionality**: Scans Markdown files for broken or redirected links.
- **Relative Location**: `.github/workflows/check-md-links.yml`

### Agent: check-precommit

- **Purpose**: Enforces code style and quality gates.
- **Functionality**: Runs configured pre-commit hooks (formatting, linting, etc.).
- **Relative Location**: `.github/workflows/check-precommit.yml`

### Agent: ci-use-checks

- **Purpose**: Coordinates schema and code validation jobs for broad CI coverage.
- **Functionality**: Calls multiple reusable workflows; acts as a top-level CI orchestrator.
- **Relative Location**: `.github/workflows/ci-use-checks.yaml`

### Agent: ci-testing

- **Purpose**: Executes the test suite.
- **Functionality**: Runs unit/integration tests across supported environments.
- **Relative Location**: `.github/workflows/ci-testing.yml`

### Agent: ci-cli

- **Purpose**: Exercises CLI-related checks/tests.
- **Functionality**: Validates CLI utilities and their expected behavior.
- **Relative Location**: `.github/workflows/ci-cli.yml`

### Agent: ci-scripts

- **Purpose**: Validates repository scripts.
- **Functionality**: Runs tests and checks over automation scripts in this repo.
- **Relative Location**: `.github/workflows/ci-scripts.yml`

### Agent: deploy-docs

- **Purpose**: Publishes documentation.
- **Functionality**: Builds and deploys docs to the chosen hosting target.
- **Relative Location**: `.github/workflows/deploy-docs.yml`

### Agent: release-pypi

- **Purpose**: Publishes releases to PyPI.
- **Functionality**: Builds and uploads distribution packages upon release conditions.
- **Relative Location**: `.github/workflows/release-pypi.yml`

### Agent: cleanup-caches

- **Purpose**: Manually clears caches.
- **Functionality**: Provides a job to purge caches when necessary.
- **Relative Location**: `.github/workflows/cleanup-caches.yml`

### Agent: cron-clear-cache

- **Purpose**: Maintains clean build environments by scheduled cache clearance.
- **Functionality**: Invokes Python/environment cache clearing on a weekly schedule to keep CI fast.
- **Relative Location**: `.github/workflows/cron-clear-cache.yml`

### Agent: label-pr

- **Purpose**: Automatically labels pull requests.
- **Functionality**: Applies labels to PRs based on rules to streamline triage.
- **Relative Location**: `.github/workflows/label-pr.yml`

### Agent: ci-rtfd

- **Purpose**: Integrates with Read the Docs pipeline.
- **Functionality**: Coordinates or triggers Read the Docs builds/checks.
- **Relative Location**: `.github/workflows/ci-rtfd.yml`

______________________________________________________________________

## 2. GitHub Composite Actions

### Agent: cache

- **Purpose**: Facilitates caching of Python dependencies and environments.
- **Functionality**: Stores and restores pip/conda caches to reduce CI overhead and redundant downloads.
- **Relative Location**: `.github/actions/cache`

### Agent: pip-list

- **Purpose**: Provides environment visibility.
- **Functionality**: Prints pip-installed packages and writes a summarized section to the GitHub step summary.
- **Relative Location**: `.github/actions/pip-list`

### Agent: pkg-create

- **Purpose**: Builds and verifies distribution artifacts.
- **Functionality**: Creates source/wheel distributions and checks them with strict validation.
- **Relative Location**: `.github/actions/pkg-create`

### Agent: pkg-install

- **Purpose**: Installs the package in CI jobs.
- **Functionality**: Installs the built package and its dependencies for downstream steps.
- **Relative Location**: `.github/actions/pkg-install`

### Agent: setup-python (standard GitHub Action)

- **Purpose**: Sets up the specified Python version for CI jobs using the official GitHub Actions action.
- **Functionality**: Ensures a clean Python install with exact version matching workflow needs by leveraging [`actions/setup-python`](https://github.com/actions/setup-python).
- **Relative Location**: Referenced in `.github/workflows/*.yml` (not a custom agent or composite action in this repository)

______________________________________________________________________

## 3. CLI Utility Agents

### Agent: lightning_utilities.cli.dependencies

- **Purpose**: Automates manipulation of requirement files and dependency specifications.
- **Functionality**: Provides utilities to prune packages, pin/replace minimal versions, and swap package names across requirement files and pyproject.
- **Relative Location**: `src/lightning_utilities/cli/dependencies.py`

### Agent: lightning_utilities.cli.__main__

- **Purpose**: Entry point for CLI utilities.
- **Functionality**: Enables invoking the CLI package via `python -m lightning_utilities.cli` and orchestrates subcommands.
- **Relative Location**: `src/lightning_utilities/cli/__main__.py`

______________________________________________________________________

## 4. Core Python Utility Agents

### Agent: module_available (from core.imports)

- **Purpose**: Checks runtime availability of a named module or package.
- **Functionality**: Returns Boolean status for importability; supports conditional execution patterns.
- **Relative Location**: `src/lightning_utilities/core/imports.py`

### Agent: RequirementCache

- **Purpose**: Encapsulates package requirement availability and version checks.
- **Functionality**: Determines if current environment meets module/package specs, with diagnostics.
- **Relative Location**: `src/lightning_utilities/core/imports.py`

### Agent: LazyModule

- **Purpose**: Defers import of a Python module until actually accessed.
- **Functionality**: Creates a proxy agent for cost-saving, lazy import pattern on first attribute look-up.
- **Relative Location**: `src/lightning_utilities/core/imports.py`

### Agent: requires (decorator)

- **Purpose**: Ensures that decorated routines only execute if required modules/packages are present.
- **Functionality**: Throws warning or error if preconditions not met, halting execution.
- **Relative Location**: `src/lightning_utilities/core/imports.py`

### Agent: apply_func

- **Purpose**: Applies a function recursively to all items within complex/nested data collections.
- **Functionality**: Recursively transforms dicts, lists, tuples with a callable, supporting agent-style iteration.
- **Relative Location**: `src/lightning_utilities/core/apply_func.py`

### Agent: packaging

- **Purpose**: Parses, verifies and resolves package metadata and versions.
- **Functionality**: Provides robust version and compatibility checking for automation and CI tasks.
- **Relative Location**: `src/lightning_utilities/core/packaging.py`

______________________________________________________________________

## 5. Agent-like Scripts

### Agent: (repository scripts)

- **Purpose**: Various automation helpers (e.g., changelog, release, status).
- **Functionality**: Targeted, self-contained scripts for specialized project tasks.
- **Relative Location**: `scripts/` (per script)

______________________________________________________________________

## 6. Orchestration and Decision Agents

### Agent pattern: Declarative workflow orchestration

- **Purpose**: Enables CI jobs and composite actions to trigger, communicate, and conditionally execute based on context or outputs.
- **Functionality**: Uses matrix, outputs, and conditional logic in YAML to link agent logic across jobs.
- **Relative Location**: `.github/workflows/*.yml` (see usage of `needs`, `if`, `with`, outputs)

### Agent pattern: Enum/decision-flow agents

- **Purpose**: Encodes execution paths and decisions for agents.
- **Functionality**: Provides enumerated constants used throughout automation logic.
- **Relative Location**: `src/lightning_utilities/core/enums.py`

______________________________________________________________________

## AGENTS.md Conventions

- Document each agent by **name**, **purpose**, **functionality**, and **location**.
- Use flat Markdown headings for agent categories and individual agents.
- Reference file locations relative to repository root.
- Update and expand this file as new agents, utilities, CLI commands, or automation scripts are introduced.
- Align all documentation with [standard AGENTS.md conventions](https://github.com/openai/agents.md), [Agent Rules](https://agent-rules.org/).
- Keep instructions clear, explicit, and brief for both human collaborators and machine/agent consumers.

______________________________________________________________________
