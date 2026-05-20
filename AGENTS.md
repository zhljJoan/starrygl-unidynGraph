# Repository Guidelines

## Project Structure & Module Organization
- Core Python package lives in `src/atc_starrygl_lib`.
- Major domains:
  - `preprocess/`: dataset, partition, rank, feature, and pipeline artifact generation.
  - `ctdg/` and `dtdg/`: mode-specific runtime and preprocess integration.
  - `models/`: CTDG/DTDG/shared model components.
  - `sampling/`, `memory/`, `comm/`, `runtime/`, `tasks/`: execution path utilities.
- Native code is under `csrc/` (sampler/partition logic), with build config in `CMakeLists.txt`.
- Tests are in `tests/`, named `test_*.py`.

## Build, Test, and Development Commands
- Install editable package:
  - `pip install -e .`
- Run unit tests:
  - `python -m pytest tests`
- Run targeted tests during development:
  - `python -m pytest tests/test_preprocess_dataset.py`
- Compile-check Python modules:
  - `python -m py_compile src/atc_starrygl_lib/preprocess/*.py`
- Build native sampler library (when C++ changes):
  - `cmake -S . -B build && cmake --build build -j`

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, type hints preferred for public functions.
- Naming:
  - modules/files: `snake_case.py`
  - functions/variables: `snake_case`
  - classes: `PascalCase`
  - constants: `UPPER_SNAKE_CASE`
- Keep artifact field names stable (`time_ptr_2`, `split_time_ptr`, `read_dist_index`, etc.) to avoid runtime incompatibility.
- Prefer small, composable helpers over large monolithic functions.

## Testing Guidelines
- Framework: `pytest`.
- New behavior must include focused tests in `tests/` with deterministic tensors/small fixtures.
- For preprocess changes, validate both content and shape/ordering (e.g., stable sort, split windows, index mapping).
- Run related test files before opening a PR; run full `tests/` for cross-module changes.

## Commit & Pull Request Guidelines
- Commit messages should be short, imperative, and scoped (e.g., `preprocess: add split_time_ptr for event mode`).
- PRs should include:
  - what changed and why,
  - affected artifacts/interfaces,
  - test commands run and results,
  - migration notes if formats/config keys changed.
- For runtime/preprocess contract updates, explicitly call out backward compatibility and fallback behavior.

## Configuration & Migration Notes
- Prefer config-driven behavior (`preprocess.*`) over hardcoded paths.
- For event mode, preserve rule: split train/val/test first, then batch per split.
- Keep `time_ptr_2` compatibility when introducing newer structures like `split_time_ptr`.
