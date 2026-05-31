# Contributing

Thanks for helping improve AI Token Usage.

## Development setup

This project is intentionally dependency-light. Python 3.10+ is enough for the core CLI and dashboard.

```bash
python3 -m py_compile ai_token_usage.py
python3 ai_token_usage.py --help
python3 ai_token_usage.py --serve
```

## Contribution guidelines

- Keep parsing privacy-preserving: aggregate numeric usage and metadata only; do not print or store chat content.
- Prefer read-only access for existing app storage. Do not introduce a project-owned database or migration step.
- Keep the single-file CLI usable without mandatory third-party packages.
- Update `README.md` whenever a user-facing command, output field, or dashboard section changes.
- Include cross-platform behavior when adding default paths.

## Pull request checklist

- [ ] `python3 -m py_compile ai_token_usage.py` passes.
- [ ] `python3 ai_token_usage.py --help` works.
- [ ] README is updated for user-facing changes.
- [ ] No local logs, app storage files, screenshots with private data, or generated caches are committed.
