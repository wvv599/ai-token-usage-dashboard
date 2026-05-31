# Security Policy

## Supported versions

The latest version on the default branch is supported.

## Reporting a vulnerability

Please open a private security advisory if the repository is hosted on GitHub, or contact the maintainers privately before publishing details.

## Privacy model

AI Token Usage is designed to summarize local token usage without exposing chat content.

- Codex and Claude Code JSONL readers aggregate usage fields and minimal metadata only.
- OpenCode and Hermes readers open those apps' existing local storage files in read-only mode. This project does not create or require its own database.
- The dashboard is served on `127.0.0.1` by default.
- Generated exports may contain local project paths and session identifiers; review them before sharing.
