# AI Token Usage

Privacy-preserving local token usage reports for **Codex**, **OpenCode**, **Claude Code**, and **Hermes**.

Run one Python file to inspect local usage logs, open a dashboard, export CSV/JSON, compare tools, and find high-consumption time periods without sending your data anywhere.

> Screenshot/GIF coming soon. The dashboard is fully local and can be opened with `python3 codex_token_usage.py --serve --source all`.

## Highlights

- **Local-first and private**: reads local logs/databases and aggregates numeric usage fields only.
- **No required third-party dependencies**: Python 3.10+ standard library is enough.
- **Multi-tool support**: Codex, OpenCode, Claude Code, Hermes, or all sources together.
- **Dashboard included**: cards, daily chart, hourly time-of-day chart, model/project/session tables, tool and skill tables.
- **Cross-platform defaults**: detects common macOS, Linux, and Windows user data locations.
- **Export friendly**: text, JSON, and CSV outputs.
- **Optional cost estimates**: provide your own model price file when you want API-equivalent estimates.

## Quick start

```bash
python3 codex_token_usage.py --serve --source all
```

Open:

```text
http://127.0.0.1:8765
```

The dashboard defaults to today's local date. Use the date-range picker to inspect any custom range.

## Requirements

- Python 3.10+
- No required Python packages
- Local usage data from one or more supported tools

## CLI usage

```bash
# Terminal report, Codex by default
python3 codex_token_usage.py

# Select one source
python3 codex_token_usage.py --source opencode
python3 codex_token_usage.py --source claude
python3 codex_token_usage.py --source hermes

# Combine all sources
python3 codex_token_usage.py --source all

# Date filtering
python3 codex_token_usage.py --days 7
python3 codex_token_usage.py --since 2026-05-01 --until 2026-05-29

# Machine-readable output
python3 codex_token_usage.py --format json
python3 codex_token_usage.py --format csv > token-events.csv

# Dashboard
python3 codex_token_usage.py --serve
python3 codex_token_usage.py --serve --source all

# Diagnostics
python3 codex_token_usage.py --doctor
python3 codex_token_usage.py --version
```

## Dashboard features

- Total/input/cached/output/session/API request cards
- Daily token bar chart
- Hourly token bar chart using `00:00` through `23:00` on the x-axis
- Compact value labels on hourly bars, such as `1.2k`, `56k`, and `1.3m`
- Hover tooltip for hourly bars with:
  - time period
  - total tokens
  - session count
  - usage records
  - API requests
  - input/cached/output/reasoning token split
- Source tabs: Codex, OpenCode, Claude Code, Hermes, All
- Model distribution table
- Project distribution table
- Tool category and raw tool-call tables
- Skill invocation table
- Top session table
- Optional API-equivalent cost estimates

## Cross-platform data discovery

By default, the script checks common per-user locations across macOS, Linux, and Windows.

| Tool | Common locations |
| --- | --- |
| Codex | `~/.codex`, plus common app-data directories named `codex` |
| OpenCode | `$XDG_DATA_HOME/opencode`, `~/.local/share/opencode`, `~/Library/Application Support/opencode`, `%LOCALAPPDATA%\opencode`, `%APPDATA%\opencode` |
| Claude Code | `~/.claude`, plus common app-data directories named `Claude` or `claude` |
| Hermes | `~/.hermes`, plus common app-data directories named `hermes` |

Manual paths always take priority:

```bash
python3 codex_token_usage.py --codex-path ~/.codex
python3 codex_token_usage.py --opencode-path ~/.local/share/opencode/opencode.db
python3 codex_token_usage.py --claude-path ~/.claude/projects
python3 codex_token_usage.py --hermes-path ~/.hermes/state.db
```

## Persistent settings

If the default discovery cannot find an app on a user's computer, create a settings file and point the tool at the exact location.

The script automatically checks these files if they exist:

```text
./ai-token-usage.json
~/.config/ai-token-usage/config.json
~/.ai-token-usage.json
```

You can also set an environment variable:

```bash
export AI_TOKEN_USAGE_CONFIG=/path/to/ai-token-usage.json
```

Or pass a config file explicitly:

```bash
python3 codex_token_usage.py --config /path/to/ai-token-usage.json --serve
```

Example:

```json
{
  "source": "all",
  "timezone": "Asia/Shanghai",
  "host": "127.0.0.1",
  "port": 8765,
  "hermes_paths": [
    "~/.hermes",
    "/Volumes/Data/Hermes/state.db"
  ],
  "opencode_paths": [
    "~/.local/share/opencode",
    "~/Library/Application Support/opencode",
    "%LOCALAPPDATA%/opencode"
  ],
  "price_config": "prices.example.json"
}
```

Supported settings fields:

| Field | Type | Description |
| --- | --- | --- |
| `source` | string | `codex`, `opencode`, `claude`, `hermes`, or `all` |
| `timezone` | string | IANA timezone, for example `Asia/Shanghai` or `UTC` |
| `host` | string | Dashboard host |
| `port` | number | Dashboard port |
| `codex_paths` / `codex_path` | array/string | Codex log root or JSONL file |
| `opencode_paths` / `opencode_path` | array/string | OpenCode data directory or `opencode.db` file |
| `claude_paths` / `claude_path` | array/string | Claude Code root, projects directory, or JSONL file |
| `hermes_paths` / `hermes_path` | array/string | Hermes home/profile directory or `state.db` file |
| `price_config` | string | Optional price config JSON path |
| `custom_sources` | array | User-defined JSONL token sources for apps that are not built in |

CLI flags override settings file values.

## Custom applications

Apps such as Tencent QClaw, OpenClaw, or internal tools may not have a built-in parser yet. You can still add them by exporting or converting their usage data to JSONL and registering a `custom_sources` entry in `ai-token-usage.json`.

Each line in the JSONL file should represent one usage event. The default field names are intentionally simple:

```jsonl
{"timestamp":"2026-05-31T13:00:00+08:00","session_id":"q1","model":"qclaw-model","cwd":"/repo/app","input_tokens":1200,"cached_input_tokens":300,"output_tokens":450,"reasoning_output_tokens":0,"total_tokens":1950}
{"timestamp":"2026-05-31T14:00:00+08:00","session_id":"q2","model":"qclaw-model","cwd":"/repo/app","usage":{"input_tokens":2000,"output_tokens":800,"total_tokens":2800}}
```

Then add a custom source:

```json
{
  "source": "all",
  "custom_sources": [
    {
      "name": "qclaw",
      "label": "腾讯 QClaw",
      "format": "jsonl",
      "paths": ["~/qclaw-token-usage.jsonl"],
      "mapping": {
        "timestamp": ["timestamp", "created_at", "time"],
        "input_tokens": ["input_tokens", "usage.input_tokens"],
        "cached_input_tokens": ["cached_input_tokens", "usage.cached_input_tokens"],
        "output_tokens": ["output_tokens", "usage.output_tokens"],
        "reasoning_output_tokens": ["reasoning_output_tokens", "usage.reasoning_output_tokens"],
        "total_tokens": ["total_tokens", "usage.total_tokens"],
        "session_id": ["session_id", "conversation_id"],
        "model": ["model", "model_id"],
        "cwd": ["cwd", "project_path"]
      }
    },
    {
      "name": "openclaw",
      "label": "OpenClaw",
      "format": "jsonl",
      "paths": ["~/openclaw-token-usage.jsonl"]
    }
  ]
}
```

After saving the config:

```bash
python3 codex_token_usage.py --doctor
python3 codex_token_usage.py --serve --source all
python3 codex_token_usage.py --source custom:qclaw
python3 codex_token_usage.py --source custom
```

Dashboard source tabs are generated dynamically. Each custom source appears as its own tab, and `自定义` aggregates all custom apps. To remove an app, delete its object from `custom_sources` and refresh/restart the dashboard.

## Diagnostics

Use `--doctor` when a source shows no data or when you need to confirm which files are being scanned.

```bash
python3 codex_token_usage.py --doctor
python3 codex_token_usage.py --doctor --config /path/to/ai-token-usage.json
```

It prints:

- tool version and Python version
- active source and timezone
- loaded settings file and settings keys
- configured paths and whether they exist
- discovered log/database files
- usage event counts for each source
- attribution notes, including the Hermes session-level limitation

For release/debug scripts, use:

```bash
python3 codex_token_usage.py --version
```

## Time zone

Daily and hourly grouping defaults to `Asia/Shanghai`.

```bash
python3 codex_token_usage.py --timezone Asia/Shanghai
python3 codex_token_usage.py --timezone UTC
```

## Optional cost estimates

Cost estimates are disabled by default. To enable them, provide a price config keyed by model name. A `*` entry is used as fallback.

```json
{
  "*": {
    "input_per_million": 0,
    "cached_input_per_million": 0,
    "output_per_million": 0,
    "reasoning_output_per_million": 0
  },
  "anthropic/claude-sonnet-4-5": {
    "input_per_million": 3,
    "cached_input_per_million": 0.3,
    "output_per_million": 15,
    "reasoning_output_per_million": 15
  }
}
```

Run with:

```bash
python3 codex_token_usage.py --price-config prices.example.json
python3 codex_token_usage.py --serve --source all --price-config prices.example.json
```

## Counting model

### Codex

Codex logs write `token_count` events with cumulative `total_token_usage`. The script computes the delta from the previous cumulative value in the same session file. This avoids double-counting repeated cumulative events.

### OpenCode

OpenCode usage is read from assistant message token metadata in `opencode.db` using SQLite read-only mode. This keeps timestamps aligned with the actual request time, so hourly charts can show usage in the correct hour instead of attributing a whole session to its latest update time. `tokens.cache.read + tokens.cache.write` is reported as `cached_input_tokens`. Message content is not printed or stored.

### Claude Code

Claude Code usage-bearing JSONL records are counted per usage-bearing record. `cache_creation_input_tokens + cache_read_input_tokens` is reported as `cached_input_tokens` so all tools share the same dashboard fields.

### Hermes

Hermes session totals are read from `state.db` and profile databases. `cache_read_tokens + cache_write_tokens` is reported as `cached_input_tokens`. Tool and historical skill information is counted from tool metadata; message content is not printed or stored.

Hermes currently exposes reliable input/output/cache/reasoning token totals at the session level, so usage is attributed to `ended_at` when available, otherwise `started_at`. This means Hermes hourly charts are session-time attributed rather than exact per-request attribution. Codex, OpenCode, and Claude Code use event/message/usage-record timestamps.

## Output fields

Common token fields:

- `input_tokens`
- `cached_input_tokens`
- `output_tokens`
- `reasoning_output_tokens`
- `total_tokens`
- `api_requests`
- `usage_records`
- `sessions`

Dashboard/table rollups include:

- `by_day`
- `by_hour`
- `by_tool`
- `by_model`
- `by_project`
- `by_tool_category`
- `by_tool_call`
- `by_skill_invocation`
- `sessions`

## Privacy and security

- The dashboard binds to `127.0.0.1` by default.
- OpenCode and Hermes databases are opened in read-only mode.
- The script aggregates numeric usage fields and minimal metadata.
- It does not intentionally print or store chat content.
- JSON/CSV exports can include local paths and session IDs; review exports before sharing.

See [`SECURITY.md`](SECURITY.md) for reporting and privacy notes.

## Development

```bash
python3 -m py_compile codex_token_usage.py
python3 codex_token_usage.py --help
python3 codex_token_usage.py --serve --source all
```

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).
