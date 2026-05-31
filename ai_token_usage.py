#!/usr/bin/env python3
"""Summarize local AI token usage from Codex/OpenCode/Claude/Hermes logs."""

from __future__ import annotations

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo


TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)

__version__ = "0.1.0"

SKILL_TOOL_MAP = {
    "bash": "命令执行",
    "terminal": "命令执行",
    "apply_patch": "代码修改",
    "edit": "代码修改",
    "write": "代码修改",
    "write_file": "代码修改",
    "read": "文件读取",
    "read_file": "文件读取",
    "glob": "文件搜索",
    "search_files": "文件搜索",
    "grep": "内容搜索",
    "todowrite": "任务管理",
    "kanban_show": "任务管理",
    "skill_view": "Skill 管理",
    "skills_list": "Skill 管理",
    "skill_manage": "Skill 管理",
    "webfetch": "网络访问",
    "browser_navigate": "浏览器操作",
    "browser_snapshot": "浏览器操作",
    "browser_scroll": "浏览器操作",
    "browser_click": "浏览器操作",
    "browser_console": "浏览器操作",
    "question": "用户确认",
    "clarify": "用户确认",
    "memory": "记忆管理",
    "task": "子任务代理",
    "multi_tool_use.parallel": "并行工具",
}

BUNDLED_SKILLS = {
    "batch",
    "claude-api",
    "code-review",
    "debug",
    "fewer-permission-prompts",
    "loop",
    "run",
    "run-skill-generator",
    "verify",
}

SKILL_COMMAND_RE = re.compile(r"^/(?P<name>[A-Za-z0-9_.:-][A-Za-z0-9_.:-]*)(?:\s|$)")
HERMES_SKILL_PATH_RE = re.compile(r"(?:^|/)(?P<name>[A-Za-z0-9_.:-]+)/(?:SKILL\.md|skill\.md)")

SESSION_ID_RE = re.compile(
    r"rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-(?P<id>[^/]+)\.jsonl$"
)


@dataclass(frozen=True)
class UsageEvent:
    timestamp: datetime
    date: str
    tool: str
    session_id: str
    model: str
    cwd: str
    source: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "date": self.date,
            "hour": self.hour,
            "tool": self.tool,
            "session_id": self.session_id,
            "model": self.model,
            "cwd": self.cwd,
            "project": self.project,
            "source": self.source,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
            "api_requests": self.api_requests,
        }

    @property
    def hour(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d %H:00")

    @property
    def project(self) -> str:
        if not self.cwd:
            return "(unknown)"
        return Path(self.cwd).name or self.cwd


@dataclass(frozen=True)
class ToolCallEvent:
    timestamp: datetime
    date: str
    source_tool: str
    tool_name: str
    skill: str
    session_id: str
    request_id: str
    model: str
    cwd: str
    source: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "date": self.date,
            "hour": self.hour,
            "source_tool": self.source_tool,
            "tool_name": self.tool_name,
            "tool_category": self.skill,
            "skill": self.skill,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "model": self.model,
            "cwd": self.cwd,
            "project": self.project,
            "source": self.source,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }

    @property
    def hour(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d %H:00")

    @property
    def project(self) -> str:
        if not self.cwd:
            return "(unknown)"
        return Path(self.cwd).name or self.cwd


@dataclass(frozen=True)
class SkillInvocationEvent:
    timestamp: datetime
    date: str
    source_tool: str
    skill_name: str
    skill_command: str
    skill_source: str
    plugin_name: str
    invocation_type: str
    session_id: str
    request_id: str
    model: str
    cwd: str
    source: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "date": self.date,
            "hour": self.hour,
            "source_tool": self.source_tool,
            "skill_name": self.skill_name,
            "skill_command": self.skill_command,
            "skill_source": self.skill_source,
            "plugin_name": self.plugin_name,
            "invocation_type": self.invocation_type,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "model": self.model,
            "cwd": self.cwd,
            "project": self.project,
            "source": self.source,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }

    @property
    def hour(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d %H:00")

    @property
    def project(self) -> str:
        if not self.cwd:
            return "(unknown)"
        return Path(self.cwd).name or self.cwd


def parse_timestamp(value: str, local_tz: ZoneInfo) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(local_tz)


def parse_epoch_millis(value: Any, local_tz: ZoneInfo) -> datetime:
    millis = safe_int(value)
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).astimezone(local_tz)


def parse_epoch_seconds(value: Any, local_tz: ZoneInfo) -> datetime:
    if isinstance(value, bool):
        seconds = 0.0
    elif isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        try:
            seconds = float(value)
        except ValueError:
            seconds = 0.0
    else:
        seconds = 0.0
    return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(local_tz)


def safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def token_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {field: 0 for field in TOKEN_FIELDS}
    return {field: safe_int(value.get(field)) for field in TOKEN_FIELDS}


def split_usage_evenly(usage: dict[str, int], parts: int) -> list[dict[str, int]]:
    if parts < 1:
        return []
    split = [{field: usage.get(field, 0) // parts for field in TOKEN_FIELDS} for _ in range(parts)]
    for field in TOKEN_FIELDS:
        remainder = usage.get(field, 0) - sum(part[field] for part in split)
        for index in range(remainder):
            split[index][field] += 1
    return split


def skill_for_tool(tool_name: str) -> str:
    normalized = tool_name.strip().lower()
    if not normalized:
        return "其他"
    if normalized.startswith("functions."):
        normalized = normalized.split(".", 1)[1]
    return SKILL_TOOL_MAP.get(normalized, "其他")


def normalize_skill_name(value: str) -> str:
    value = value.strip()
    if value.startswith("/"):
        value = value[1:]
    return value.split()[0] if value else ""


def scan_skill_registry(cwds: Iterable[str]) -> dict[str, dict[str, str]]:
    registry: dict[str, dict[str, str]] = {}
    for name in BUNDLED_SKILLS:
        registry[name] = {"skill_source": "bundled", "plugin_name": ""}

    def add_skill(name: str, source: str, plugin_name: str = "") -> None:
        name = normalize_skill_name(name)
        if not name:
            return
        registry.setdefault(name, {"skill_source": source, "plugin_name": plugin_name})

    def scan_root(root: Path, source: str) -> None:
        skills_dir = root / ".claude" / "skills"
        if skills_dir.is_dir():
            for skill_md in skills_dir.glob("*/SKILL.md"):
                add_skill(skill_md.parent.name, source)
        commands_dir = root / ".claude" / "commands"
        if commands_dir.is_dir():
            for command_md in commands_dir.glob("*.md"):
                add_skill(command_md.stem, source)

    user_root = Path("~").expanduser()
    user_skills = user_root / ".claude" / "skills"
    if user_skills.is_dir():
        for skill_md in user_skills.glob("*/SKILL.md"):
            add_skill(skill_md.parent.name, "user")
    user_commands = user_root / ".claude" / "commands"
    if user_commands.is_dir():
        for command_md in user_commands.glob("*.md"):
            add_skill(command_md.stem, "user")

    seen_roots: set[Path] = set()
    for cwd in cwds:
        if not cwd:
            continue
        path = Path(cwd).expanduser()
        candidates = [path, *path.parents]
        for candidate in candidates:
            if candidate in seen_roots:
                continue
            seen_roots.add(candidate)
            scan_root(candidate, "project")
            if (candidate / ".git").exists():
                break
    return registry


def classify_skill(name: str, registry: dict[str, dict[str, str]]) -> dict[str, str]:
    name = normalize_skill_name(name)
    if ":" in name:
        plugin_name, skill_name = name.split(":", 1)
        return {
            "skill_name": name,
            "skill_source": "plugin",
            "plugin_name": plugin_name,
            "display_name": skill_name,
        }
    info = registry.get(name)
    if info:
        return {"skill_name": name, "display_name": name, **info}
    return {"skill_name": name, "display_name": name, "skill_source": "unknown", "plugin_name": ""}


def extract_skill_command(text: Any, registry: dict[str, dict[str, str]]) -> str:
    if not isinstance(text, str):
        return ""
    match = SKILL_COMMAND_RE.match(text.strip())
    if not match:
        return ""
    name = normalize_skill_name(match.group("name"))
    if ":" in name or name in registry:
        return name
    return ""


def extract_hermes_skill_name(tool_name: Any, content: Any) -> str:
    tool = str(tool_name or "")
    if not tool.startswith("skill"):
        return ""
    if not isinstance(content, str):
        return tool or "skill"
    matches = [match.group("name") for match in HERMES_SKILL_PATH_RE.finditer(content)]
    if matches:
        counts: dict[str, int] = defaultdict(int)
        for name in matches:
            counts[name] += 1
        return sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]
    return tool or "skill"


def text_from_claude_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def opencode_model_name(value: Any) -> str:
    if not value:
        return "unknown"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if not isinstance(value, dict):
        return "unknown"
    model_id = str(value.get("id") or value.get("modelID") or "unknown")
    provider_id = str(value.get("providerID") or "")
    return f"{provider_id}/{model_id}" if provider_id else model_id


def first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def get_nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def first_nested(item: dict[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value = get_nested(item, path)
        if value not in (None, ""):
            return value
    return None


def claude_usage_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {field: 0 for field in TOKEN_FIELDS}
    input_tokens = safe_int(value.get("input_tokens"))
    cached_input_tokens = safe_int(value.get("cache_creation_input_tokens")) + safe_int(
        value.get("cache_read_input_tokens")
    )
    output_tokens = safe_int(value.get("output_tokens"))
    reasoning_output_tokens = safe_int(value.get("reasoning_output_tokens"))
    total_tokens = input_tokens + cached_input_tokens + output_tokens + reasoning_output_tokens
    if total_tokens == 0:
        total_tokens = safe_int(value.get("total_tokens"))
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": total_tokens,
    }


def subtract_usage(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    deltas = {field: current[field] - previous.get(field, 0) for field in TOKEN_FIELDS}
    return deltas


def discover_logs(paths: list[Path]) -> list[Path]:
    logs: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_file() and expanded.suffix == ".jsonl":
            logs.append(expanded)
            continue
        if not expanded.is_dir():
            continue
        if expanded.name == ".codex":
            logs.extend((expanded / "sessions").rglob("*.jsonl"))
            logs.extend((expanded / "archived_sessions").glob("*.jsonl"))
            continue
        logs.extend(expanded.rglob("*.jsonl"))
    return sorted(set(logs))


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    output: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        output.append(expanded)
    return output


def existing_or_all(paths: Iterable[Path]) -> list[Path]:
    candidates = unique_paths(paths)
    existing = [path for path in candidates if path.exists()]
    return existing or candidates


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value)


def user_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def app_data_candidates(app_name: str) -> list[Path]:
    """Return common per-user app data locations for macOS, Linux, and Windows."""
    home = Path.home()
    lower = app_name.lower()
    candidates: list[Path] = []

    xdg_data_home = env_path("XDG_DATA_HOME") or home / ".local" / "share"
    candidates.append(xdg_data_home / lower)

    candidates.append(home / "Library" / "Application Support" / lower)
    candidates.append(home / "Library" / "Application Support" / app_name)

    for env_name in ("LOCALAPPDATA", "APPDATA"):
        base = env_path(env_name)
        if base:
            candidates.append(base / lower)
            candidates.append(base / app_name)

    candidates.append(home / "AppData" / "Local" / lower)
    candidates.append(home / "AppData" / "Local" / app_name)
    candidates.append(home / "AppData" / "Roaming" / lower)
    candidates.append(home / "AppData" / "Roaming" / app_name)
    return unique_paths(candidates)


def default_codex_paths() -> list[Path]:
    return existing_or_all([Path("~/.codex"), *app_data_candidates("codex")])


def default_opencode_paths() -> list[Path]:
    return existing_or_all([*app_data_candidates("opencode"), Path("~/.local/share/opencode")])


def default_claude_paths() -> list[Path]:
    return existing_or_all([Path("~/.claude"), *app_data_candidates("Claude"), *app_data_candidates("claude")])


def default_hermes_paths() -> list[Path]:
    return existing_or_all([Path("~/.hermes"), *app_data_candidates("hermes")])


def session_id_from_path(path: Path) -> str:
    match = SESSION_ID_RE.search(str(path))
    if match:
        return match.group("id")
    return path.stem


def parse_codex_log(path: Path, local_tz: ZoneInfo) -> Iterable[UsageEvent]:
    session_id = session_id_from_path(path)
    model = "unknown"
    cwd = ""
    previous_total = {field: 0 for field in TOKEN_FIELDS}

    try:
        lines = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return

    with lines:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: invalid JSON in {path}:{line_number}: {exc}", file=sys.stderr)
                continue

            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue

            payload_type = payload.get("type")
            if item.get("type") == "session_meta":
                session_id = str(payload.get("id") or session_id)
                continue

            if item.get("type") == "turn_context":
                model = str(payload.get("model") or model)
                cwd = str(payload.get("cwd") or cwd)
                continue

            if payload_type != "token_count":
                continue

            info = payload.get("info")
            if not isinstance(info, dict):
                continue

            current_total = token_map(info.get("total_token_usage"))
            has_current_total = any(current_total.values())
            delta = subtract_usage(current_total, previous_total)

            if has_current_total and all(value == 0 for value in delta.values()):
                continue

            if not has_current_total or any(value < 0 for value in delta.values()):
                fallback = token_map(info.get("last_token_usage"))
                if all(value == 0 for value in fallback.values()):
                    if has_current_total:
                        previous_total = current_total
                    continue
                delta = fallback

            if has_current_total:
                previous_total = current_total
            timestamp_raw = item.get("timestamp")
            if not isinstance(timestamp_raw, str):
                continue
            timestamp = parse_timestamp(timestamp_raw, local_tz)
            yield UsageEvent(
                timestamp=timestamp,
                date=timestamp.date().isoformat(),
                tool="codex",
                session_id=session_id,
                model=model,
                cwd=cwd,
                source=str(path),
                **delta,
            )


def load_codex_events(paths: list[Path], local_tz: ZoneInfo) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    for log in discover_logs(paths):
        events.extend(parse_codex_log(log, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def parse_codex_tool_log(path: Path, local_tz: ZoneInfo) -> Iterable[ToolCallEvent]:
    session_id = session_id_from_path(path)
    model = "unknown"
    cwd = ""
    previous_total = {field: 0 for field in TOKEN_FIELDS}
    pending_tools: list[tuple[str, str]] = []

    try:
        lines = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return

    with lines:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue

            payload_type = payload.get("type")
            if item.get("type") == "session_meta":
                session_id = str(payload.get("id") or session_id)
                continue
            if item.get("type") == "turn_context":
                model = str(payload.get("model") or model)
                cwd = str(payload.get("cwd") or cwd)
                continue
            if payload_type in ("function_call", "custom_tool_call", "mcp_tool_call_end", "tool_search_call", "web_search_call"):
                tool_name = first_string(payload.get("name"), payload.get("tool"), payload_type)
                request_id = first_string(payload.get("call_id"), payload.get("id"), f"{path.stem}:{line_number}")
                pending_tools.append((tool_name, request_id))
                continue
            if payload_type != "token_count":
                continue

            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            current_total = token_map(info.get("total_token_usage"))
            has_current_total = any(current_total.values())
            delta = subtract_usage(current_total, previous_total)
            if has_current_total and all(value == 0 for value in delta.values()):
                continue
            if not has_current_total or any(value < 0 for value in delta.values()):
                fallback = token_map(info.get("last_token_usage"))
                if all(value == 0 for value in fallback.values()):
                    if has_current_total:
                        previous_total = current_total
                    continue
                delta = fallback
            if has_current_total:
                previous_total = current_total
            if not pending_tools:
                continue
            timestamp_raw = item.get("timestamp")
            if not isinstance(timestamp_raw, str):
                pending_tools = []
                continue
            try:
                timestamp = parse_timestamp(timestamp_raw, local_tz)
            except ValueError:
                pending_tools = []
                continue
            tools = pending_tools
            pending_tools = []
            for (tool_name, request_id), usage in zip(tools, split_usage_evenly(delta, len(tools))):
                yield ToolCallEvent(
                    timestamp=timestamp,
                    date=timestamp.date().isoformat(),
                    source_tool="codex",
                    tool_name=tool_name,
                    skill=skill_for_tool(tool_name),
                    session_id=session_id,
                    request_id=request_id,
                    model=model,
                    cwd=cwd,
                    source=str(path),
                    **usage,
                )


def load_codex_tool_events(paths: list[Path], local_tz: ZoneInfo) -> list[ToolCallEvent]:
    events: list[ToolCallEvent] = []
    for log in discover_logs(paths):
        events.extend(parse_codex_tool_log(log, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def discover_opencode_dbs(paths: list[Path]) -> list[Path]:
    dbs: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_file():
            dbs.append(expanded)
            continue
        if expanded.is_dir():
            candidate = expanded / "opencode.db"
            if candidate.is_file():
                dbs.append(candidate)
    return sorted(set(dbs))


def load_opencode_tool_events_from_db(path: Path, local_tz: ZoneInfo) -> list[ToolCallEvent]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"warning: cannot open OpenCode database {path}: {exc}", file=sys.stderr)
        return []

    query = """
        SELECT
            m.id,
            m.session_id,
            json_extract(m.data, '$.time.completed'),
            json_extract(m.data, '$.time.created'),
            json_extract(m.data, '$.modelID'),
            json_extract(m.data, '$.providerID'),
            json_extract(m.data, '$.path.cwd'),
            json_extract(m.data, '$.tokens.input'),
            json_extract(m.data, '$.tokens.cache.read'),
            json_extract(m.data, '$.tokens.cache.write'),
            json_extract(m.data, '$.tokens.output'),
            json_extract(m.data, '$.tokens.reasoning'),
            json_extract(m.data, '$.tokens.total'),
            json_extract(p.data, '$.tool')
        FROM message m
        JOIN part p ON p.message_id = m.id
        WHERE json_extract(m.data, '$.role') = 'assistant'
          AND json_type(m.data, '$.tokens') IS NOT NULL
          AND json_extract(p.data, '$.type') = 'tool'
          AND json_extract(p.data, '$.tool') IS NOT NULL
        ORDER BY m.time_updated, m.id, p.id
    """
    try:
        grouped: dict[str, dict[str, Any]] = {}
        for row in connection.execute(query):
            (
                message_id,
                session_id,
                completed_at,
                created_at,
                model_id,
                provider_id,
                cwd,
                input_tokens,
                cache_read_tokens,
                cache_write_tokens,
                output_tokens,
                reasoning_tokens,
                total_tokens,
                tool_name,
            ) = row
            item = grouped.setdefault(
                str(message_id),
                {
                    "session_id": str(session_id),
                    "timestamp_raw": completed_at or created_at,
                    "model": f"{provider_id}/{model_id}" if provider_id else str(model_id or "unknown"),
                    "cwd": str(cwd or ""),
                    "usage": {
                        "input_tokens": safe_int(input_tokens),
                        "cached_input_tokens": safe_int(cache_read_tokens) + safe_int(cache_write_tokens),
                        "output_tokens": safe_int(output_tokens),
                        "reasoning_output_tokens": safe_int(reasoning_tokens),
                        "total_tokens": safe_int(total_tokens),
                    },
                    "tools": [],
                },
            )
            item["tools"].append(str(tool_name or "unknown"))
    except sqlite3.Error as exc:
        print(f"warning: cannot read OpenCode tool events {path}: {exc}", file=sys.stderr)
        return []
    finally:
        connection.close()

    events: list[ToolCallEvent] = []
    for message_id, item in grouped.items():
        tools = item["tools"]
        try:
            timestamp = parse_epoch_millis(item["timestamp_raw"], local_tz)
        except Exception:
            continue
        for tool_name, usage in zip(tools, split_usage_evenly(item["usage"], len(tools))):
            events.append(
                ToolCallEvent(
                    timestamp=timestamp,
                    date=timestamp.date().isoformat(),
                    source_tool="opencode",
                    tool_name=tool_name,
                    skill=skill_for_tool(tool_name),
                    session_id=item["session_id"],
                    request_id=message_id,
                    model=item["model"],
                    cwd=item["cwd"],
                    source=str(path),
                    **usage,
                )
            )
    return sorted(events, key=lambda event: event.timestamp)


def parse_opencode_db(path: Path, local_tz: ZoneInfo) -> Iterable[UsageEvent]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"warning: cannot open OpenCode database {path}: {exc}", file=sys.stderr)
        return

    query = """
        SELECT
            m.id,
            m.session_id,
            json_extract(m.data, '$.time.completed'),
            json_extract(m.data, '$.time.created'),
            json_extract(m.data, '$.modelID'),
            json_extract(m.data, '$.providerID'),
            json_extract(m.data, '$.path.cwd'),
            json_extract(m.data, '$.tokens.input'),
            json_extract(m.data, '$.tokens.cache.read'),
            json_extract(m.data, '$.tokens.cache.write'),
            json_extract(m.data, '$.tokens.output'),
            json_extract(m.data, '$.tokens.reasoning'),
            json_extract(m.data, '$.tokens.total'),
            s.directory,
            s.model
        FROM message m
        LEFT JOIN session s ON s.id = m.session_id
        WHERE json_extract(m.data, '$.role') = 'assistant'
          AND json_type(m.data, '$.tokens') IS NOT NULL
        ORDER BY COALESCE(json_extract(m.data, '$.time.completed'), json_extract(m.data, '$.time.created')), m.id
    """
    try:
        rows = connection.execute(query)
        for row in rows:
            (
                _message_id,
                session_id,
                completed_at,
                created_at,
                model_id,
                provider_id,
                message_cwd,
                input_tokens,
                cache_read_tokens,
                cache_write_tokens,
                output_tokens,
                reasoning_tokens,
                total_tokens,
                session_directory,
                session_model,
            ) = row
            timestamp_raw = completed_at or created_at
            if timestamp_raw is None:
                continue
            timestamp = parse_epoch_millis(timestamp_raw, local_tz)
            cached_input_tokens = safe_int(cache_read_tokens) + safe_int(cache_write_tokens)
            input_count = safe_int(input_tokens)
            output_count = safe_int(output_tokens)
            reasoning_count = safe_int(reasoning_tokens)
            total_count = safe_int(total_tokens)
            if total_count == 0:
                total_count = input_count + cached_input_tokens + output_count + reasoning_count
            if total_count == 0:
                continue
            model_name = f"{provider_id}/{model_id}" if provider_id else opencode_model_name(session_model)
            yield UsageEvent(
                timestamp=timestamp,
                date=timestamp.date().isoformat(),
                tool="opencode",
                session_id=str(session_id),
                model=model_name,
                cwd=str(message_cwd or session_directory or ""),
                source=str(path),
                input_tokens=input_count,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_count,
                reasoning_output_tokens=reasoning_count,
                total_tokens=total_count,
                api_requests=1,
            )
    except sqlite3.Error as exc:
        print(f"warning: cannot read OpenCode database {path}: {exc}", file=sys.stderr)
    finally:
        connection.close()


def load_opencode_events(paths: list[Path], local_tz: ZoneInfo) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    for db in discover_opencode_dbs(paths):
        events.extend(parse_opencode_db(db, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def discover_hermes_dbs(paths: list[Path]) -> list[Path]:
    dbs: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_file():
            dbs.append(expanded)
            continue
        if not expanded.is_dir():
            continue
        candidate = expanded / "state.db"
        if candidate.is_file():
            dbs.append(candidate)
        profiles = expanded / "profiles"
        if profiles.is_dir():
            for profile_state in profiles.glob("*/state.db"):
                if profile_state.is_file():
                    dbs.append(profile_state)
    return sorted(set(dbs))


def hermes_profile_name(path: Path) -> str:
    parts = path.parts
    if "profiles" in parts:
        index = parts.index("profiles")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "default"


def parse_hermes_db(path: Path, local_tz: ZoneInfo) -> Iterable[UsageEvent]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"warning: cannot open Hermes database {path}: {exc}", file=sys.stderr)
        return

    query = """
        SELECT
            id,
            source,
            model,
            started_at,
            ended_at,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            api_call_count
        FROM sessions
        ORDER BY COALESCE(ended_at, started_at), id
    """
    try:
        profile = hermes_profile_name(path)
        cwd = str(path.parent)
        for row in connection.execute(query):
            (
                session_id,
                session_source,
                model,
                started_at,
                ended_at,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                reasoning_tokens,
                api_call_count,
            ) = row
            input_count = safe_int(input_tokens)
            cached_input_tokens = safe_int(cache_read_tokens) + safe_int(cache_write_tokens)
            output_count = safe_int(output_tokens)
            reasoning_count = safe_int(reasoning_tokens)
            total_count = input_count + cached_input_tokens + output_count + reasoning_count
            if total_count == 0:
                continue
            timestamp = parse_epoch_seconds(ended_at or started_at, local_tz)
            model_name = str(model or "unknown")
            if session_source:
                model_name = f"{session_source}/{model_name}"
            yield UsageEvent(
                timestamp=timestamp,
                date=timestamp.date().isoformat(),
                tool="hermes",
                session_id=f"{profile}:{session_id}",
                model=model_name,
                cwd=cwd,
                source=str(path),
                input_tokens=input_count,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_count,
                reasoning_output_tokens=reasoning_count,
                total_tokens=total_count,
                api_requests=safe_int(api_call_count),
            )
    except sqlite3.Error as exc:
        print(f"warning: cannot read Hermes database {path}: {exc}", file=sys.stderr)
    finally:
        connection.close()


def load_hermes_events(paths: list[Path], local_tz: ZoneInfo) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    for db in discover_hermes_dbs(paths):
        events.extend(parse_hermes_db(db, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def load_hermes_tool_events_from_db(path: Path, local_tz: ZoneInfo) -> list[ToolCallEvent]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"warning: cannot open Hermes database {path}: {exc}", file=sys.stderr)
        return []

    query = """
        SELECT
            m.id,
            m.session_id,
            m.tool_name,
            COALESCE(s.ended_at, s.started_at, m.timestamp),
            s.source,
            s.model
        FROM messages m
        LEFT JOIN sessions s ON s.id = m.session_id
        WHERE COALESCE(m.tool_name, '') <> ''
        ORDER BY m.timestamp, m.id
    """
    events: list[ToolCallEvent] = []
    try:
        profile = hermes_profile_name(path)
        cwd = str(path.parent)
        for message_id, session_id, tool_name, timestamp_raw, session_source, model in connection.execute(query):
            timestamp = parse_epoch_seconds(timestamp_raw, local_tz)
            model_name = str(model or "unknown")
            if session_source:
                model_name = f"{session_source}/{model_name}"
            events.append(
                ToolCallEvent(
                    timestamp=timestamp,
                    date=timestamp.date().isoformat(),
                    source_tool="hermes",
                    tool_name=str(tool_name or "unknown"),
                    skill=skill_for_tool(str(tool_name or "unknown")),
                    session_id=f"{profile}:{session_id}",
                    request_id=f"{profile}:{message_id}",
                    model=model_name,
                    cwd=cwd,
                    source=str(path),
                )
            )
    except sqlite3.Error as exc:
        print(f"warning: cannot read Hermes tool events {path}: {exc}", file=sys.stderr)
    finally:
        connection.close()
    return sorted(events, key=lambda event: event.timestamp)


def load_hermes_tool_events(paths: list[Path], local_tz: ZoneInfo) -> list[ToolCallEvent]:
    events: list[ToolCallEvent] = []
    for db in discover_hermes_dbs(paths):
        events.extend(load_hermes_tool_events_from_db(db, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def load_hermes_skill_events_from_db(path: Path, local_tz: ZoneInfo) -> list[SkillInvocationEvent]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"warning: cannot open Hermes database {path}: {exc}", file=sys.stderr)
        return []

    query = """
        SELECT
            m.id,
            m.session_id,
            m.tool_call_id,
            m.tool_name,
            COALESCE(s.ended_at, s.started_at, m.timestamp),
            m.content,
            s.source,
            s.model
        FROM messages m
        LEFT JOIN sessions s ON s.id = m.session_id
        WHERE m.role = 'tool'
          AND m.tool_name LIKE 'skill%'
        ORDER BY m.timestamp, m.id
    """
    events: list[SkillInvocationEvent] = []
    try:
        profile = hermes_profile_name(path)
        cwd = str(path.parent)
        for message_id, session_id, tool_call_id, tool_name, timestamp_raw, content, session_source, model in connection.execute(query):
            skill_name = extract_hermes_skill_name(tool_name, content)
            if not skill_name:
                continue
            timestamp = parse_epoch_seconds(timestamp_raw, local_tz)
            model_name = str(model or "unknown")
            if session_source:
                model_name = f"{session_source}/{model_name}"
            events.append(
                SkillInvocationEvent(
                    timestamp=timestamp,
                    date=timestamp.date().isoformat(),
                    source_tool="hermes",
                    skill_name=skill_name,
                    skill_command=skill_name,
                    skill_source="hermes",
                    plugin_name="",
                    invocation_type=str(tool_name or "skill"),
                    session_id=f"{profile}:{session_id}",
                    request_id=str(tool_call_id or f"{profile}:{message_id}"),
                    model=model_name,
                    cwd=cwd,
                    source=str(path),
                )
            )
    except sqlite3.Error as exc:
        print(f"warning: cannot read Hermes skill events {path}: {exc}", file=sys.stderr)
    finally:
        connection.close()
    return sorted(events, key=lambda event: event.timestamp)


def load_hermes_skill_events(paths: list[Path], local_tz: ZoneInfo) -> list[SkillInvocationEvent]:
    events: list[SkillInvocationEvent] = []
    for db in discover_hermes_dbs(paths):
        events.extend(load_hermes_skill_events_from_db(db, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def discover_claude_logs(paths: list[Path]) -> list[Path]:
    logs: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_file() and expanded.suffix == ".jsonl":
            logs.append(expanded)
            continue
        if not expanded.is_dir():
            continue
        projects = expanded / "projects"
        if expanded.name == ".claude" and projects.is_dir():
            logs.extend(projects.rglob("*.jsonl"))
            continue
        logs.extend(expanded.rglob("*.jsonl"))
    return sorted(set(logs))


def parse_claude_log(path: Path, local_tz: ZoneInfo) -> Iterable[UsageEvent]:
    session_id = path.stem
    model = "unknown"
    cwd = ""

    try:
        lines = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return

    with lines:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: invalid JSON in {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            if not isinstance(item, dict):
                continue

            message = item.get("message")
            if not isinstance(message, dict):
                message = {}

            session_id = first_string(
                item.get("sessionId"),
                item.get("session_id"),
                item.get("sessionID"),
                item.get("uuid"),
                item.get("id"),
                session_id,
            )
            model = first_string(item.get("model"), message.get("model"), model) or "unknown"
            cwd = first_string(
                item.get("cwd"),
                item.get("projectPath"),
                item.get("project_path"),
                cwd,
            )

            usage = item.get("usage")
            if not isinstance(usage, dict):
                usage = message.get("usage")
            delta = claude_usage_map(usage)
            if all(delta[field] == 0 for field in TOKEN_FIELDS):
                continue

            timestamp_raw = first_string(
                item.get("timestamp"),
                item.get("created_at"),
                item.get("createdAt"),
            )
            if not timestamp_raw:
                continue
            try:
                timestamp = parse_timestamp(timestamp_raw, local_tz)
            except ValueError as exc:
                print(f"warning: invalid timestamp in {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            yield UsageEvent(
                timestamp=timestamp,
                date=timestamp.date().isoformat(),
                tool="claude",
                session_id=session_id,
                model=model,
                cwd=cwd,
                source=str(path),
                **delta,
            )


def load_claude_events(paths: list[Path], local_tz: ZoneInfo) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    for log in discover_claude_logs(paths):
        events.extend(parse_claude_log(log, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def claude_tool_names_from_item(item: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for key in ("tool", "tool_name", "name"):
        value = item.get(key)
        if isinstance(value, str) and value:
            tools.append(value)

    message = item.get("message")
    if not isinstance(message, dict):
        return tools
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("tool_use", "server_tool_use"):
                name = first_string(part.get("name"), part.get("tool"), part.get("tool_name"))
                if name:
                    tools.append(name)
    return tools


def parse_claude_tool_log(path: Path, local_tz: ZoneInfo) -> Iterable[ToolCallEvent]:
    session_id = path.stem
    model = "unknown"
    cwd = ""
    try:
        lines = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return
    with lines:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            message = item.get("message")
            if not isinstance(message, dict):
                message = {}
            session_id = first_string(
                item.get("sessionId"), item.get("session_id"), item.get("sessionID"), item.get("uuid"), item.get("id"), session_id
            )
            model = first_string(item.get("model"), message.get("model"), model) or "unknown"
            cwd = first_string(item.get("cwd"), item.get("projectPath"), item.get("project_path"), cwd)
            usage = item.get("usage")
            if not isinstance(usage, dict):
                usage = message.get("usage")
            delta = claude_usage_map(usage)
            if all(delta[field] == 0 for field in TOKEN_FIELDS):
                continue
            tools = claude_tool_names_from_item(item)
            if not tools:
                continue
            timestamp_raw = first_string(item.get("timestamp"), item.get("created_at"), item.get("createdAt"))
            if not timestamp_raw:
                continue
            try:
                timestamp = parse_timestamp(timestamp_raw, local_tz)
            except ValueError:
                continue
            for index, (tool_name, split) in enumerate(zip(tools, split_usage_evenly(delta, len(tools))), 1):
                yield ToolCallEvent(
                    timestamp=timestamp,
                    date=timestamp.date().isoformat(),
                    source_tool="claude",
                    tool_name=tool_name,
                    skill=skill_for_tool(tool_name),
                    session_id=session_id,
                    request_id=f"{session_id}:{line_number}:{index}",
                    model=model,
                    cwd=cwd,
                    source=str(path),
                    **split,
                )


def load_claude_tool_events(paths: list[Path], local_tz: ZoneInfo) -> list[ToolCallEvent]:
    events: list[ToolCallEvent] = []
    for log in discover_claude_logs(paths):
        events.extend(parse_claude_tool_log(log, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def parse_claude_skill_log(
    path: Path, local_tz: ZoneInfo, registry: dict[str, dict[str, str]]
) -> Iterable[SkillInvocationEvent]:
    session_id = path.stem
    model = "unknown"
    cwd = ""
    pending_skills: list[tuple[str, str]] = []
    try:
        lines = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return
    with lines:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            message = item.get("message")
            if not isinstance(message, dict):
                message = {}
            session_id = first_string(
                item.get("sessionId"), item.get("session_id"), item.get("sessionID"), item.get("uuid"), item.get("id"), session_id
            )
            model = first_string(item.get("model"), message.get("model"), model) or "unknown"
            cwd = first_string(item.get("cwd"), item.get("projectPath"), item.get("project_path"), cwd)

            role = first_string(item.get("role"), message.get("role"), item.get("type"))
            if role == "user":
                skill_name = extract_skill_command(text_from_claude_message(message) or first_string(item.get("text"), item.get("content")), registry)
                if skill_name:
                    pending_skills.append((skill_name, f"{path.stem}:{line_number}"))
                continue

            for tool_name in claude_tool_names_from_item(item):
                if tool_name.lower() == "skill":
                    raw_name = first_string(
                        item.get("skill"), item.get("skill_name"), item.get("name"), item.get("command")
                    )
                    if raw_name:
                        pending_skills.append((normalize_skill_name(raw_name), f"{path.stem}:{line_number}"))

            usage = item.get("usage")
            if not isinstance(usage, dict):
                usage = message.get("usage")
            delta = claude_usage_map(usage)
            if all(delta[field] == 0 for field in TOKEN_FIELDS):
                continue
            if not pending_skills:
                continue
            timestamp_raw = first_string(item.get("timestamp"), item.get("created_at"), item.get("createdAt"))
            if not timestamp_raw:
                pending_skills = []
                continue
            try:
                timestamp = parse_timestamp(timestamp_raw, local_tz)
            except ValueError:
                pending_skills = []
                continue
            skills = pending_skills
            pending_skills = []
            for (skill_name, request_id), split in zip(skills, split_usage_evenly(delta, len(skills))):
                info = classify_skill(skill_name, registry)
                yield SkillInvocationEvent(
                    timestamp=timestamp,
                    date=timestamp.date().isoformat(),
                    source_tool="claude",
                    skill_name=info["skill_name"],
                    skill_command=f"/{info['skill_name']}",
                    skill_source=info["skill_source"],
                    plugin_name=info["plugin_name"],
                    invocation_type="manual",
                    session_id=session_id,
                    request_id=request_id,
                    model=model,
                    cwd=cwd,
                    source=str(path),
                    **split,
                )


def load_skill_events(
    source: str,
    claude_paths: list[Path],
    hermes_paths: list[Path],
    local_tz: ZoneInfo,
    cwds: Iterable[str],
) -> list[SkillInvocationEvent]:
    events: list[SkillInvocationEvent] = []
    if source in ("claude", "all"):
        registry = scan_skill_registry(cwds)
        for log in discover_claude_logs(claude_paths):
            events.extend(parse_claude_skill_log(log, local_tz, registry))
    if source in ("hermes", "all"):
        events.extend(load_hermes_skill_events(hermes_paths, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def custom_source_configs(settings: dict[str, Any]) -> list[dict[str, Any]]:
    value = settings.get("custom_sources") or []
    if not isinstance(value, list):
        raise SystemExit("settings field 'custom_sources' must be an array")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise SystemExit(f"custom_sources[{index}] must be an object")
        name = str(item.get("name") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise SystemExit(f"custom_sources[{index}].name must use letters, numbers, dot, underscore, or dash")
        paths = item.get("paths", item.get("path"))
        if isinstance(paths, str):
            parsed_paths = [user_path(paths)]
        elif isinstance(paths, list) and all(isinstance(path, str) for path in paths):
            parsed_paths = [user_path(path) for path in paths]
        else:
            raise SystemExit(f"custom_sources[{index}].paths must be a string or an array of strings")
        fmt = str(item.get("format") or "jsonl").lower()
        if fmt != "jsonl":
            raise SystemExit(f"custom_sources[{index}].format currently supports only 'jsonl'")
        output.append({**item, "name": name, "paths": parsed_paths, "format": fmt})
    return output


def custom_source_key(name: str) -> str:
    return f"custom:{name}"


def custom_source_label(config: dict[str, Any]) -> str:
    return str(config.get("label") or config["name"])


def custom_mapping_paths(mapping: dict[str, Any], key: str, defaults: list[str]) -> list[str]:
    value = mapping.get(key)
    if value is None:
        return defaults
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise SystemExit(f"custom source mapping field {key!r} must be a string or an array of strings")


def custom_event_from_item(
    item: dict[str, Any], config: dict[str, Any], source_path: Path, local_tz: ZoneInfo, line_number: int
) -> UsageEvent | None:
    mapping = config.get("mapping") or {}
    if not isinstance(mapping, dict):
        raise SystemExit(f"custom source {config['name']!r} mapping must be an object")

    timestamp_raw = first_nested(item, custom_mapping_paths(mapping, "timestamp", ["timestamp", "time", "created_at", "createdAt", "date"]))
    if timestamp_raw is None:
        return None
    try:
        if isinstance(timestamp_raw, (int, float)):
            timestamp = parse_epoch_millis(timestamp_raw, local_tz) if timestamp_raw > 10_000_000_000 else parse_epoch_seconds(timestamp_raw, local_tz)
        else:
            timestamp = parse_timestamp(str(timestamp_raw), local_tz)
    except Exception:
        print(f"warning: invalid timestamp in custom source {source_path}:{line_number}", file=sys.stderr)
        return None

    def mapped_int(key: str, defaults: list[str]) -> int:
        return safe_int(first_nested(item, custom_mapping_paths(mapping, key, defaults)))

    input_tokens = mapped_int("input_tokens", ["input_tokens", "input", "usage.input_tokens", "tokens.input"])
    cached_input_tokens = mapped_int("cached_input_tokens", ["cached_input_tokens", "cache_tokens", "usage.cached_input_tokens", "tokens.cache", "tokens.cache.read"])
    output_tokens = mapped_int("output_tokens", ["output_tokens", "output", "usage.output_tokens", "tokens.output"])
    reasoning_output_tokens = mapped_int("reasoning_output_tokens", ["reasoning_output_tokens", "reasoning_tokens", "usage.reasoning_output_tokens", "tokens.reasoning"])
    total_tokens = mapped_int("total_tokens", ["total_tokens", "total", "usage.total_tokens", "tokens.total"])
    if total_tokens == 0:
        total_tokens = input_tokens + cached_input_tokens + output_tokens + reasoning_output_tokens
    if total_tokens == 0:
        return None

    name = str(config["name"])
    session_id = first_nested(item, custom_mapping_paths(mapping, "session_id", ["session_id", "sessionId", "conversation_id", "id"]))
    model = first_nested(item, custom_mapping_paths(mapping, "model", ["model", "model_id", "modelID", "provider_model"]))
    cwd = first_nested(item, custom_mapping_paths(mapping, "cwd", ["cwd", "project", "project_path", "path.cwd"]))
    api_requests = mapped_int("api_requests", ["api_requests", "requests", "calls"])
    return UsageEvent(
        timestamp=timestamp,
        date=timestamp.date().isoformat(),
        tool=name,
        session_id=str(session_id or f"{name}:{source_path.name}:{line_number}"),
        model=str(model or "unknown"),
        cwd=str(cwd or ""),
        source=str(source_path),
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        total_tokens=total_tokens,
        api_requests=api_requests or 1,
    )


def load_custom_source_events(config: dict[str, Any], local_tz: ZoneInfo) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    for log in discover_logs(config["paths"]):
        try:
            lines = log.open("r", encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"warning: cannot read custom source {log}: {exc}", file=sys.stderr)
            continue
        with lines:
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"warning: invalid JSON in custom source {log}:{line_number}: {exc}", file=sys.stderr)
                    continue
                if not isinstance(item, dict):
                    continue
                event = custom_event_from_item(item, config, log, local_tz, line_number)
                if event:
                    events.append(event)
    return sorted(events, key=lambda event: event.timestamp)


def filter_skill_events(
    events: list[SkillInvocationEvent], days: int | None, since: str | None, until: str | None
) -> list[SkillInvocationEvent]:
    if not events:
        return []
    filtered = events
    if days is not None:
        last_date = datetime.fromisoformat(events[-1].date).date()
        first_date = last_date - timedelta(days=days - 1)
        filtered = [
            event
            for event in filtered
            if first_date.isoformat() <= event.date <= last_date.isoformat()
        ]
    if since:
        filtered = [event for event in filtered if event.date >= since]
    if until:
        filtered = [event for event in filtered if event.date <= until]
    return filtered


def load_tool_events(
    source: str,
    codex_paths: list[Path],
    opencode_paths: list[Path],
    claude_paths: list[Path],
    hermes_paths: list[Path],
    local_tz: ZoneInfo,
) -> list[ToolCallEvent]:
    events: list[ToolCallEvent] = []
    if source in ("codex", "all"):
        events.extend(load_codex_tool_events(codex_paths, local_tz))
    if source in ("opencode", "all"):
        for db in discover_opencode_dbs(opencode_paths):
            events.extend(load_opencode_tool_events_from_db(db, local_tz))
    if source in ("claude", "all"):
        events.extend(load_claude_tool_events(claude_paths, local_tz))
    if source in ("hermes", "all"):
        events.extend(load_hermes_tool_events(hermes_paths, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def load_events(
    source: str,
    codex_paths: list[Path],
    opencode_paths: list[Path],
    claude_paths: list[Path],
    hermes_paths: list[Path],
    local_tz: ZoneInfo,
    custom_sources: list[dict[str, Any]] | None = None,
) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    if source in ("codex", "all"):
        events.extend(load_codex_events(codex_paths, local_tz))
    if source in ("opencode", "all"):
        events.extend(load_opencode_events(opencode_paths, local_tz))
    if source in ("claude", "all"):
        events.extend(load_claude_events(claude_paths, local_tz))
    if source in ("hermes", "all"):
        events.extend(load_hermes_events(hermes_paths, local_tz))
    for config in custom_sources or []:
        key = custom_source_key(config["name"])
        if source in ("all", "custom", key, config["name"]):
            events.extend(load_custom_source_events(config, local_tz))
    return sorted(events, key=lambda event: event.timestamp)


def filter_events(
    events: list[UsageEvent], days: int | None, since: str | None, until: str | None
) -> list[UsageEvent]:
    if not events:
        return []
    filtered = events
    if days is not None:
        last_date = datetime.fromisoformat(events[-1].date).date()
        first_date = last_date - timedelta(days=days - 1)
        filtered = [
            event
            for event in filtered
            if first_date.isoformat() <= event.date <= last_date.isoformat()
        ]
    if since:
        filtered = [event for event in filtered if event.date >= since]
    if until:
        filtered = [event for event in filtered if event.date <= until]
    return filtered


def filter_tool_events(
    events: list[ToolCallEvent], days: int | None, since: str | None, until: str | None
) -> list[ToolCallEvent]:
    if not events:
        return []
    filtered = events
    if days is not None:
        last_date = datetime.fromisoformat(events[-1].date).date()
        first_date = last_date - timedelta(days=days - 1)
        filtered = [
            event
            for event in filtered
            if first_date.isoformat() <= event.date <= last_date.isoformat()
        ]
    if since:
        filtered = [event for event in filtered if event.date >= since]
    if until:
        filtered = [event for event in filtered if event.date <= until]
    return filtered


def load_price_config(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read price config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid price config JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("price config must be a JSON object keyed by model name")
    prices: dict[str, dict[str, float]] = {}
    for model, config in data.items():
        if not isinstance(config, dict):
            continue
        prices[str(model)] = {
            str(key): float(value)
            for key, value in config.items()
            if isinstance(value, (int, float))
        }
    return prices


def default_settings_paths() -> list[Path]:
    candidates: list[Path] = []
    env_config = os.environ.get("AI_TOKEN_USAGE_CONFIG")
    if env_config:
        candidates.append(user_path(env_config))
    candidates.extend(
        [
            Path("ai-token-usage.json"),
            Path("~/.config/ai-token-usage/config.json"),
            Path("~/.ai-token-usage.json"),
        ]
    )
    return unique_paths(candidates)


def resolve_settings_path(path: Path | None) -> Path | None:
    if path is not None:
        config_path = path.expanduser()
        if not config_path.is_file():
            raise SystemExit(f"settings file not found: {config_path}")
        return config_path
    for candidate in default_settings_paths():
        expanded = candidate.expanduser()
        if expanded.is_file():
            return expanded
    return None


def load_settings(path: Path | None) -> dict[str, Any]:
    config_path = resolve_settings_path(path)
    if config_path is None:
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read settings file {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid settings JSON {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("settings file must be a JSON object")
    return data


def settings_paths(settings: dict[str, Any], plural_key: str, singular_key: str) -> list[Path] | None:
    value = settings.get(plural_key, settings.get(singular_key))
    if value is None:
        return None
    if isinstance(value, str):
        return [user_path(value)]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [user_path(item) for item in value]
    raise SystemExit(f"settings field {plural_key!r} must be a string or an array of strings")


def settings_path(settings: dict[str, Any], key: str) -> Path | None:
    value = settings.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return user_path(value)
    raise SystemExit(f"settings field {key!r} must be a string")


def print_path_status(label: str, paths: list[Path], discovered: list[Path], events: int | None = None) -> None:
    print(f"\n{label}")
    if not paths:
        print("  configured paths: none")
    else:
        print("  configured paths:")
        for path in paths:
            status = "found" if path.expanduser().exists() else "missing"
            print(f"  - [{status}] {path.expanduser()}")
    if discovered:
        print("  discovered data:")
        for item in discovered:
            print(f"  - {item}")
    else:
        print("  discovered data: none")
    if events is not None:
        print(f"  usage events: {events:,}")


def output_doctor(
    settings_path_value: Path | None,
    settings: dict[str, Any],
    source: str,
    local_tz: ZoneInfo,
    codex_paths: list[Path],
    opencode_paths: list[Path],
    claude_paths: list[Path],
    hermes_paths: list[Path],
    price_config: Path | None,
    custom_sources: list[dict[str, Any]] | None = None,
) -> None:
    print("AI Token Usage doctor")
    print(f"Version: {__version__}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Source: {source}")
    print(f"Timezone: {local_tz.key}")
    print(f"Settings file: {settings_path_value or '(none)'}")
    print(f"Settings keys: {', '.join(sorted(settings)) if settings else '(none)'}")
    print(f"Price config: {price_config.expanduser() if price_config else '(none)'}")

    codex_logs = discover_logs(codex_paths)
    opencode_dbs = discover_opencode_dbs(opencode_paths)
    claude_logs = discover_claude_logs(claude_paths)
    hermes_dbs = discover_hermes_dbs(hermes_paths)

    print_path_status("Codex", codex_paths, codex_logs, len(load_codex_events(codex_paths, local_tz)) if codex_logs else 0)
    print_path_status("OpenCode", opencode_paths, opencode_dbs, len(load_opencode_events(opencode_paths, local_tz)) if opencode_dbs else 0)
    print_path_status("Claude Code", claude_paths, claude_logs, len(load_claude_events(claude_paths, local_tz)) if claude_logs else 0)
    print_path_status("Hermes", hermes_paths, hermes_dbs, len(load_hermes_events(hermes_paths, local_tz)) if hermes_dbs else 0)
    for config in custom_sources or []:
        logs = discover_logs(config["paths"])
        print_path_status(
            f"Custom: {custom_source_label(config)} ({custom_source_key(config['name'])})",
            config["paths"],
            logs,
            len(load_custom_source_events(config, local_tz)) if logs else 0,
        )

    if not any((codex_logs, opencode_dbs, claude_logs, hermes_dbs)):
        print("\nNo data sources were found. Create ai-token-usage.json or pass --*-path flags to point to your app data.")
    print("\nNotes:")
    print("  - Codex/OpenCode/Claude Code hourly charts use event/message/usage-record timestamps.")
    print("  - Hermes hourly charts use session ended_at, or started_at if ended_at is missing.")


def estimate_event_cost(event: UsageEvent, prices: dict[str, dict[str, float]]) -> float:
    config = prices.get(event.model) or prices.get("*") or {}
    if not config:
        return 0.0
    input_cost = event.input_tokens * config.get("input_per_million", 0.0)
    cached_cost = event.cached_input_tokens * config.get("cached_input_per_million", 0.0)
    output_cost = event.output_tokens * config.get("output_per_million", 0.0)
    reasoning_rate = config.get(
        "reasoning_output_per_million", config.get("output_per_million", 0.0)
    )
    reasoning_cost = event.reasoning_output_tokens * reasoning_rate
    return (input_cost + cached_cost + output_cost + reasoning_cost) / 1_000_000


def rollup(
    events: Iterable[UsageEvent],
    key_fields: tuple[str, ...],
    prices: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    sessions: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    usage_records: dict[tuple[Any, ...], int] = defaultdict(int)
    api_requests: dict[tuple[Any, ...], int] = defaultdict(int)
    costs: dict[tuple[Any, ...], float] = defaultdict(float)
    prices = prices or {}

    for event in events:
        key = tuple(getattr(event, field) for field in key_fields)
        if key not in grouped:
            grouped[key] = {field: getattr(event, field) for field in key_fields}
            for token_field in TOKEN_FIELDS:
                grouped[key][token_field] = 0
        for token_field in TOKEN_FIELDS:
            grouped[key][token_field] += getattr(event, token_field)
        sessions[key].add(event.session_id)
        usage_records[key] += 1
        api_requests[key] += event.api_requests
        costs[key] += estimate_event_cost(event, prices)

    rows = []
    for key, row in grouped.items():
        row["api_requests"] = api_requests[key]
        row["usage_records"] = usage_records[key]
        row["calls"] = api_requests[key]
        row["sessions"] = len(sessions[key])
        row["cost_usd"] = round(costs[key], 6)
        rows.append(row)
    return sorted(rows, key=lambda row: tuple(str(row[field]) for field in key_fields))


def tool_rollup(events: Iterable[ToolCallEvent], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    sessions: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    requests: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    calls: dict[tuple[Any, ...], int] = defaultdict(int)
    for event in events:
        key = tuple(getattr(event, field) for field in key_fields)
        if key not in grouped:
            grouped[key] = {field: getattr(event, field) for field in key_fields}
            for token_field in TOKEN_FIELDS:
                grouped[key][token_field] = 0
        for token_field in TOKEN_FIELDS:
            grouped[key][token_field] += getattr(event, token_field)
        sessions[key].add(event.session_id)
        requests[key].add(event.request_id)
        calls[key] += event.calls
    rows = []
    for key, row in grouped.items():
        row["calls"] = calls[key]
        row["api_requests"] = len(requests[key])
        row["sessions"] = len(sessions[key])
        rows.append(row)
    return sorted(rows, key=lambda row: row["total_tokens"], reverse=True)


def totals(
    events: Iterable[UsageEvent], prices: dict[str, dict[str, float]] | None = None
) -> dict[str, Any]:
    output = {field: 0 for field in TOKEN_FIELDS}
    usage_record_count = 0
    api_request_count = 0
    session_ids: set[str] = set()
    cost = 0.0
    prices = prices or {}
    for event in events:
        usage_record_count += 1
        api_request_count += event.api_requests
        session_ids.add(event.session_id)
        for field in TOKEN_FIELDS:
            output[field] += getattr(event, field)
        cost += estimate_event_cost(event, prices)
    output["api_requests"] = api_request_count
    output["usage_records"] = usage_record_count
    output["calls"] = api_request_count
    output["sessions"] = len(session_ids)
    output["cost_usd"] = round(cost, 6)
    return output


def session_rollup(
    events: Iterable[UsageEvent], prices: dict[str, dict[str, float]] | None = None
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    prices = prices or {}
    for event in events:
        row = grouped.setdefault(
            event.session_id,
            {
                "tool": event.tool,
                "session_id": event.session_id,
                "date": event.date,
                "started_at": event.timestamp.isoformat(),
                "ended_at": event.timestamp.isoformat(),
                "model": event.model,
                "cwd": event.cwd,
                "project": event.project,
                "api_requests": 0,
                "usage_records": 0,
                "calls": 0,
                **{field: 0 for field in TOKEN_FIELDS},
                "cost_usd": 0.0,
            },
        )
        row["started_at"] = min(row["started_at"], event.timestamp.isoformat())
        row["ended_at"] = max(row["ended_at"], event.timestamp.isoformat())
        row["date"] = min(row["date"], event.date)
        row["tool"] = event.tool if row["tool"] == event.tool else "mixed"
        row["model"] = event.model if row["model"] == "unknown" else row["model"]
        row["cwd"] = event.cwd or row["cwd"]
        row["project"] = event.project if event.project != "(unknown)" else row["project"]
        row["api_requests"] += event.api_requests
        row["usage_records"] += 1
        row["calls"] = row["api_requests"]
        for field in TOKEN_FIELDS:
            row[field] += getattr(event, field)
        row["cost_usd"] += estimate_event_cost(event, prices)
    rows = list(grouped.values())
    for row in rows:
        row["cost_usd"] = round(row["cost_usd"], 6)
    return sorted(rows, key=lambda row: row["total_tokens"], reverse=True)


def fmt_int(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no data)")
        return
    widths = {
        column: max(len(column), *(len(fmt_int(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        values = []
        for column in columns:
            value = fmt_int(row.get(column, ""))
            if isinstance(row.get(column), int):
                values.append(value.rjust(widths[column]))
            else:
                values.append(value.ljust(widths[column]))
        print("  ".join(values))


def output_text(
    events: list[UsageEvent],
    prices: dict[str, dict[str, float]],
    tool_events: list[ToolCallEvent] | None = None,
    skill_events: list[SkillInvocationEvent] | None = None,
) -> None:
    tool_events = tool_events or []
    skill_events = skill_events or []
    total = totals(events, prices)
    print("AI token usage")
    print(f"API requests: {total['api_requests']:,}")
    print(f"Usage records: {total['usage_records']:,}")
    print(f"Sessions: {total['sessions']:,}")
    print(f"Input: {total['input_tokens']:,}")
    print(f"Cached input: {total['cached_input_tokens']:,}")
    print(f"Output: {total['output_tokens']:,}")
    print(f"Reasoning output: {total['reasoning_output_tokens']:,}")
    print(f"Total: {total['total_tokens']:,}")
    if prices:
        print(f"Estimated cost: ${total['cost_usd']:.4f}")

    common_columns = [
        "api_requests",
        "usage_records",
        "sessions",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
    ]
    if prices:
        common_columns.append("cost_usd")
    print_table("By day", rollup(events, ("date",), prices), ["date", *common_columns])
    print_table("By tool", rollup(events, ("tool",), prices), ["tool", *common_columns])
    print_table("By model", rollup(events, ("tool", "model"), prices), ["tool", "model", *common_columns])
    print_table(
        "By project",
        rollup(events, ("tool", "project", "cwd"), prices),
        ["tool", "project", "cwd", *common_columns],
    )
    print_table(
        "Top sessions",
        session_rollup(events, prices)[:10],
        ["date", "tool", "project", "model", "api_requests", "total_tokens", "input_tokens", "output_tokens"]
        + (["cost_usd"] if prices else []),
    )
    print_table(
        "By tool category",
        tool_rollup(tool_events, ("source_tool", "skill"))[:20],
        ["source_tool", "skill", "calls", "api_requests", "sessions", "total_tokens", "input_tokens", "output_tokens"],
    )
    print_table(
        "By tool call",
        tool_rollup(tool_events, ("source_tool", "tool_name", "skill"))[:20],
        ["source_tool", "tool_name", "skill", "calls", "api_requests", "total_tokens"],
    )
    print_table(
        "By skill invocation",
        tool_rollup(skill_events, ("source_tool", "skill_name", "skill_source", "plugin_name", "invocation_type"))[:20],
        ["source_tool", "skill_name", "skill_source", "plugin_name", "invocation_type", "calls", "api_requests", "sessions"],
    )


def summary_payload(
    events: list[UsageEvent],
    prices: dict[str, dict[str, float]],
    source: str = "codex",
    tool_events: list[ToolCallEvent] | None = None,
    skill_events: list[SkillInvocationEvent] | None = None,
    custom_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tool_events = tool_events or []
    skill_events = skill_events or []
    by_tool_category = tool_rollup(tool_events, ("source_tool", "skill"))
    return {
        "source": source,
        "sources": sorted({event.tool for event in events}),
        "available_sources": [
            {"key": "codex", "label": "Codex"},
            {"key": "opencode", "label": "OpenCode"},
            {"key": "claude", "label": "Claude Code"},
            {"key": "hermes", "label": "Hermes"},
            *[
                {"key": custom_source_key(config["name"]), "label": custom_source_label(config)}
                for config in custom_sources or []
            ],
            *([{"key": "custom", "label": "自定义"}] if custom_sources else []),
            {"key": "all", "label": "全部"},
        ],
        "totals": totals(events, prices),
        "by_day": rollup(events, ("date",), prices),
        "by_hour": rollup(events, ("hour",), prices),
        "by_tool": rollup(events, ("tool",), prices),
        "by_model": rollup(events, ("tool", "model"), prices),
        "by_project": rollup(events, ("tool", "project", "cwd"), prices),
        "sessions": session_rollup(events, prices),
        "by_tool_category": by_tool_category,
        "by_skill": by_tool_category,
        "by_tool_call": tool_rollup(tool_events, ("source_tool", "tool_name", "skill")),
        "by_skill_invocation": tool_rollup(
            skill_events, ("source_tool", "skill_name", "skill_source", "plugin_name", "invocation_type")
        ),
        "tool_events": [event.as_dict() for event in tool_events],
        "skill_events": [event.as_dict() for event in skill_events],
        "events": [event.as_dict() for event in events],
        "has_price_config": bool(prices),
    }


def output_json(
    events: list[UsageEvent],
    prices: dict[str, dict[str, float]],
    source: str,
    tool_events: list[ToolCallEvent] | None = None,
    skill_events: list[SkillInvocationEvent] | None = None,
    custom_sources: list[dict[str, Any]] | None = None,
) -> None:
    payload = summary_payload(events, prices, source, tool_events, skill_events, custom_sources)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def output_csv(events: list[UsageEvent]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "timestamp",
            "date",
            "hour",
            "tool",
            "session_id",
            "model",
            "cwd",
            "project",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
            "api_requests",
            "source",
        ],
    )
    writer.writeheader()
    for event in events:
        writer.writerow(event.as_dict())


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Token 用量看板</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: rgba(255, 255, 255, 0.92);
      --panel-solid: #ffffff;
      --text: #111827;
      --muted: #667085;
      --line: #e6eaf0;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --accent-2: #16a34a;
      --accent-3: #7c3aed;
      --danger: #b42318;
      --shadow: 0 16px 44px rgba(20, 34, 64, 0.10);
      --shadow-soft: 0 8px 22px rgba(20, 34, 64, 0.06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 12% -10%, rgba(37, 99, 235, 0.22) 0, transparent 34%),
        radial-gradient(circle at 88% 0%, rgba(124, 58, 237, 0.15) 0, transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 38%, #eef4fb 100%);
      color: var(--text);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid rgba(228, 231, 236, 0.78);
      background: rgba(248, 251, 255, 0.82);
      backdrop-filter: blur(18px);
      box-shadow: 0 1px 0 rgba(255,255,255,0.55) inset;
    }
    .wrap { width: min(1280px, calc(100vw - 32px)); margin: 0 auto; }
    .topbar {
      min-height: 82px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }
    h1 { margin: 0; font-size: clamp(22px, 3vw, 30px); line-height: 1.08; letter-spacing: -0.035em; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .controls { display: flex; align-items: center; justify-content: flex-end; gap: 10px; flex-wrap: wrap; }
    .tabs { display: flex; gap: 4px; flex-wrap: wrap; padding: 4px; border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,0.72); box-shadow: var(--shadow-soft); }
    .date-range {
      position: relative;
    }
    .date-trigger { min-width: 230px; text-align: left; font-variant-numeric: tabular-nums; }
    .date-popover {
      position: absolute;
      right: 0;
      top: calc(100% + 8px);
      z-index: 4;
      width: min(680px, calc(100vw - 32px));
      background: var(--panel-solid);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 18px 42px rgba(31, 41, 51, 0.18);
      padding: 12px;
    }
    .date-popover[hidden] { display: none; }
    .date-popover-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
    .months { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .month-title { font-weight: 700; font-size: 13px; margin-bottom: 8px; text-align: center; }
    .calendar-grid { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 4px; }
    .dow { color: var(--muted); font-size: 11px; text-align: center; padding: 4px 0; }
    .day {
      min-height: 32px;
      border-radius: 6px;
      border: 1px solid transparent;
      background: transparent;
      padding: 0;
      text-align: center;
    }
    .day:hover { border-color: var(--accent); color: var(--accent); }
    .day.selected { background: var(--accent); border-color: var(--accent); color: white; }
    .day.in-range { background: var(--accent-soft); }
    .day.selected.in-range { background: var(--accent); }
    button, select {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel-solid);
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; transition: all 0.18s ease; }
    button:hover { transform: translateY(-1px); border-color: rgba(37, 99, 235, 0.35); }
    .date-popover button:hover { transform: none; }
    button.tab { border-color: transparent; background: transparent; min-height: 32px; padding: 0 12px; }
    button.tab.active { background: linear-gradient(135deg, var(--text), #24324b); border-color: transparent; color: white; box-shadow: 0 8px 18px rgba(17, 24, 39, 0.18); }
    button.primary { background: linear-gradient(135deg, var(--accent), #38bdf8); border-color: transparent; color: white; box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24); font-weight: 700; }
    main { padding: 24px 0 44px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .card {
      position: relative;
      isolation: isolate;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,255,255,0.86));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 17px 16px;
      min-width: 0;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .card::before {
      content: "";
      position: absolute;
      inset: 0 0 auto;
      height: 3px;
      background: linear-gradient(90deg, var(--accent), #38bdf8, var(--accent-3));
      z-index: -1;
    }
    .label { color: var(--muted); font-size: 12px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .value { font-size: clamp(22px, 2.2vw, 30px); font-weight: 850; margin-top: 6px; letter-spacing: -0.035em; font-variant-numeric: tabular-nums; }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
      align-items: start;
    }
    .wide-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
      align-items: start;
      margin-top: 2px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }
    section h2 {
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      background: linear-gradient(90deg, rgba(248,250,252,0.98), rgba(255,255,255,0.88));
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    section h2::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent), #38bdf8);
      box-shadow: 0 0 0 4px var(--accent-soft);
    }
    section > div { overflow-x: auto; }
    .chart { padding: 14px 16px; display: grid; gap: 10px; overflow-x: visible; }
    .hourly-chart { overflow-x: auto; }
    .hour-axis-chart {
      min-height: 260px;
      min-width: 760px;
      display: grid;
      grid-template-columns: repeat(24, minmax(22px, 1fr));
      gap: 8px;
      align-items: end;
      padding-top: 16px;
    }
    .hour-column {
      position: relative;
      min-width: 0;
      display: grid;
      grid-template-rows: 1fr auto;
      gap: 8px;
      height: 250px;
    }
    .hour-bar-wrap {
      height: 200px;
      display: flex;
      flex-direction: column;
      align-items: end;
      justify-content: end;
      border-bottom: 1px solid #e5eaf2;
    }
    .hour-value {
      width: 100%;
      min-height: 18px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      line-height: 1;
      text-align: center;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: 5px;
    }
    .hour-bar {
      width: 100%;
      max-width: 28px;
      min-height: 2px;
      border-radius: 8px 8px 2px 2px;
      background: linear-gradient(180deg, #38bdf8, var(--accent));
      box-shadow: 0 6px 16px rgba(37, 99, 235, 0.22);
    }
    .hour-label {
      color: var(--muted);
      font-size: 11px;
      text-align: center;
      font-variant-numeric: tabular-nums;
      transform: rotate(-40deg);
      transform-origin: center top;
      white-space: nowrap;
    }
    .bar-row { display: grid; grid-template-columns: 120px minmax(90px, 1fr) 98px; gap: 12px; align-items: center; font-size: 12px; }
    .bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }
    .bar-track { height: 11px; background: #edf2f7; border-radius: 99px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.06); }
    .bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), #38bdf8); border-radius: 99px; min-width: 2px; }
    .bar-value { text-align: right; font-variant-numeric: tabular-nums; }
    table { width: 100%; min-width: 560px; border-collapse: collapse; font-size: 12px; }
    #models table,
    #projects table,
    #sessions table,
    #tool-categories table,
    #tool-calls table,
    #skill-invocations table { min-width: 980px; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #eef2f6; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 750; background: rgba(248, 250, 252, 0.92); position: sticky; top: 0; z-index: 1; }
    tbody tr:hover { background: rgba(37, 99, 235, 0.035); }
    tbody tr:last-child td { border-bottom: 0; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    .path { max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .notice { color: var(--muted); font-size: 12px; padding: 10px 14px; border-top: 1px solid var(--line); }
    .hover-tooltip {
      position: fixed;
      z-index: 20;
      pointer-events: none;
      max-width: min(300px, calc(100vw - 24px));
      padding: 10px 12px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.94);
      color: #fff;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-line;
      box-shadow: 0 14px 34px rgba(15, 23, 42, 0.28);
      transform: translate(12px, 12px);
    }
    .hover-tooltip[hidden] { display: none; }
    @media (max-width: 900px) {
      .topbar { align-items: flex-start; flex-direction: column; padding: 14px 0; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      .wide-grid { grid-template-columns: 1fr; }
      .date-popover { left: 0; right: auto; }
      .months { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 86px minmax(70px, 1fr) 80px; }
    }
    @media (max-width: 560px) {
      .wrap { width: min(100vw - 20px, 1280px); }
      .cards { grid-template-columns: 1fr; }
      .controls { width: 100%; justify-content: flex-start; }
      .tabs { width: 100%; }
      button.tab { flex: 1 1 auto; }
      .date-trigger { min-width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div>
        <h1>AI Token 用量看板</h1>
        <div class="sub" id="meta">默认展示当天数据，可自定义日期范围查看用量。</div>
      </div>
      <div class="controls">
        <div class="tabs" id="source-tabs">
          <button class="tab active" data-source="codex">Codex</button>
          <button class="tab" data-source="opencode">OpenCode</button>
          <button class="tab" data-source="claude">Claude Code</button>
          <button class="tab" data-source="hermes">Hermes</button>
          <button class="tab" data-source="all">全部</button>
        </div>
        <div class="date-range" id="date-range">
          <button class="date-trigger" id="date-trigger" type="button">今天</button>
          <div class="date-popover" id="date-popover" hidden>
            <div class="date-popover-head">
              <button id="prev-month" type="button">‹</button>
              <div class="sub" id="range-hint">一次选择起始日期和结束日期后自动刷新</div>
              <button id="next-month" type="button">›</button>
            </div>
            <div class="months" id="months"></div>
          </div>
        </div>
        <button id="refresh" class="primary">刷新</button>
      </div>
    </div>
  </header>
  <main class="wrap">
    <div class="cards" id="cards"></div>
    <div class="grid">
      <section><h2>每日用量</h2><div class="chart" id="daily"></div></section>
      <section>
        <h2>每小时 Token 消耗</h2>
        <div class="chart hourly-chart" id="hourly"></div>
      </section>
      <section><h2>模型分布</h2><div id="models"></div></section>
      <section><h2>项目分布</h2><div id="projects"></div></section>
      <section><h2>高用量会话</h2><div id="sessions"></div></section>
    </div>
    <div class="wide-grid">
      <section><h2>能力分类</h2><div id="tool-categories"></div></section>
      <section><h2>Tool 调用分布</h2><div id="tool-calls"></div></section>
    </div>
    <section><h2>Skill 调用</h2><div id="skill-invocations"></div></section>
  </main>
  <div class="hover-tooltip" id="hover-tooltip" hidden></div>
  <script>
    const nf = new Intl.NumberFormat();
    const usd = new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 4 });
    const token = v => nf.format(v || 0);
    function compactToken(value) {
      const num = Math.abs(value || 0);
      const fmt = n => Number.isInteger(n) ? String(n) : n.toFixed(1).replace(/\.0$/, '');
      if (num >= 1_000_000_000) return `${fmt(num / 1_000_000_000)}b`;
      if (num >= 1_000_000) return `${fmt(num / 1_000_000)}m`;
      if (num >= 1_000) return `${fmt(num / 1_000)}k`;
      return String(num);
    }
    const money = v => usd.format(v || 0);
    let activeSource = '__DEFAULT_SOURCE__';
    let sinceDate = null;
    let untilDate = null;
    let draftStart = null;
    let visibleMonth = null;
    let sourceLabels = { codex: 'Codex', opencode: 'OpenCode', claude: 'Claude Code', hermes: 'Hermes', custom: '自定义', all: '全部' };

    function rowCells(row, columns) {
      return columns.map(([key, label, cls]) => `<td class="${cls || ''}" title="${escapeHtml(String(row[key] ?? ''))}">${escapeHtml(formatValue(key, row[key]))}</td>`).join('');
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, ch => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#039;'}[ch]));
    }

    function formatValue(key, value) {
      if (key === 'cost_usd') return money(value);
      if (typeof value === 'number') return token(value);
      return value ?? '';
    }

    function renderCards(totals, hasPrice) {
      const cards = [
        ['总 Token', token(totals.total_tokens)],
        ['输入', token(totals.input_tokens)],
        ['缓存输入', token(totals.cached_input_tokens)],
        ['输出', token(totals.output_tokens)],
        ['会话数', token(totals.sessions)],
        [hasPrice ? '预估费用' : 'API 请求', hasPrice ? money(totals.cost_usd) : token(totals.api_requests)]
      ];
      document.getElementById('cards').innerHTML = cards.map(([label, value]) => `
        <div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>
      `).join('');
    }

    function renderBars(id, rows, labelKey, limit = 30) {
      const target = document.getElementById(id);
      const max = Math.max(1, ...rows.map(r => r.total_tokens || 0));
      const visibleRows = limit ? rows.slice(-limit) : rows;
      target.innerHTML = visibleRows.map(row => {
        const pct = Math.max(1, Math.round((row.total_tokens || 0) / max * 100));
        return `<div class="bar-row">
          <div class="bar-label" title="${escapeHtml(String(row[labelKey] || ''))}">${escapeHtml(String(row[labelKey] || ''))}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          <div class="bar-value">${token(row.total_tokens)}</div>
        </div>`;
      }).join('') || '<div class="notice">暂无数据</div>';
    }

    function hourOfDayLabel(hour) {
      const match = String(hour || '').match(/(?:^| )(\d{2}):00$/);
      return match ? `${match[1]}:00` : String(hour || '');
    }

    function nextHourOfDayLabel(label) {
      const hour = Number(String(label || '').slice(0, 2));
      if (!Number.isFinite(hour)) return '';
      return `${String((hour + 1) % 24).padStart(2, '0')}:00`;
    }

    function aggregateByHourOfDay(rows) {
      const buckets = Array.from({ length: 24 }, (_, hour) => ({
        hour: `${String(hour).padStart(2, '0')}:00`,
        input_tokens: 0,
        cached_input_tokens: 0,
        output_tokens: 0,
        reasoning_output_tokens: 0,
        total_tokens: 0,
        api_requests: 0,
        usage_records: 0,
        session_ids: new Set(),
        sessions: 0,
      }));
      for (const row of rows || []) {
        const label = hourOfDayLabel(row.hour);
        const hour = Number(label.slice(0, 2));
        if (!Number.isFinite(hour) || hour < 0 || hour > 23) continue;
        for (const key of ['input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens', 'total_tokens', 'api_requests']) {
          buckets[hour][key] += row[key] || 0;
        }
        if (row.session_id) {
          buckets[hour].session_ids.add(row.session_id);
          buckets[hour].usage_records += 1;
        } else {
          buckets[hour].sessions += row.sessions || 0;
          buckets[hour].usage_records += row.usage_records || 0;
        }
      }
      for (const bucket of buckets) {
        if (bucket.session_ids.size) bucket.sessions = bucket.session_ids.size;
        delete bucket.session_ids;
      }
      return buckets;
    }

    function renderHourAxisChart(rows) {
      const target = document.getElementById('hourly');
      const max = Math.max(1, ...rows.map(row => row.total_tokens || 0));
      target.innerHTML = `<div class="hour-axis-chart">${rows.map(row => {
        const height = Math.max(2, Math.round((row.total_tokens || 0) / max * 180));
        const title = `${row.hour} - ${nextHourOfDayLabel(row.hour)}\n总量：${token(row.total_tokens)}\n会话数：${token(row.sessions)}\n用量记录：${token(row.usage_records)}\nAPI 请求：${token(row.api_requests)}\n输入：${token(row.input_tokens)}\n缓存输入：${token(row.cached_input_tokens)}\n输出：${token(row.output_tokens)}\n推理输出：${token(row.reasoning_output_tokens)}`;
        return `<div class="hour-column" data-tooltip="${escapeHtml(title)}">
          <div class="hour-bar-wrap"><div class="hour-value">${row.total_tokens ? compactToken(row.total_tokens) : ''}</div><div class="hour-bar" style="height:${height}px"></div></div>
          <div class="hour-label">${row.hour}</div>
        </div>`;
      }).join('')}</div>`;
      attachHourTooltips(target);
    }

    function moveTooltip(event) {
      const tooltip = document.getElementById('hover-tooltip');
      const margin = 14;
      tooltip.hidden = false;
      const rect = tooltip.getBoundingClientRect();
      let left = event.clientX + 14;
      let top = event.clientY + 14;
      if (left + rect.width + margin > window.innerWidth) left = event.clientX - rect.width - 14;
      if (top + rect.height + margin > window.innerHeight) top = event.clientY - rect.height - 14;
      tooltip.style.left = `${Math.max(margin, left)}px`;
      tooltip.style.top = `${Math.max(margin, top)}px`;
    }

    function attachHourTooltips(container) {
      const tooltip = document.getElementById('hover-tooltip');
      container.querySelectorAll('.hour-column[data-tooltip]').forEach(column => {
        column.addEventListener('mouseenter', event => {
          tooltip.textContent = column.dataset.tooltip || '';
          moveTooltip(event);
        });
        column.addEventListener('mousemove', moveTooltip);
        column.addEventListener('mouseleave', () => {
          tooltip.hidden = true;
        });
      });
    }

    function renderHourly(rows) {
      const hourly = aggregateByHourOfDay(rows);
      renderHourAxisChart(hourly);
    }

    function sourceLabel(source) {
      return sourceLabels[source] || source;
    }

    function renderSourceTabs(sources) {
      if (!Array.isArray(sources) || sources.length === 0) return;
      const tabs = document.getElementById('source-tabs');
      const existing = Array.from(tabs.querySelectorAll('button')).map(button => button.dataset.source).join('|');
      const next = sources.map(source => source.key).join('|');
      sources.forEach(source => { sourceLabels[source.key] = source.label; });
      if (existing === next) return;
      tabs.innerHTML = sources.map(source => `<button class="tab ${source.key === activeSource ? 'active' : ''}" data-source="${escapeHtml(source.key)}">${escapeHtml(source.label)}</button>`).join('');
      tabs.querySelectorAll('button').forEach(button => {
        button.addEventListener('click', () => {
          activeSource = button.dataset.source;
          tabs.querySelectorAll('button').forEach(item => item.classList.toggle('active', item === button));
          load();
        });
      });
    }

    function localISODate(date = new Date()) {
      const offset = date.getTimezoneOffset() * 60000;
      return new Date(date.getTime() - offset).toISOString().slice(0, 10);
    }

    function parseISODate(value) {
      const [year, month, day] = value.split('-').map(Number);
      return new Date(year, month - 1, day);
    }

    function monthStart(date) {
      return new Date(date.getFullYear(), date.getMonth(), 1);
    }

    function addMonths(date, count) {
      return new Date(date.getFullYear(), date.getMonth() + count, 1);
    }

    function sameDate(a, b) {
      return a && b && localISODate(a) === localISODate(b);
    }

    function activeRange() {
      if (draftStart) return { start: draftStart, end: draftStart };
      if (!sinceDate || !untilDate) return null;
      return { start: sinceDate, end: untilDate };
    }

    function inSelectedRange(date) {
      const range = activeRange();
      if (!range) return false;
      return date >= range.start && date <= range.end;
    }

    function updateDateTrigger() {
      const label = sinceDate && untilDate
        ? `${localISODate(sinceDate)} 至 ${localISODate(untilDate)}`
        : '选择日期范围';
      document.getElementById('date-trigger').textContent = label;
    }

    function renderCalendarMonth(date) {
      const year = date.getFullYear();
      const month = date.getMonth();
      const title = date.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long' });
      const first = new Date(year, month, 1);
      const daysInMonth = new Date(year, month + 1, 0).getDate();
      const leading = first.getDay();
      const weekdays = ['日', '一', '二', '三', '四', '五', '六'];
      let cells = weekdays.map(day => `<div class="dow">${day}</div>`).join('');
      for (let i = 0; i < leading; i += 1) cells += '<div></div>';
      for (let day = 1; day <= daysInMonth; day += 1) {
        const cellDate = new Date(year, month, day);
        const iso = localISODate(cellDate);
        const classes = ['day'];
        const range = activeRange();
        if (range && (sameDate(cellDate, range.start) || sameDate(cellDate, range.end))) classes.push('selected');
        if (inSelectedRange(cellDate)) classes.push('in-range');
        cells += `<button type="button" class="${classes.join(' ')}" data-date="${iso}">${day}</button>`;
      }
      return `<div class="month"><div class="month-title">${escapeHtml(title)}</div><div class="calendar-grid">${cells}</div></div>`;
    }

    function renderCalendars() {
      const months = document.getElementById('months');
      months.innerHTML = renderCalendarMonth(visibleMonth) + renderCalendarMonth(addMonths(visibleMonth, 1));
      months.querySelectorAll('[data-date]').forEach(button => {
        button.addEventListener('click', event => {
          event.stopPropagation();
          selectDate(parseISODate(button.dataset.date));
        });
      });
      document.getElementById('range-hint').textContent = draftStart
        ? `开始：${localISODate(draftStart)} · 请选择结束日期`
        : '默认当天；可一次选择起始日期和结束日期';
    }

    function openDatePicker() {
      document.getElementById('date-popover').hidden = false;
      draftStart = null;
      visibleMonth = monthStart(sinceDate || new Date());
      renderCalendars();
    }

    function closeDatePicker() {
      document.getElementById('date-popover').hidden = true;
      draftStart = null;
    }

    function selectDate(date) {
      if (!draftStart) {
        draftStart = date;
        renderCalendars();
        return;
      }
      if (date < draftStart) {
        sinceDate = date;
        untilDate = draftStart;
      } else {
        sinceDate = draftStart;
        untilDate = date;
      }
      updateDateTrigger();
      closeDatePicker();
      load();
    }

    function renderTable(id, rows, columns, limit = 20) {
      if (!rows || rows.length === 0) {
        document.getElementById(id).innerHTML = '<div class="notice">暂无数据</div>';
        return;
      }
      const header = columns.map(([key, label, cls]) => `<th class="${cls || ''}">${label}</th>`).join('');
      const body = rows.slice(0, limit).map(row => `<tr>${rowCells(row, columns)}</tr>`).join('');
      document.getElementById(id).innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
    }

    async function load() {
      const params = new URLSearchParams({ source: activeSource });
      if (sinceDate) params.set('since', localISODate(sinceDate));
      if (untilDate) params.set('until', localISODate(untilDate));
      const qs = `?${params.toString()}`;
      const res = await fetch(`/api/summary${qs}`, { cache: 'no-store' });
      const data = await res.json();
      renderSourceTabs(data.available_sources);
      renderCards(data.totals, data.has_price_config);
      renderBars('daily', data.by_day, 'date');
      renderHourly(data.events || data.by_hour || []);
      renderTable('models', data.by_model.sort((a, b) => b.total_tokens - a.total_tokens), [
        ['tool', '来源'], ['model', '模型'], ['sessions', '会话', 'num'], ['api_requests', 'API 请求', 'num'], ['total_tokens', '总量', 'num'], ['output_tokens', '输出', 'num']
      ]);
      renderTable('projects', data.by_project.sort((a, b) => b.total_tokens - a.total_tokens), [
        ['tool', '来源'], ['project', '项目'], ['sessions', '会话', 'num'], ['api_requests', 'API 请求', 'num'], ['total_tokens', '总量', 'num'], ['cwd', '路径', 'path']
      ]);
      renderTable('tool-categories', data.by_tool_category || data.by_skill || [], [
        ['source_tool', '来源'], ['skill', '能力分类'], ['calls', '调用', 'num'], ['api_requests', 'API 请求', 'num'], ['total_tokens', '总量', 'num'], ['input_tokens', '输入', 'num'], ['output_tokens', '输出', 'num']
      ]);
      renderTable('tool-calls', data.by_tool_call || [], [
        ['source_tool', '来源'], ['tool_name', 'Tool'], ['skill', '能力分类'], ['calls', '调用', 'num'], ['api_requests', 'API 请求', 'num'], ['total_tokens', '总量', 'num']
      ]);
      renderTable('skill-invocations', data.by_skill_invocation || [], [
        ['source_tool', '来源'], ['skill_name', 'Skill'], ['skill_source', '来源类型'], ['plugin_name', '插件'], ['invocation_type', '调用类型'], ['calls', '调用', 'num'], ['api_requests', 'API 请求', 'num'], ['sessions', '关联会话数', 'num']
      ]);
      const sessionColumns = [
        ['date', '日期'], ['tool', '来源'], ['project', '项目'], ['model', '模型'], ['api_requests', 'API 请求', 'num'], ['total_tokens', '总量', 'num']
      ];
      if (data.has_price_config) sessionColumns.push(['cost_usd', '费用', 'num']);
      renderTable('sessions', data.sessions, sessionColumns, 15);
      document.getElementById('meta').textContent = `${sourceLabel(data.source)} · ${token(data.events.length)} 条用量记录 · ${token(data.totals.sessions)} 个会话 · ${new Date().toLocaleTimeString('zh-CN')} 已刷新`;
    }

    document.getElementById('refresh').addEventListener('click', load);
    document.getElementById('date-trigger').addEventListener('click', event => {
      event.stopPropagation();
      const popover = document.getElementById('date-popover');
      if (popover.hidden) openDatePicker();
      else closeDatePicker();
    });
    document.getElementById('date-popover').addEventListener('click', event => {
      event.stopPropagation();
    });
    document.getElementById('prev-month').addEventListener('click', event => {
      event.stopPropagation();
      visibleMonth = addMonths(visibleMonth, -1);
      renderCalendars();
    });
    document.getElementById('next-month').addEventListener('click', event => {
      event.stopPropagation();
      visibleMonth = addMonths(visibleMonth, 1);
      renderCalendars();
    });
    document.addEventListener('click', event => {
      if (!document.getElementById('date-range').contains(event.target)) closeDatePicker();
    });
    renderSourceTabs([
      { key: 'codex', label: 'Codex' },
      { key: 'opencode', label: 'OpenCode' },
      { key: 'claude', label: 'Claude Code' },
      { key: 'hermes', label: 'Hermes' },
      { key: 'all', label: '全部' },
    ]);
    const today = localISODate();
    sinceDate = parseISODate(today);
    untilDate = parseISODate(today);
    visibleMonth = monthStart(sinceDate);
    updateDateTrigger();
    document.querySelectorAll('#source-tabs button').forEach(item => item.classList.toggle('active', item.dataset.source === activeSource));
    load();
  </script>
</body>
</html>
"""


def serve_dashboard(
    host: str,
    port: int,
    codex_paths: list[Path],
    opencode_paths: list[Path],
    claude_paths: list[Path],
    hermes_paths: list[Path],
    local_tz: ZoneInfo,
    default_days: int | None,
    default_since: str | None,
    default_until: str | None,
    default_source: str,
    prices: dict[str, dict[str, float]],
    custom_sources: list[dict[str, Any]],
) -> None:
    source_keys = {"codex", "opencode", "claude", "hermes", "custom", "all"}
    for config in custom_sources:
        source_keys.add(config["name"])
        source_keys.add(custom_source_key(config["name"]))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

        def send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.replace("__DEFAULT_SOURCE__", default_source).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            if parsed.path == "/api/summary":
                query = parse_qs(parsed.query)
                days = default_days
                if query.get("days", [""])[0]:
                    try:
                        days = int(query["days"][0])
                    except ValueError:
                        days = default_days
                since = query.get("since", [default_since or ""])[0] or default_since
                until = query.get("until", [default_until or ""])[0] or default_until
                source = query.get("source", [default_source])[0]
                if source not in source_keys:
                    source = default_source
                events = load_events(source, codex_paths, opencode_paths, claude_paths, hermes_paths, local_tz, custom_sources)
                events = filter_events(events, days, since, until)
                tool_events = load_tool_events(source, codex_paths, opencode_paths, claude_paths, hermes_paths, local_tz)
                tool_events = filter_tool_events(tool_events, days, since, until)
                skill_events = load_skill_events(source, claude_paths, hermes_paths, local_tz, (event.cwd for event in events))
                skill_events = filter_skill_events(skill_events, days, since, until)
                self.send_json(summary_payload(events, prices, source, tool_events, skill_events, custom_sources))
                return

            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving AI token dashboard at http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize local Codex, OpenCode, Claude Code, and Hermes token usage."
    )
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        default=None,
        help="Alias for --codex-path. Kept for compatibility.",
    )
    parser.add_argument(
        "--codex-path",
        action="append",
        type=Path,
        default=None,
        help="Codex log root/file to scan. Defaults to common per-user locations for macOS/Linux/Windows.",
    )
    parser.add_argument(
        "--opencode-path",
        action="append",
        type=Path,
        default=None,
        help="OpenCode data directory or existing local storage file. Defaults to common per-user locations for macOS/Linux/Windows.",
    )
    parser.add_argument(
        "--claude-path",
        action="append",
        type=Path,
        default=None,
        help="Claude Code log root/file to scan. Defaults to common per-user locations for macOS/Linux/Windows.",
    )
    parser.add_argument(
        "--hermes-path",
        action="append",
        type=Path,
        default=None,
        help="Hermes home/profile directory or existing local storage file. Defaults to common per-user locations for macOS/Linux/Windows.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Usage source to show: codex, opencode, claude, hermes, custom, all, or custom:<name>. Default: settings value or codex.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only show the last N local dates present in the logs.",
    )
    parser.add_argument("--since", help="Only include local dates >= YYYY-MM-DD.")
    parser.add_argument("--until", help="Only include local dates <= YYYY-MM-DD.")
    parser.add_argument(
        "--timezone",
        default=None,
        help="Local timezone for daily/hourly grouping. Default: settings value or Asia/Shanghai.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local web dashboard instead of printing a terminal report.",
    )
    parser.add_argument("--host", default=None, help="Dashboard host. Default: settings value or 127.0.0.1.")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port. Default: settings value or 8765.")
    parser.add_argument(
        "--price-config",
        type=Path,
        help="Optional JSON model price config for estimated API-equivalent cost.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional AI Token Usage settings JSON file. Defaults to AI_TOKEN_USAGE_CONFIG, ./ai-token-usage.json, ~/.config/ai-token-usage/config.json, or ~/.ai-token-usage.json if present.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Print diagnostics for settings, discovered data paths, and source event counts.",
    )
    parser.add_argument("--version", action="version", version=f"AI Token Usage {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings_file = resolve_settings_path(args.config)
    settings = load_settings(settings_file)

    if args.days is not None and args.days < 1:
        parser.error("--days must be >= 1")

    custom_sources = custom_source_configs(settings)
    source = args.source or str(settings.get("source") or "codex")
    source_keys = {"codex", "opencode", "claude", "hermes", "custom", "all"}
    for config in custom_sources:
        source_keys.add(config["name"])
        source_keys.add(custom_source_key(config["name"]))
    if source not in source_keys:
        parser.error("source must be one of: codex, opencode, claude, hermes, custom, all, or custom:<name> from custom_sources")

    timezone_name = args.timezone or str(settings.get("timezone") or "Asia/Shanghai")
    host = args.host or str(settings.get("host") or "127.0.0.1")
    port = args.port if args.port is not None else safe_int(settings.get("port") or 8765)
    if port < 1 or port > 65535:
        parser.error("port must be between 1 and 65535")

    try:
        local_tz = ZoneInfo(timezone_name)
    except Exception as exc:
        parser.error(f"invalid timezone {timezone_name!r}: {exc}")

    codex_paths = args.codex_path or args.path or settings_paths(settings, "codex_paths", "codex_path") or default_codex_paths()
    opencode_paths = args.opencode_path or settings_paths(settings, "opencode_paths", "opencode_path") or default_opencode_paths()
    claude_paths = args.claude_path or settings_paths(settings, "claude_paths", "claude_path") or default_claude_paths()
    hermes_paths = args.hermes_path or settings_paths(settings, "hermes_paths", "hermes_path") or default_hermes_paths()
    price_config = args.price_config or settings_path(settings, "price_config")
    prices = load_price_config(price_config)

    if args.doctor:
        output_doctor(
            settings_file,
            settings,
            source,
            local_tz,
            codex_paths,
            opencode_paths,
            claude_paths,
            hermes_paths,
            price_config,
            custom_sources,
        )
        return 0

    if args.serve:
        serve_dashboard(
            host,
            port,
            codex_paths,
            opencode_paths,
            claude_paths,
            hermes_paths,
            local_tz,
            args.days,
            args.since,
            args.until,
            source,
            prices,
            custom_sources,
        )
        return 0

    events = load_events(source, codex_paths, opencode_paths, claude_paths, hermes_paths, local_tz, custom_sources)
    events = filter_events(events, args.days, args.since, args.until)
    tool_events = load_tool_events(source, codex_paths, opencode_paths, claude_paths, hermes_paths, local_tz)
    tool_events = filter_tool_events(tool_events, args.days, args.since, args.until)
    skill_events = load_skill_events(source, claude_paths, hermes_paths, local_tz, (event.cwd for event in events))
    skill_events = filter_skill_events(skill_events, args.days, args.since, args.until)

    if args.format == "json":
        output_json(events, prices, source, tool_events, skill_events, custom_sources)
    elif args.format == "csv":
        output_csv(events)
    else:
        output_text(events, prices, tool_events, skill_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
