import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import ai_token_usage as usage


UTC = ZoneInfo("UTC")


def millis(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def seconds(value: str) -> float:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


class UsageLogicTests(unittest.TestCase):
    def test_codex_cumulative_token_counts_are_delta_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            rows = [
                {"type": "turn_context", "payload": {"model": "gpt-test", "cwd": "/repo/demo"}},
                {
                    "timestamp": "2026-05-31T13:00:00Z",
                    "payload": {
                        "type": "token_count",
                        "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}},
                    },
                },
                {
                    "timestamp": "2026-05-31T13:10:00Z",
                    "payload": {
                        "type": "token_count",
                        "info": {"total_token_usage": {"input_tokens": 150, "output_tokens": 45, "total_tokens": 195}},
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            events = list(usage.parse_codex_log(path, UTC))

        self.assertEqual([event.total_tokens for event in events], [120, 75])
        self.assertEqual(events[1].input_tokens, 50)
        self.assertEqual(events[1].output_tokens, 25)

    def test_codex_state_sqlite_threads_are_counted_when_jsonl_has_no_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state_5.sqlite"
            connection = sqlite3.connect(db)
            connection.execute(
                """
                CREATE TABLE threads (
                    id TEXT,
                    updated_at INTEGER,
                    updated_at_ms INTEGER,
                    created_at INTEGER,
                    created_at_ms INTEGER,
                    model_provider TEXT,
                    model TEXT,
                    cwd TEXT,
                    tokens_used INTEGER,
                    rollout_path TEXT
                )
                """
            )
            connection.execute(
                "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "c1",
                    seconds("2026-05-31T13:30:00"),
                    None,
                    seconds("2026-05-31T13:00:00"),
                    None,
                    "openai",
                    "gpt-test",
                    "/repo/codex",
                    1234,
                    "/repo/codex/session.jsonl",
                ),
            )
            connection.commit()
            connection.close()

            events = usage.load_codex_state_events([Path(tmp)], UTC)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].session_id, "c1")
        self.assertEqual(events[0].model, "openai/gpt-test")
        self.assertEqual(events[0].hour, "2026-05-31 13:00")
        self.assertEqual(events[0].total_tokens, 1234)

    def test_opencode_usage_is_attributed_to_message_time_not_session_update_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "opencode.db"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE session (id TEXT, directory TEXT, model TEXT)")
            connection.execute("CREATE TABLE message (id TEXT, session_id TEXT, data TEXT, time_updated INTEGER)")
            session_model = json.dumps({"providerID": "fallback", "id": "fallback-model"})
            connection.execute("INSERT INTO session VALUES (?, ?, ?)", ("ses_1", "/repo/demo", session_model))
            messages = [
                (
                    "msg_1",
                    "ses_1",
                    {
                        "role": "assistant",
                        "time": {"completed": millis("2026-05-31T13:05:00")},
                        "modelID": "model-a",
                        "providerID": "provider-a",
                        "path": {"cwd": "/repo/demo"},
                        "tokens": {"input": 100, "cache": {"read": 30, "write": 5}, "output": 20, "reasoning": 7, "total": 162},
                    },
                ),
                (
                    "msg_2",
                    "ses_1",
                    {
                        "role": "assistant",
                        "time": {"completed": millis("2026-05-31T21:15:00")},
                        "modelID": "model-a",
                        "providerID": "provider-a",
                        "path": {"cwd": "/repo/demo"},
                        "tokens": {"input": 200, "cache": {"read": 40, "write": 10}, "output": 30, "reasoning": 0, "total": 280},
                    },
                ),
            ]
            for message_id, session_id, data in messages:
                connection.execute(
                    "INSERT INTO message VALUES (?, ?, ?, ?)",
                    (message_id, session_id, json.dumps(data), millis("2026-05-31T23:59:00")),
                )
            connection.commit()
            connection.close()

            events = list(usage.parse_opencode_db(db, UTC))

        self.assertEqual(len(events), 2)
        self.assertEqual([event.hour for event in events], ["2026-05-31 13:00", "2026-05-31 21:00"])
        self.assertEqual([event.total_tokens for event in events], [162, 280])
        self.assertEqual(events[0].cached_input_tokens, 35)
        self.assertEqual(events[0].api_requests, 1)

    def test_hermes_session_totals_are_session_time_attributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.db"
            connection = sqlite3.connect(db)
            connection.execute(
                """
                CREATE TABLE sessions (
                    id TEXT,
                    source TEXT,
                    model TEXT,
                    started_at REAL,
                    ended_at REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cache_read_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    api_call_count INTEGER
                )
                """
            )
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("h1", "hermes", "model-h", seconds("2026-05-31T12:00:00"), seconds("2026-05-31T14:00:00"), 100, 20, 30, 5, 2, 3),
            )
            connection.commit()
            connection.close()

            events = list(usage.parse_hermes_db(db, UTC))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].hour, "2026-05-31 14:00")
        self.assertEqual(events[0].total_tokens, 157)
        self.assertEqual(events[0].api_requests, 3)

    def test_hermes_tool_and_skill_events_can_use_json_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.db"
            connection = sqlite3.connect(db)
            connection.execute(
                """
                CREATE TABLE sessions (
                    id TEXT,
                    source TEXT,
                    model TEXT,
                    started_at REAL,
                    ended_at REAL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE messages (
                    id TEXT,
                    session_id TEXT,
                    timestamp REAL,
                    data TEXT,
                    tool_calls TEXT
                )
                """
            )
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                ("h1", "hermes", "model-h", seconds("2026-05-31T12:00:00"), seconds("2026-05-31T14:00:00")),
            )
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                (
                    "m1",
                    "h1",
                    seconds("2026-05-31T12:30:00"),
                    json.dumps({"tool_name": "skill", "content": "/home/me/.hermes/skills/code-review/SKILL.md"}),
                    None,
                ),
            )
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                ("m2", "h1", seconds("2026-05-31T12:40:00"), json.dumps({"toolName": "bash"}), None),
            )
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                (
                    "m3",
                    "h1",
                    seconds("2026-05-31T12:50:00"),
                    None,
                    json.dumps(
                        [
                            {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}},
                            {"id": "call_2", "function": {"name": "skill_view", "arguments": json.dumps({"name": "debug"})}},
                        ]
                    ),
                ),
            )
            connection.commit()
            connection.close()

            tool_events = usage.load_hermes_tool_events_from_db(db, UTC)
            skill_events = usage.load_hermes_skill_events_from_db(db, UTC)

        self.assertEqual([event.tool_name for event in tool_events], ["skill", "bash", "terminal", "skill_view"])
        self.assertEqual(tool_events[1].skill, "命令执行")
        self.assertEqual(len(skill_events), 2)
        self.assertEqual(skill_events[0].skill_name, "code-review")
        self.assertEqual(skill_events[1].skill_name, "debug")
        self.assertEqual(skill_events[0].source_tool, "hermes")

    def test_hermes_plain_assistant_messages_are_not_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.db"
            connection = sqlite3.connect(db)
            connection.execute(
                """
                CREATE TABLE messages (
                    id TEXT,
                    session_id TEXT,
                    timestamp REAL,
                    role TEXT,
                    data TEXT
                )
                """
            )
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                ("m1", "h1", seconds("2026-05-31T12:30:00"), "assistant", json.dumps({"content": "hello"})),
            )
            connection.commit()
            connection.close()

            tool_events = usage.load_hermes_tool_events_from_db(db, UTC)
            skill_events = usage.load_hermes_skill_events_from_db(db, UTC)

        self.assertEqual(tool_events, [])
        self.assertEqual(skill_events, [])

    def test_settings_paths_accept_string_or_list(self) -> None:
        self.assertEqual(usage.settings_paths({"hermes_path": "~/.hermes"}, "hermes_paths", "hermes_path"), [Path("~/.hermes").expanduser()])
        self.assertEqual(
            usage.settings_paths({"hermes_paths": ["~/.hermes", "~/custom"]}, "hermes_paths", "hermes_path"),
            [Path("~/.hermes").expanduser(), Path("~/custom").expanduser()],
        )

    def test_custom_jsonl_source_uses_mapping_and_dynamic_source_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "qclaw.jsonl"
            log.write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-31T13:30:00Z",
                        "conversation_id": "q1",
                        "model_id": "qclaw-large",
                        "project_path": "/repo/qclaw",
                        "usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 30, "total_tokens": 150},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            settings = {
                "custom_sources": [
                    {
                        "name": "qclaw",
                        "label": "腾讯 QClaw",
                        "paths": [str(log)],
                        "mapping": {
                            "timestamp": "created_at",
                            "input_tokens": "usage.input_tokens",
                            "cached_input_tokens": "usage.cached_input_tokens",
                            "output_tokens": "usage.output_tokens",
                            "total_tokens": "usage.total_tokens",
                            "session_id": "conversation_id",
                            "model": "model_id",
                            "cwd": "project_path",
                        },
                    }
                ]
            }
            custom_sources = usage.custom_source_configs(settings)
            events = usage.load_events("custom:qclaw", [], [], [], [], UTC, custom_sources)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].tool, "qclaw")
        self.assertEqual(events[0].hour, "2026-05-31 13:00")
        self.assertEqual(events[0].total_tokens, 150)
        self.assertEqual(events[0].cached_input_tokens, 20)
        self.assertEqual(events[0].session_id, "q1")


if __name__ == "__main__":
    unittest.main()
