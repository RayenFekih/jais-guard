"""Opt-in TypeSafe HTTP evaluation with durable per-attempt checkpoints; no auto-paid calls."""

import asyncio
import fcntl
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from .test_workflow import read_frozen_test, sha, write_new

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
RETRYABLE = {408, 429, 500, 502, 503, 504, 529}
FATAL = {400, 401, 402, 403, 404, 413, 415, 422}
PINNED = re.compile(r"jev-\d+\.\d+\.\d+\Z")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def load_question(path):
    question = json.loads(Path(path).read_text())
    if (
        question["endpoint"] != ENDPOINT
        or question["question_type"] != "noul"
        or question["yes_threshold"] != 0.5
        or question["yes_label"] != "VALID"
    ):
        raise ValueError("Expected the fixed noul/VALID >= 0.5 benchmark protocol")
    if not question["instructions"].strip() or not question["question_id"]:
        raise ValueError("Missing fixed question")
    return question


def payload(text, model, question):
    return {
        "model": model,
        "state": text,
        "questions": {
            question["question_id"]: {"type": "noul", "instructions": question["instructions"]}
        },
    }


def frozen_rows(path):
    rows = read_frozen_test(path)
    if len(rows) != 400 or Counter(r["label"] for r in rows) != {"VALID": 188, "GIBBERISH": 212}:
        raise ValueError("Expected the unchanged human-reviewed 400-example test")
    return rows


def validate_response(data, question_id, secret=""):
    """Return only documented fields. Never persist raw headers/error bodies or credentials."""
    if not isinstance(data, dict):
        raise ValueError("response_object")
    model = data.get("model")
    if (
        not isinstance(model, str)
        or not re.fullmatch(r"jev-[A-Za-z0-9._-]{1,80}", model)
        or secret
        and secret in model
    ):
        raise ValueError("response_model")
    answers = data.get("answers")
    if not isinstance(answers, dict) or set(answers) != {question_id}:
        raise ValueError("response_answers")
    answer = answers[question_id]
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise ValueError("response_answer_type")
    probability = answer.get("noul")
    if (
        type(probability) not in (int, float)
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("response_probability")
    usage = safe_usage(data)
    if usage is None:
        raise ValueError("response_usage")
    return {
        "model": model,
        "answers": {question_id: {"type": "noul", "noul": float(probability)}},
        "usage": usage,
    }


def safe_usage(data):
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict) or any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")
    ):
        return None
    return {k: usage[k] for k in ("input_tokens", "output_tokens")}


def retry_delay(headers, number, base, cap):
    delay = base * 2**number
    try:
        if "retry-after-ms" in headers:
            delay = max(delay, float(headers["retry-after-ms"]) / 1000)
        elif "retry-after" in headers:
            value = headers["retry-after"]
            try:
                server = float(value)
            except ValueError:
                server = parsedate_to_datetime(value).timestamp() - time.time()
            delay = max(delay, server)
    except (ValueError, TypeError, OverflowError):
        pass
    if not math.isfinite(delay) or delay > cap:
        return None  # Pause rather than violate a server wait longer than the configured bound.
    return max(0.0, delay)


@contextmanager
def runner_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another runner is using this result directory") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def manifest_config(stage, rows, question, model, settings, test=None, comparison=None):
    config = {
        "stage": stage,
        "model": model,
        "question": question,
        "rows_sha256": digest(rows),
        "examples": len(rows),
        "execution": settings,
        "threshold_policy": "p_yes >= 0.5 => VALID (0); otherwise GIBBERISH (1)",
        "test": str(Path(test).resolve()) if test else None,
        "comparison": str(Path(comparison).resolve()) if comparison else None,
    }
    if test:
        config["test_files"] = {
            str(Path(test).resolve() / n): sha(Path(test) / n)
            for n in ("test.jsonl", "manifest.json", "freeze.json")
        }
    return config


def initialize(directory, config, rows):
    directory = Path(directory)
    manifest = directory / "manifest.json"
    dbpath = directory / "requests.sqlite3"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["config"] != config:
            raise ValueError(
                "Run protocol/model/question/data/settings changed; use a new versioned directory"
            )
        if not dbpath.exists():
            raise ValueError("Run manifest exists without checkpoint DB; refusing reset")
    else:
        if dbpath.exists():
            raise ValueError("Checkpoint DB exists without manifest; refusing reset")
        write_new(
            manifest, {"created_at": now(), "config": config, "protocol_sha256": digest(config)}
        )
    db = sqlite3.connect(dbpath)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS examples(
          id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, row_json TEXT NOT NULL,
          status TEXT NOT NULL, response_json TEXT, last_error TEXT, total_ms REAL);
        CREATE TABLE IF NOT EXISTS attempts(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL,
          number INTEGER NOT NULL, started_at TEXT NOT NULL, ended_at TEXT,
          status_code INTEGER, error_type TEXT, latency_ms REAL, response_json TEXT,
          usage_json TEXT, UNIQUE(id,number));
    """)
    expected = sha(manifest)
    saved = db.execute("SELECT value FROM metadata WHERE key='manifest_sha256'").fetchone()
    if saved is None:
        if db.execute("SELECT COUNT(*) FROM examples").fetchone()[0]:
            db.close()
            raise ValueError("Unverified existing checkpoint state")
        with db:
            db.execute("INSERT INTO metadata VALUES ('manifest_sha256',?)", (expected,))
            db.executemany(
                "INSERT INTO examples VALUES (?,?,?,'pending',NULL,NULL,NULL)",
                [(r["id"], i, json.dumps(r, ensure_ascii=False)) for i, r in enumerate(rows)],
            )
    elif saved[0] != expected:
        db.close()
        raise ValueError("Run manifest was modified")
    stored = [
        json.loads(r[0]) for r in db.execute("SELECT row_json FROM examples ORDER BY ordinal")
    ]
    if stored != rows:
        db.close()
        raise ValueError("Checkpoint inputs differ from immutable dataset")
    return db


def read_ledger(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(manifest["config"]) != manifest["protocol_sha256"]:
        raise ValueError("Run protocol hash mismatch")
    with sqlite3.connect(
        f"file:{(directory / 'requests.sqlite3').resolve()}?mode=ro", uri=True
    ) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        fingerprint = db.execute(
            "SELECT value FROM metadata WHERE key='manifest_sha256'"
        ).fetchone()
        if fingerprint is None or fingerprint[0] != sha(directory / "manifest.json"):
            raise ValueError("Checkpoint manifest hash mismatch")
        examples = [dict(r) for r in db.execute("SELECT * FROM examples ORDER BY ordinal")]
        attempts = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY sequence")]
    if digest([json.loads(r["row_json"]) for r in examples]) != manifest["config"]["rows_sha256"]:
        raise ValueError("Checkpoint inputs changed")
    return manifest, examples, attempts


async def execute(
    directory,
    config,
    rows,
    api_key,
    *,
    retry_failed=False,
    retry_uncertain=False,
    transport=None,
    sleep=asyncio.sleep,
):
    """The caller must explicitly authorize paid execution. Tests inject MockTransport."""
    with runner_lock(directory):
        db = initialize(directory, config, rows)
        settings = config["execution"]
        stop = asyncio.Event()
        try:
            with db:
                # A previous process may have completed server-side without saving the response.
                db.execute(
                    "UPDATE examples SET "
                    "status='uncertain',last_error='interrupted_in_flight' WHERE "
                    "status='in_flight'"
                )
                db.execute("UPDATE examples SET status='pending' WHERE status='retrying'")
            statuses = {"pending"}
            if retry_failed:
                statuses.add("failed")
            if retry_uncertain:
                statuses.add("uncertain")
            queue = asyncio.Queue()
            for record in db.execute("SELECT * FROM examples ORDER BY ordinal"):
                if record["status"] in statuses:
                    queue.put_nowait(dict(record))
            timeout = httpx.Timeout(settings["timeout_seconds"])
            # No redirects, proxy credentials, SDK autologging or request/error-body dumps.
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
                transport=transport,
                limits=httpx.Limits(
                    max_connections=settings["concurrency"],
                    max_keepalive_connections=settings["concurrency"],
                ),
            ) as client:

                async def one(record):
                    item = json.loads(record["row_json"])
                    started = time.perf_counter()
                    for retry in range(settings["max_retries"] + 1):
                        if stop.is_set():
                            return
                        number = (
                            db.execute(
                                "SELECT COUNT(*) FROM attempts WHERE id=?", (item["id"],)
                            ).fetchone()[0]
                            + 1
                        )
                        with db:
                            cursor = db.execute(
                                "INSERT INTO attempts(id,number,started_at,error_type) VALUES "
                                "(?,?,?,'in_flight')",
                                (item["id"], number, now()),
                            )
                            sequence = cursor.lastrowid
                            db.execute(
                                "UPDATE examples SET status='in_flight' WHERE id=?", (item["id"],)
                            )
                        tick = time.perf_counter()
                        code, safe, usage, error, retryable, headers = (
                            None,
                            None,
                            None,
                            None,
                            False,
                            {},
                        )
                        try:
                            response = await client.post(
                                ENDPOINT,
                                json=payload(
                                    item["input_text"], config["model"], config["question"]
                                ),
                            )
                            elapsed = (time.perf_counter() - tick) * 1000
                            code, headers = response.status_code, response.headers
                            try:
                                data = (
                                    response.json()
                                    if len(response.content) <= 1024 * 1024
                                    else None
                                )
                            except ValueError:
                                data = None
                            usage = safe_usage(data)
                            if not 200 <= code < 300:
                                error, retryable = "http_error", code in RETRYABLE
                            else:
                                try:
                                    safe = validate_response(
                                        data, config["question"]["question_id"], api_key
                                    )
                                except ValueError:
                                    error = "schema_error"
                                    stop.set()
                                if safe:
                                    requested = config["model"]
                                    saved_model = db.execute(
                                        "SELECT value FROM metadata WHERE key='returned_model'"
                                    ).fetchone()
                                    expected = (
                                        requested
                                        if PINNED.fullmatch(requested)
                                        else (saved_model[0] if saved_model else None)
                                    )
                                    if expected and safe["model"] != expected:
                                        error = "model_drift"
                                        stop.set()
                                    elif saved_model is None:
                                        with db:
                                            db.execute(
                                                "INSERT INTO metadata VALUES ('returned_model',?)",
                                                (safe["model"],),
                                            )
                        except httpx.RequestError:
                            elapsed = (time.perf_counter() - tick) * 1000
                            error, retryable = "transport_error", True
                        total = (time.perf_counter() - started) * 1000
                        with db:
                            db.execute(
                                "UPDATE attempts SET "
                                "ended_at=?,status_code=?,error_type=?,latency_ms=?,"
                                "response_json=?,usage_json=? "
                                "WHERE sequence=?",
                                (
                                    now(),
                                    code,
                                    error,
                                    elapsed,
                                    json.dumps(safe) if safe else None,
                                    json.dumps(usage) if usage is not None else None,
                                    sequence,
                                ),
                            )
                            status = "success" if error is None else "failed"
                            db.execute(
                                "UPDATE examples SET "
                                "status=?,response_json=?,last_error=?,total_ms=? WHERE id=?",
                                (
                                    status,
                                    json.dumps(safe) if error is None else None,
                                    error,
                                    total,
                                    item["id"],
                                ),
                            )
                        if error is None:
                            return
                        if code in FATAL:
                            stop.set()
                        if not retryable or retry == settings["max_retries"] or stop.is_set():
                            if code in {429, 529}:
                                stop.set()
                            return
                        delay = retry_delay(
                            headers,
                            retry,
                            settings["backoff_seconds"],
                            settings["max_backoff_seconds"],
                        )
                        if delay is None:
                            stop.set()
                            return
                        with db:
                            db.execute(
                                "UPDATE examples SET status='retrying' WHERE id=?", (item["id"],)
                            )
                        await sleep(delay)

                async def worker():
                    while not stop.is_set():
                        try:
                            record = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        await one(record)

                tasks = [asyncio.create_task(worker()) for _ in range(settings["concurrency"])]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            return dict(Counter(r[0] for r in db.execute("SELECT status FROM examples")))
        finally:
            db.close()
