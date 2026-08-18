import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from vlmeval.smp.file import INFER_FAIL_MSG


CACHE_SCHEMA_VERSION = 1
# A response that starts with INFER_FAIL_MSG (or its API-specific suffix) is a
# wrapper-side failure marker rather than a real prediction, and must not be
# cached. Centralise both forms here so we only have to update the marker text
# in one place (smp/file.py).
FAILURE_MARKERS = (
    INFER_FAIL_MSG,
    f"{INFER_FAIL_MSG} via API.",
)


class ResponseCache:
    """SQLite response cache with per-rank write databases and a merged root DB."""

    def __init__(
        self,
        enabled=False,
        cache_root=None,
        run_id=None,
        rank=0,
        world_size=1,
        model_name=None,
        logger=None,
    ):
        self.enabled = enabled
        self.cache_root = Path(cache_root).expanduser() if cache_root else None
        self.run_id = _safe_name(run_id or "run")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_name = model_name
        self.logger = logger
        self.stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "skips": 0,
        }
        self.rank_conn = None
        self.root_conn = None
        self.closed = True

        if not enabled:
            return

        self.run_dir = self.cache_root / "runs" / self.run_id
        self.root_db_path = self.cache_root / "cache.db"
        self.rank_db_path = self.run_dir / f"rank_{self.rank}.db"
        self.audit_path = self.run_dir / f"rank_{self.rank}.audit.jsonl"

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.rank_conn = sqlite3.connect(self.rank_db_path, timeout=60)
        _prepare_connection(self.rank_conn)

        if self.root_db_path.exists():
            self.root_conn = _connect_read_only(self.root_db_path)
        self.closed = False

    @classmethod
    def create(cls, cache_root=None, run_id=None, rank=0, world_size=1, model_name=None, logger=None):
        if not cache_root:
            return cls(enabled=False)
        return cls(
            enabled=True,
            cache_root=cache_root,
            run_id=run_id,
            rank=rank,
            world_size=world_size,
            model_name=model_name,
            logger=logger,
        )

    @classmethod
    def disabled(cls):
        return cls(enabled=False)

    def __bool__(self):
        return self.enabled

    def make_key(self, model, model_name, dataset_name, sample_index, message):
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "model": _model_fingerprint(model, model_name),
            "dataset_name": dataset_name,
            "sample_index": _jsonable(sample_index),
            "message": _normalize_for_hash(message),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return f"vlmeval-cache-v{CACHE_SCHEMA_VERSION}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"

    def lookup(self, key):
        if not self.enabled:
            return None
        self.stats["lookups"] += 1
        for conn in (self.rank_conn, self.root_conn):
            if conn is None:
                continue
            value = _lookup(conn, key)
            if value is not None:
                self.stats["hits"] += 1
                return value
        self.stats["misses"] += 1
        return None

    def store_response(self, key, response, metadata=None):
        if not self.enabled:
            return False
        if not _is_cacheable_response(response):
            self.stats["skips"] += 1
            return False

        created_at = _now()
        metadata = metadata or {}
        try:
            value_json = json.dumps(response, ensure_ascii=False, sort_keys=True)
            metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        except TypeError:
            self.stats["skips"] += 1
            return False

        self.rank_conn.execute(
            """
            INSERT OR REPLACE INTO response_cache(key, value_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, value_json, metadata_json, created_at),
        )
        self.rank_conn.commit()
        self.stats["stores"] += 1
        self._audit(key, response, metadata, created_at)
        return True

    def get_or_generate(self, model, model_name, dataset_name, sample_index, message, generate_fn):
        if not self.enabled or not _is_deterministic_model(model):
            if self.enabled:
                self.stats["skips"] += 1
            return generate_fn()

        key = self.make_key(
            model=model,
            model_name=model_name,
            dataset_name=dataset_name,
            sample_index=sample_index,
            message=message,
        )
        cached = self.lookup(key)
        if cached is not None:
            return cached

        response = generate_fn()
        self.store_response(
            key,
            response,
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "sample_index": _jsonable(sample_index),
            },
        )
        return response

    def finalize(self, use_distributed_barrier=True):
        if not self.enabled:
            return

        self.close()
        if use_distributed_barrier:
            _dist_barrier()

        if self.rank == 0:
            # The shared-root merge is a cross-run optimization; per-rank DBs
            # remain the source of truth, so a lock timeout (concurrent jobs
            # finalizing against the same cache root) must not fail the run.
            try:
                merged = self._merge_rank_databases()
            except sqlite3.OperationalError as exc:
                if self.logger is not None:
                    self.logger.warning(
                        "Response cache merge into %s skipped: %s (rank DBs kept at %s)",
                        self.root_db_path,
                        exc,
                        self.run_dir,
                    )
            else:
                if self.logger is not None:
                    self.logger.info(
                        "Response cache finalized at %s: merged=%s stats=%s",
                        self.root_db_path,
                        merged,
                        self.stats,
                    )

        if use_distributed_barrier:
            _dist_barrier()

    def close(self):
        if self.closed:
            return
        if self.rank_conn is not None:
            self.rank_conn.commit()
            self.rank_conn.close()
            self.rank_conn = None
        if self.root_conn is not None:
            self.root_conn.close()
            self.root_conn = None
        self.closed = True

    def _merge_rank_databases(self):
        root_conn = sqlite3.connect(self.root_db_path, timeout=60)
        _prepare_connection(root_conn)
        merged = 0
        try:
            for rank in range(self.world_size):
                rank_db = self.run_dir / f"rank_{rank}.db"
                if not rank_db.exists():
                    continue
                # Copy inside SQLite: the rows never enter Python, which keeps
                # the write lock on the shared root held for as short a time as
                # possible (every other rank is blocked on the barrier here).
                root_conn.execute("ATTACH DATABASE ? AS src", (str(rank_db),))
                try:
                    cur = root_conn.execute(
                        """
                        INSERT OR REPLACE INTO response_cache(key, value_json, metadata_json, created_at)
                        SELECT key, value_json, metadata_json, created_at FROM src.response_cache
                        """
                    )
                    merged += cur.rowcount if cur.rowcount > 0 else 0
                    root_conn.commit()
                finally:
                    root_conn.execute("DETACH DATABASE src")
        finally:
            root_conn.close()
        return merged

    def _audit(self, key, response, metadata, created_at):
        response_json = json.dumps(response, ensure_ascii=False, sort_keys=True, default=str)
        record = {
            "created_at": created_at,
            "key": key,
            "response_sha256": hashlib.sha256(response_json.encode("utf-8")).hexdigest(),
            "metadata": metadata,
        }
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _prepare_connection(conn):
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS response_cache (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _prepare_read_connection(conn):
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA query_only=ON")


def _connect_read_only(path):
    encoded_path = quote(os.fspath(path), safe="/")
    conn = sqlite3.connect(f"file:{encoded_path}?mode=ro", uri=True, timeout=60)
    _prepare_read_connection(conn)
    return conn


def _lookup(conn, key):
    try:
        row = conn.execute("SELECT value_json FROM response_cache WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return json.loads(row[0])


def _is_deterministic_model(model):
    explicit = getattr(model, "response_cache_deterministic", None)
    if explicit is not None:
        return bool(explicit)

    kwargs = getattr(model, "generate_kwargs", None)
    if not kwargs:
        return False

    temperature = kwargs.get("temperature", 0.0)
    do_sample = kwargs.get("do_sample", False)
    n = kwargs.get("n", kwargs.get("num_return_sequences", 1))
    best_of = kwargs.get("best_of", 1)
    try:
        temperature = float(temperature or 0.0)
        n = int(n or 1)
        best_of = int(best_of or 1)
    except (TypeError, ValueError):
        return False
    return temperature <= 0.0 and not bool(do_sample) and n <= 1 and best_of <= 1


def _is_cacheable_response(response):
    if isinstance(response, str):
        return not any(marker in response for marker in FAILURE_MARKERS)
    try:
        json.dumps(response, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return False
    return True


def _model_fingerprint(model, model_name):
    attrs = [
        "model_path",
        "image_pipeline",
        "serving_path",
        "tokenizer_path",
        "chat_template_hash",
        "vllm_prompt_contract",
        "tokenizer_batch_size",
        "tokenizer_add_special_tokens",
        "max_model_len",
    ]
    fingerprint = {
        "model_name": model_name,
        "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "generate_kwargs": _jsonable(getattr(model, "generate_kwargs", {})),
    }
    for attr in attrs:
        if hasattr(model, attr):
            fingerprint[attr] = _jsonable(getattr(model, attr))
    return fingerprint


def _normalize_for_hash(value):
    if isinstance(value, dict):
        normalized = {
            str(k): _normalize_for_hash(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
        media_type = normalized.get("type")
        media_value = normalized.get("value")
        if media_type in {"image", "video", "audio"} and isinstance(media_value, str):
            file_info = _file_fingerprint(media_value)
            if file_info is not None:
                normalized["file"] = file_info
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(x) for x in value]
    if isinstance(value, bytes):
        return {
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }
    return _jsonable(value)


_FILE_FINGERPRINT_CACHE = {}


def _file_fingerprint(path):
    # Image-token cache hits stream the same files through SHA-256 thousands
    # of times in a single eval. Keying on (path, size, mtime_ns) is enough
    # to detect "the file changed on disk" without re-hashing per call.
    try:
        file_path = Path(path)
        if not file_path.is_file():
            return None
        stat = file_path.stat()
        key = (os.fspath(file_path), stat.st_size, stat.st_mtime_ns)
        cached = _FILE_FINGERPRINT_CACHE.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        info = {
            "path": key[0],
            "size": stat.st_size,
            "sha256": digest.hexdigest(),
        }
        _FILE_FINGERPRINT_CACHE[key] = info
        return info
    except OSError:
        return None


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def _safe_name(value):
    value = str(value)
    safe = [c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value]
    return "".join(safe) or "run"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _dist_barrier():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return
