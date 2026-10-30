from __future__ import annotations

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any


EXPERIMENT_ID = "EXP-101-DEATH-CENSUS"
TASK_SHA256 = "82c29750adf81e80bfb2c744be564b990dce4d90f7e8ed3d46bccc82b925f3e9"
TRANSFORM_ID = "binance_rich_text_document_order_v2"
BLOCK_TAGS = frozenset(
    {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div", "tr", "table"}
)


class FlattenError(RuntimeError):
    pass


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, _attrs: Any) -> None:
        if tag in BLOCK_TAGS or tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _render_rich_node(value: Any) -> str:
    if isinstance(value, list):
        return "".join(_render_rich_node(item) for item in value)
    if not isinstance(value, dict):
        return ""
    if value.get("node") == "text":
        return str(value.get("text") or "")
    if value.get("node") == "root":
        return "\n".join(_render_rich_node(item) for item in value.get("child", []))
    tag = str(value.get("tag") or "").lower()
    if tag == "br":
        return "\n"
    rendered = "".join(_render_rich_node(item) for item in value.get("child", []))
    if tag == "td":
        return rendered + "\t"
    if tag in BLOCK_TAGS:
        return rendered + "\n"
    return rendered


def flatten_body(body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parser = _TextExtractor()
        parser.feed(body)
        parser.close()
        rendered = "".join(parser.parts)
    else:
        if not isinstance(parsed, (dict, list)):
            raise FlattenError("rich-text body JSON is not an object or array")
        rendered = _render_rich_node(parsed)
    flattened = re.sub(r"\s+", " ", html.unescape(rendered).replace("\u00a0", " ")).strip()
    if not flattened:
        raise FlattenError("flattened article body is empty")
    return flattened


def flatten_corpus(candidate_csv: Path, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FlattenError("flattened corpus output already exists; refusing overwrite")
    with candidate_csv.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["article_code"])
    if len(rows) != 159 or len({row["article_code"] for row in rows}) != 159:
        raise FlattenError("flattening requires the frozen 159 unique candidates")

    records: list[dict[str, Any]] = []
    for row in rows:
        article_id = str(row["article_code"])
        source = source_dir / "articles" / f"{article_id}.json"
        if source.is_symlink() or not source.is_file():
            raise FlattenError(f"source article missing or symlinked: {article_id}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if str(payload.get("code")) != article_id or str(payload.get("title")) != row["title"]:
            raise FlattenError(f"source article identity mismatch: {article_id}")
        flattened = {
            "body": flatten_body(str(payload["body"])),
            "code": article_id,
            "title": str(payload["title"]),
        }
        target = output_dir / "articles" / f"{article_id}.json"
        target_bytes = (
            json.dumps(flattened, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        _atomic_write(target, target_bytes)
        records.append(
            {
                "article_id": article_id,
                "source_sha256": file_sha256(source),
                "flattened_sha256": hashlib.sha256(target_bytes).hexdigest(),
                "source_body_chars": len(str(payload["body"])),
                "flattened_body_chars": len(flattened["body"]),
            }
        )

    manifest: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "deterministic_rich_text_flattening",
        "task_sha256": TASK_SHA256,
        "transform_id": TRANSFORM_ID,
        "candidate_csv_sha256": file_sha256(candidate_csv),
        "article_count": len(records),
        "article_records": records,
        "model_called": False,
        "protected_rows_read": False,
        "performance_or_death_metrics_computed": False,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _atomic_write(
        output_dir / "flatten-manifest.json",
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Flatten EXP-101 Binance rich-text bodies")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = flatten_corpus(args.candidate_csv, args.source_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
