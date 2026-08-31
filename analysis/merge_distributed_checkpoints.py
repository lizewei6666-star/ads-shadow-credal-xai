from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".merge_tmp_{path.name}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def discover_complete_worker_manifest(source: Path) -> tuple[Path, dict]:
    manifests = sorted(
        source.glob("distributed_workers/*/worker_manifest.json")
    )
    complete = []
    for path in manifests:
        payload = read_json(path)
        if str(payload.get("status", "")).lower() == "complete":
            complete.append((path, payload))
    if len(complete) != 1:
        raise RuntimeError(
            "The source must contain exactly one completed worker manifest; "
            f"found {len(complete)}."
        )
    return complete[0]


def validate_source_contract(
    source: Path,
    target: Path,
    manifest: dict,
) -> None:
    source_identity = read_json(source / "tables" / "dataset_identity.json")
    target_identity = read_json(target / "tables" / "dataset_identity.json")
    if source_identity != target_identity:
        raise RuntimeError(
            "Dataset identity mismatch. The two computers did not use the "
            "same cleaned ordered analysis table."
        )
    if manifest.get("dataset_identity") != source_identity:
        raise RuntimeError(
            "The worker manifest does not match its dataset_identity.json."
        )

    source_contract = read_json(
        source / "checkpoints" / "credal_contract_fingerprints.json"
    )
    target_contract = read_json(
        target / "checkpoints" / "credal_contract_fingerprints.json"
    )
    source_digest = str(
        source_contract.get("statistical_method_digest", "")
    )
    target_digest = str(
        target_contract.get("statistical_method_digest", "")
    )
    manifest_digest = str(
        manifest.get("statistical_method_digest", "")
    )
    if not source_digest or not target_digest:
        raise RuntimeError("A statistical method digest is missing.")
    if not (source_digest == target_digest == manifest_digest):
        raise RuntimeError(
            "Statistical method digest mismatch. Do not merge results from "
            "different package versions, model sets, package versions, or "
            "analysis settings."
        )


def partition_prefixes(partitions: Sequence[int]) -> tuple[str, ...]:
    return tuple(f"p{int(value):03d}_" for value in partitions)


def partition_files(
    source_root: Path,
    partitions: Sequence[int],
) -> Iterable[Path]:
    prefixes = partition_prefixes(partitions)
    if not source_root.exists():
        return []
    files: List[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        if any(part.startswith("_") for part in relative.parts[:-1]):
            continue
        if path.name.startswith(prefixes):
            files.append(path)
    return sorted(files)


def validate_checkpoint_markers(
    source_checkpoint_root: Path,
    partitions: Sequence[int],
) -> None:
    marker_files = [
        path
        for path in partition_files(source_checkpoint_root, partitions)
        if path.suffix.lower() == ".json"
    ]
    if not marker_files:
        raise RuntimeError(
            "No partition-specific checkpoint markers were found in the "
            "worker output."
        )
    for marker_path in marker_files:
        marker = read_json(marker_path)
        payload_path = marker_path.with_suffix(".pkl")
        if not payload_path.is_file():
            raise RuntimeError(
                f"Checkpoint marker lacks its payload: {marker_path}"
            )
        expected = str(marker.get("payload_sha256", ""))
        observed = file_sha256(payload_path)
        if not expected or expected != observed:
            raise RuntimeError(
                f"Checkpoint checksum mismatch: {payload_path}"
            )


def copy_without_overwrite(
    source_file: Path,
    source_root: Path,
    target_root: Path,
    audit_rows: List[Dict[str, object]],
) -> None:
    relative = source_file.relative_to(source_root)
    target_file = target_root / relative
    source_hash = file_sha256(source_file)
    if target_file.exists():
        target_hash = file_sha256(target_file)
        if target_hash != source_hash:
            raise RuntimeError(
                "Refusing to overwrite a nonidentical checkpoint: "
                f"{target_file}"
            )
        status = "identical_existing"
    else:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        if file_sha256(target_file) != source_hash:
            raise RuntimeError(
                f"Post-copy checksum verification failed: {target_file}"
            )
        status = "copied"
    audit_rows.append(
        {
            "relative_path": str(relative),
            "sha256": source_hash,
            "bytes": int(source_file.stat().st_size),
            "status": status,
        }
    )


def copy_audit_if_missing(
    source_file: Path,
    source_root: Path,
    target_root: Path,
    audit_rows: List[Dict[str, object]],
) -> None:
    """Retain an existing passed implementation audit; otherwise import it."""
    relative = source_file.relative_to(source_root)
    target_file = target_root / relative
    source_hash = file_sha256(source_file)
    if target_file.exists():
        audit_rows.append(
            {
                "relative_path": str(relative),
                "sha256": source_hash,
                "target_sha256": file_sha256(target_file),
                "bytes": int(source_file.stat().st_size),
                "status": "existing_implementation_audit_retained",
            }
        )
        return
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target_file)
    if file_sha256(target_file) != source_hash:
        raise RuntimeError(
            f"Post-copy checksum verification failed: {target_file}"
        )
    audit_rows.append(
        {
            "relative_path": str(relative),
            "sha256": source_hash,
            "bytes": int(source_file.stat().st_size),
            "status": "implementation_audit_copied",
        }
    )


def merge_worker_output(source: Path, target: Path) -> Path:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if source == target:
        raise ValueError("Source and target output directories must differ.")
    if not source.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source}")
    if not target.is_dir():
        raise NotADirectoryError(f"Target directory not found: {target}")

    manifest_path, manifest = discover_complete_worker_manifest(source)
    partitions = [
        int(value)
        for value in manifest.get("assigned_outer_partitions", [])
    ]
    if not partitions:
        raise RuntimeError(
            "The completed worker manifest contains no assigned partitions."
        )
    validate_source_contract(source, target, manifest)

    source_units = source / "checkpoints" / "credal_shadow_units"
    target_units = target / "checkpoints" / "credal_shadow_units"
    validate_checkpoint_markers(source_units, partitions)

    audit_rows: List[Dict[str, object]] = []
    for source_root, target_root in [
        (source_units, target_units),
        (
            source / "checkpoints" / "loco_cache_audit",
            target / "checkpoints" / "loco_cache_audit",
        ),
    ]:
        for source_file in partition_files(source_root, partitions):
            copy_without_overwrite(
                source_file=source_file,
                source_root=source_root,
                target_root=target_root,
                audit_rows=audit_rows,
            )

    for audit_name in [
        "target_shap_audit",
        "pfi_batch_equivalence",
    ]:
        source_audit_root = source / "checkpoints" / audit_name
        target_audit_root = target / "checkpoints" / audit_name
        if source_audit_root.exists():
            for source_file in sorted(source_audit_root.rglob("*")):
                if source_file.is_file():
                    copy_audit_if_missing(
                        source_file=source_file,
                        source_root=source_audit_root,
                        target_root=target_audit_root,
                        audit_rows=audit_rows,
                    )

    target_worker_dir = (
        target
        / "distributed_workers"
        / str(manifest.get("worker_id", "worker"))
    )
    target_worker_dir.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(manifest_path.parent.glob("*")):
        if source_file.is_file():
            copy_without_overwrite(
                source_file=source_file,
                source_root=manifest_path.parent,
                target_root=target_worker_dir,
                audit_rows=audit_rows,
            )

    merge_record = {
        "status": "complete",
        "source_output": str(source),
        "target_output": str(target),
        "worker_id": manifest.get("worker_id"),
        "merged_outer_partitions": partitions,
        "statistical_method_digest": manifest.get(
            "statistical_method_digest"
        ),
        "dataset_identity": manifest.get("dataset_identity"),
        "files_copied": int(
            sum(
                row["status"]
                in {"copied", "implementation_audit_copied"}
                for row in audit_rows
            )
        ),
        "identical_files_skipped": int(
            sum(
                row["status"] == "identical_existing"
                for row in audit_rows
            )
        ),
        "merged_bytes": int(sum(row["bytes"] for row in audit_rows)),
        "file_audit": audit_rows,
        "merged_at": datetime.now().isoformat(timespec="seconds"),
    }
    audit_path = (
        target
        / "distributed_merges"
        / (
            f"merge_{manifest.get('worker_id', 'worker')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    )
    atomic_json_dump(merge_record, audit_path)
    return audit_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a completed distributed ADS checkpoint shard into "
            "the aggregation computer without overwriting divergent files."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Completed worker output directory copied from the other PC.",
    )
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="Existing aggregation output directory on PC1.",
    )
    args = parser.parse_args()
    audit_path = merge_worker_output(args.source, args.target)
    print("Checkpoint merge completed and verified.")
    print(f"Merge audit: {audit_path}")


if __name__ == "__main__":
    main()
