#!/usr/bin/env python3
r"""
analyze_nn_mismatches_against_gt.py

Analyze the special NN mismatch output generated from nn_results.csv against
ATLAS/GRASP-style ground-truth files.

Input can be either:
1) nn_results.csv
   - the script filters rows where nearest_train_executable_key != true_executable_key
2) nn_executable_mismatches.csv
   - the script analyzes all rows as already-filtered NN anomalies/mismatches

The GT loader is based on the GroundTruthAnalyzer design you provided:
- robust first-comma / last-comma parser
- no ast.literal_eval, so Windows paths containing \users\... are safe
- handles unknown_exec_processes.csv specially

Outputs:
- analysis_report.json
- summary.csv
- nn_mismatch_gt_matches.csv
- possible_attack_hits_not_in_unknown.csv
- ground_truth_detection_detail.csv
- per_attack_file_recall_excluding_unknown_ids.csv
- attack_files_hit.csv
- parse_warnings.csv

Example for your generated special output:
python analyze_nn_mismatches_against_gt.py \
  --nn-csv pids_nn_exec_report/nn_executable_mismatches.csv \
  --ground-truth /home/buchta/grasp2/ground_truth/atlasv2_edr \
  --output-json pids_nn_exec_report/nn_mismatch_gt_analysis.json \
  --output-dir pids_nn_exec_report/nn_mismatch_gt_analysis

Example directly from nn_results.csv:
python analyze_nn_mismatches_against_gt.py \
  --nn-csv pids_nn_exec_report/nn_results.csv \
  --input-kind nn-results \
  --ground-truth /home/buchta/grasp2/ground_truth/atlasv2_edr \
  --output-dir pids_nn_exec_report/nn_mismatch_gt_analysis

  python grasp/contextualization/analyze_nn_mismatches_against_gt.py   --nn-csv /home/buchta/grasp2/pids_nn_exec_report/nn_results.csv   --input-kind nn-results   --ground-truth /home/buchta/grasp2/ground_truth/atlasv2_edr   --output-json /home/buchta/grasp2/pids_nn_exec_report/nn_mismatch_gt_analysis.json   --output-dir /home/buchta/grasp2/pids_nn_exec_report/nn_mismatch_gt_analysis

  python grasp/contextualization/analyze_nn_mismatches_against_gt.py   --nn-csv /home/buchta/grasp2/pids_nn_exec_kdtree_report_kdtree/nn_results.csv   --input-kind nn-results   --ground-truth /home/buchta/grasp2/ground_truth/cadets_e3_sigsac   --output-json /home/buchta/grasp2/pids_nn_exec_report/nn_mismatch_gt_analysis_cadets.json   --output-dir /home/buchta/grasp2/pids_nn_exec_report/nn_mismatch_gt_analysis
  """

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import pandas as pd

UNKNOWN_FILE = "unknown_exec_processes.csv"
InputKind = Literal["auto", "nn-results", "mismatches"]


class GroundTruthAnalyzer:
    r"""
    Robust ground-truth loader for ATLAS/GRASP-style GT files.

    Expected common row shape:
        uuid,metadata,node_id

    Important design choice:
        We do NOT parse metadata with ast.literal_eval because Windows paths
        such as \program files\... or \users\... can trigger unicodeescape
        problems. For this analysis we only need uuid, node_id and source file.
    """

    def __init__(self, gt_dir: str | Path, unknown_file: str = UNKNOWN_FILE):
        self.gt_dir = Path(gt_dir)
        self.unknown_file = unknown_file
        self.by_uuid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_node_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.file_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"records": 0, "unique_ids": set(), "unique_uuids": set()}
        )
        self.parse_warnings: List[Dict[str, Any]] = []

    def load(self) -> None:
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT directory does not exist: {self.gt_dir}")
        if not self.gt_dir.is_dir():
            raise NotADirectoryError(f"GT path is not a directory: {self.gt_dir}")

        for gt_file in sorted(self.gt_dir.iterdir()):
            if not gt_file.is_file():
                continue
            with open(gt_file, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    raw_line = line.rstrip("\n")
                    line = raw_line.strip()
                    if not line:
                        continue
                    parsed = self._parse_line(line=line, file_name=gt_file.name, line_no=line_no)
                    if parsed is None:
                        continue
                    self.by_uuid[parsed["uuid"]].append(parsed)
                    self.by_node_id[parsed["node_id"]].append(parsed)
                    self.file_stats[gt_file.name]["records"] += 1
                    self.file_stats[gt_file.name]["unique_ids"].add(parsed["node_id"])
                    self.file_stats[gt_file.name]["unique_uuids"].add(parsed["uuid"])

    def _parse_line(self, line: str, file_name: str, line_no: int) -> Optional[Dict[str, Any]]:
        first = line.find(",")
        last = line.rfind(",")
        if first < 0 or last <= first:
            self.parse_warnings.append(
                {
                    "file": file_name,
                    "line": line_no,
                    "reason": "Could not find first and last comma",
                    "raw": line,
                }
            )
            return None
        uuid = line[:first].strip()
        metadata = line[first + 1 : last].strip()
        node_id = line[last + 1 :].strip()
        if not uuid or not node_id:
            self.parse_warnings.append(
                {"file": file_name, "line": line_no, "reason": "Empty uuid or node_id", "raw": line}
            )
            return None
        return {
            "uuid": uuid,
            "node_id": node_id,
            "metadata": metadata,
            "file": file_name,
            "line": line_no,
        }

    def summary(self) -> Dict[str, Any]:
        duplicate_ids = {
            node_id: records
            for node_id, records in self.by_node_id.items()
            if len({r["file"] for r in records}) > 1
        }
        duplicate_uuids = {
            uuid: records
            for uuid, records in self.by_uuid.items()
            if len({r["file"] for r in records}) > 1
        }

        top_reused_ids = []
        for node_id, records in duplicate_ids.items():
            files = sorted({r["file"] for r in records})
            top_reused_ids.append(
                {
                    "id": node_id,
                    "file_count": len(files),
                    "occurrence_count": len(records),
                    "also_in_unknown_exec_processes": self.unknown_file in files,
                    "files": files,
                    "attack_files": [f for f in files if f != self.unknown_file],
                }
            )
        top_reused_ids.sort(key=lambda x: (x["file_count"], x["occurrence_count"]), reverse=True)

        top_reused_uuids = []
        for uuid, records in duplicate_uuids.items():
            files = sorted({r["file"] for r in records})
            top_reused_uuids.append(
                {
                    "uuid": uuid,
                    "file_count": len(files),
                    "occurrence_count": len(records),
                    "also_in_unknown_exec_processes": self.unknown_file in files,
                    "files": files,
                    "attack_files": [f for f in files if f != self.unknown_file],
                }
            )
        top_reused_uuids.sort(key=lambda x: (x["file_count"], x["occurrence_count"]), reverse=True)

        unknown_records = self.file_stats.get(
            self.unknown_file, {"records": 0, "unique_ids": set(), "unique_uuids": set()}
        )
        attack_files = [f for f in self.file_stats.keys() if f != self.unknown_file]

        return {
            "total_gt_files": len(self.file_stats),
            "total_attack_gt_files_excluding_unknown": len(attack_files),
            "unknown_file_present": self.unknown_file in self.file_stats,
            "total_unique_ids": len(self.by_node_id),
            "total_unique_uuids": len(self.by_uuid),
            "duplicate_ids": len(duplicate_ids),
            "duplicate_uuids": len(duplicate_uuids),
            "unknown_exec_processes_records": unknown_records["records"],
            "unknown_exec_processes_unique_ids": len(unknown_records["unique_ids"]),
            "unknown_exec_processes_unique_uuids": len(unknown_records["unique_uuids"]),
            "parse_warning_count": len(self.parse_warnings),
            "top_reused_ids": top_reused_ids[:100],
            "top_reused_uuids": top_reused_uuids[:100],
            "file_statistics": {
                f: {
                    "records": stats["records"],
                    "unique_ids": len(stats["unique_ids"]),
                    "unique_uuids": len(stats["unique_uuids"]),
                    "is_unknown_exec_processes": f == self.unknown_file,
                }
                for f, stats in sorted(self.file_stats.items())
            },
        }


class NNMismatchAnalyzer:
    def __init__(
        self,
        nn_csv: str | Path,
        gt_dir: str | Path,
        unknown_file: str = UNKNOWN_FILE,
        input_kind: InputKind = "auto",
    ):
        self.nn_csv = Path(nn_csv)
        self.unknown_file = unknown_file
        self.input_kind = input_kind
        self.gt = GroundTruthAnalyzer(gt_dir, unknown_file=unknown_file)

    def load_mismatches(self) -> List[Dict[str, Any]]:
        if not self.nn_csv.exists():
            raise FileNotFoundError(f"NN CSV does not exist: {self.nn_csv}")
        df = pd.read_csv(self.nn_csv)

        kind = self.input_kind
        if kind == "auto":
            if "nearest_train_executable_key" in df.columns and "true_executable_key" in df.columns:
                # Safe for both nn_results.csv and already-filtered mismatch CSV.
                # If the file is already filtered, this condition keeps all mismatch rows.
                kind = "nn-results"
            else:
                kind = "mismatches"

        if kind == "nn-results":
            required = [
                "test_sample_id",
                "test_process_id",
                "test_uuid",
                "true_executable_key",
                "nearest_train_executable_key",
            ]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required nn-results columns: {missing}")
            df = df[
                df["nearest_train_executable_key"].astype(str)
                != df["true_executable_key"].astype(str)
            ].copy()
        else:
            required = ["test_sample_id", "test_process_id", "test_uuid"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required mismatch columns: {missing}")

        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "id": str(r.get("test_process_id", "")),
                    "uuid": str(r.get("test_uuid", "")),
                    "test_sample_id": str(r.get("test_sample_id", "")),
                    "true_executable": r.get("true_executable"),
                    "true_executable_key": r.get("true_executable_key"),
                    "predicted_executable": r.get("predicted_executable"),
                    "predicted_executable_key": r.get("predicted_executable_key"),
                    "nearest_train_executable": r.get("nearest_train_executable"),
                    "nearest_train_executable_key": r.get("nearest_train_executable_key"),
                    "nearest_train_sample_id": r.get("nearest_train_sample_id"),
                    "nearest_train_process_id": r.get("nearest_train_process_id"),
                    "nearest_train_uuid": r.get("nearest_train_uuid"),
                    "distance": r.get("distance"),
                    "similarity_score": r.get("similarity_score"),
                    "test_data_path": r.get("test_data_path"),
                }
            )
        return rows

    def analyze(self) -> Dict[str, Any]:
        self.gt.load()
        mismatches = self.load_mismatches()

        detected_ids = {str(a.get("id", "")) for a in mismatches if str(a.get("id", ""))}
        detected_uuids = {str(a.get("uuid", "")) for a in mismatches if str(a.get("uuid", ""))}

        attack_file_hits = Counter()
        possible_attack_hits = []
        matched_nodes = []

        attack_hits = 0
        unknown_hits = 0
        both_hits = 0
        attack_only_hits = 0
        unknown_only_hits = 0
        unmatched = 0

        for row_in in mismatches:
            node_id = str(row_in.get("id", ""))
            uuid = str(row_in.get("uuid", ""))
            node_matches = self.gt.by_node_id.get(node_id, [])
            uuid_matches = self.gt.by_uuid.get(uuid, [])
            # De-duplicate exact same GT records that match by both id and uuid.
            seen = set()
            matches = []
            for m in node_matches + uuid_matches:
                key = (m["file"], m["line"], m["node_id"], m["uuid"])
                if key not in seen:
                    seen.add(key)
                    matches.append(m)

            files = sorted({m["file"] for m in matches})
            attack_files = [f for f in files if f != self.unknown_file]
            is_attack = len(attack_files) > 0
            is_unknown = self.unknown_file in files

            if is_attack:
                attack_hits += 1
            if is_unknown:
                unknown_hits += 1
            if is_attack and is_unknown:
                both_hits += 1
            if is_attack and not is_unknown:
                attack_only_hits += 1
            if is_unknown and not is_attack:
                unknown_only_hits += 1
            if not files:
                unmatched += 1
            for f in attack_files:
                attack_file_hits[f] += 1

            out = {
                **row_in,
                "matched": bool(files),
                "attack_match": is_attack,
                "unknown_match": is_unknown,
                "attack_files": attack_files,
                "all_files": files,
            }
            matched_nodes.append(out)
            if is_attack and not is_unknown:
                possible_attack_hits.append(out)

        gt_detail = self._build_gt_detail(detected_ids=detected_ids, detected_uuids=detected_uuids)
        unknown_filtered = self._build_unknown_filtered_summary(gt_detail)
        per_attack_file_recall = self._build_per_attack_file_recall(gt_detail)

        summary = {
            **self.gt.summary(),
            "input_csv": str(self.nn_csv),
            "total_nn_mismatches": len(mismatches),
            "matched_mismatches": len(mismatches) - unmatched,
            "unmatched_mismatches": unmatched,
            "mismatch_hits_in_attack_files": attack_hits,
            "mismatch_hits_in_unknown_exec_processes": unknown_hits,
            "mismatch_hits_only_attack_files": attack_only_hits,
            "mismatch_hits_only_unknown_exec_processes": unknown_only_hits,
            "mismatch_hits_in_both": both_hits,
            "possible_attack_hits_not_in_unknown_count": len(possible_attack_hits),
            "attack_file_coverage": len(attack_file_hits),
            "gt_ids_flagged_by_nn_mismatch": sum(1 for r in gt_detail if r["detected"]),
            "gt_ids_not_flagged_by_nn_mismatch": sum(1 for r in gt_detail if not r["detected"]),
            "gt_ids_flagged_and_in_unknown": sum(
                1 for r in gt_detail if r["detected"] and r["also_in_unknown_exec_processes"]
            ),
        }

        return {
            "summary": summary,
            "unknown_filtered_summary": unknown_filtered,
            "per_attack_file_recall_excluding_unknown_ids": per_attack_file_recall,
            "attack_files_hit": dict(attack_file_hits.most_common()),
            "possible_attack_hits_not_in_unknown": possible_attack_hits,
            "matched_nodes": matched_nodes,
            "ground_truth_detection_detail": gt_detail,
            "parse_warnings": self.gt.parse_warnings,
        }

    def _build_gt_detail(
        self, detected_ids: set[str], detected_uuids: set[str]
    ) -> List[Dict[str, Any]]:
        gt_detail = []
        for node_id, records in self.gt.by_node_id.items():
            files = sorted({r["file"] for r in records})
            uuids = sorted({r["uuid"] for r in records})
            attack_files = [f for f in files if f != self.unknown_file]
            in_unknown = self.unknown_file in files
            detected_by_id = node_id in detected_ids
            detected_by_uuid = any(u in detected_uuids for u in uuids)
            detected = detected_by_id or detected_by_uuid
            gt_detail.append(
                {
                    "node_id": node_id,
                    "uuids": uuids,
                    "files": files,
                    "attack_files": attack_files,
                    "detected": detected,
                    "detected_by_id": detected_by_id,
                    "detected_by_uuid": detected_by_uuid,
                    "also_in_unknown_exec_processes": in_unknown,
                    "unknown_filtered_excluded": in_unknown,
                    "attack_only_candidate": bool(attack_files) and not in_unknown,
                    "occurrence_count": len(records),
                    "file_count": len(files),
                }
            )
        gt_detail.sort(
            key=lambda x: (not x["detected"], x["unknown_filtered_excluded"], x["node_id"])
        )
        return gt_detail

    def _build_unknown_filtered_summary(self, gt_detail: List[Dict[str, Any]]) -> Dict[str, Any]:
        attack_only_rows = [r for r in gt_detail if r["attack_only_candidate"]]
        detected_rows = [r for r in attack_only_rows if r["detected"]]
        missed_rows = [r for r in attack_only_rows if not r["detected"]]
        total = len(attack_only_rows)
        detected = len(detected_rows)
        missed = len(missed_rows)
        detected_by_file = Counter()
        total_by_file = Counter()
        missed_by_file = Counter()
        for row in attack_only_rows:
            for f in row["attack_files"]:
                total_by_file[f] += 1
                if row["detected"]:
                    detected_by_file[f] += 1
                else:
                    missed_by_file[f] += 1
        per_file = {}
        for f in sorted(total_by_file):
            t = total_by_file[f]
            d = detected_by_file[f]
            m = missed_by_file[f]
            per_file[f] = {
                "attack_only_gt_ids_total": t,
                "attack_only_gt_ids_flagged_by_nn_mismatch": d,
                "attack_only_gt_ids_not_flagged_by_nn_mismatch": m,
                "attack_only_recall": d / t if t else None,
            }
        return {
            "description": f"Special NN mismatch evaluation view where all GT IDs that appear in {self.unknown_file} are excluded completely.",
            "excluded_unknown_file": self.unknown_file,
            "attack_only_gt_ids_total": total,
            "attack_only_gt_ids_flagged_by_nn_mismatch": detected,
            "attack_only_gt_ids_not_flagged_by_nn_mismatch": missed,
            "attack_only_gt_recall": detected / total if total else None,
            "per_attack_file": per_file,
        }

    def _build_per_attack_file_recall(self, gt_detail: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_by_file = Counter()
        detected_by_file = Counter()
        for row in gt_detail:
            if row["also_in_unknown_exec_processes"]:
                continue
            for f in row["attack_files"]:
                total_by_file[f] += 1
                if row["detected"]:
                    detected_by_file[f] += 1
        result = {}
        for f in sorted(total_by_file):
            total = total_by_file[f]
            detected = detected_by_file[f]
            result[f] = {
                "gt_ids_total_excluding_unknown": total,
                "gt_ids_flagged_by_nn_mismatch_excluding_unknown": detected,
                "gt_ids_missed_excluding_unknown": total - detected,
                "recall_excluding_unknown": detected / total if total else None,
            }
        return result


def dump_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, list):
                    v = ";".join(str(x) for x in v)
                out[k] = v
            writer.writerow(out)


def write_csv_outputs(results: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten compact summary.
    summary_rows = []
    for k, v in results["summary"].items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            summary_rows.append({"section": "summary", "metric": k, "value": v})
    ufs = results["unknown_filtered_summary"]
    for k, v in ufs.items():
        if k != "per_attack_file" and (isinstance(v, (str, int, float, bool)) or v is None):
            summary_rows.append({"section": "unknown_filtered_summary", "metric": k, "value": v})
    dump_csv(output_dir / "summary.csv", summary_rows, ["section", "metric", "value"])

    matched_fields = [
        "id",
        "uuid",
        "test_sample_id",
        "matched",
        "attack_match",
        "unknown_match",
        "true_executable",
        "true_executable_key",
        "nearest_train_executable",
        "nearest_train_executable_key",
        "predicted_executable",
        "predicted_executable_key",
        "nearest_train_sample_id",
        "nearest_train_process_id",
        "distance",
        "similarity_score",
        "attack_files",
        "all_files",
        "test_data_path",
    ]
    dump_csv(output_dir / "nn_mismatch_gt_matches.csv", results["matched_nodes"], matched_fields)
    dump_csv(
        output_dir / "possible_attack_hits_not_in_unknown.csv",
        results["possible_attack_hits_not_in_unknown"],
        matched_fields,
    )

    gt_fields = [
        "node_id",
        "detected",
        "detected_by_id",
        "detected_by_uuid",
        "also_in_unknown_exec_processes",
        "unknown_filtered_excluded",
        "attack_only_candidate",
        "files",
        "attack_files",
        "uuids",
        "occurrence_count",
        "file_count",
    ]
    dump_csv(
        output_dir / "ground_truth_detection_detail.csv",
        results["ground_truth_detection_detail"],
        gt_fields,
    )

    recall_rows = []
    for attack_file, row in results["per_attack_file_recall_excluding_unknown_ids"].items():
        recall_rows.append({"attack_file": attack_file, **row})
    dump_csv(
        output_dir / "per_attack_file_recall_excluding_unknown_ids.csv",
        recall_rows,
        [
            "attack_file",
            "gt_ids_total_excluding_unknown",
            "gt_ids_flagged_by_nn_mismatch_excluding_unknown",
            "gt_ids_missed_excluding_unknown",
            "recall_excluding_unknown",
        ],
    )

    attack_hit_rows = [
        {"attack_file": k, "nn_mismatch_hits": v} for k, v in results["attack_files_hit"].items()
    ]
    dump_csv(
        output_dir / "attack_files_hit.csv", attack_hit_rows, ["attack_file", "nn_mismatch_hits"]
    )

    dump_csv(
        output_dir / "parse_warnings.csv",
        results["parse_warnings"],
        ["file", "line", "reason", "raw"],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze NN executable mismatches against GT dir with unknown_exec_processes handling."
    )
    p.add_argument(
        "--nn-csv", required=True, help="Path to nn_results.csv or nn_executable_mismatches.csv"
    )
    p.add_argument("--input-kind", choices=["auto", "nn-results", "mismatches"], default="auto")
    p.add_argument(
        "--ground-truth",
        required=True,
        help="Path to GT directory, e.g. /home/buchta/grasp2/ground_truth/atlasv2_edr",
    )
    p.add_argument("--output-json", default="nn_mismatch_gt_analysis.json")
    p.add_argument("--output-dir", default="nn_mismatch_gt_analysis")
    p.add_argument("--unknown-file", default=UNKNOWN_FILE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = NNMismatchAnalyzer(
        nn_csv=args.nn_csv,
        gt_dir=args.ground_truth,
        unknown_file=args.unknown_file,
        input_kind=args.input_kind,
    )
    results = analyzer.analyze()

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_csv_outputs(results, Path(args.output_dir))

    compact = {
        "summary": results["summary"],
        "unknown_filtered_summary": {
            k: v for k, v in results["unknown_filtered_summary"].items() if k != "per_attack_file"
        },
    }
    print(json.dumps(compact, indent=2))
    print(f"\nJSON written to: {output_json}")
    print(f"CSV outputs written to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
