#!/usr/bin/env python3

import argparse
import json
import pathlib
import re


CLASSIFICATION_ROW = re.compile(
    r"^(?P<label>\d+)\s+(?P<precision>[0-9.]+)\s+"
    r"(?P<recall>[0-9.]+)\s+(?P<f1>[0-9.]+)\s+(?P<support>\d+)$"
)
SUMMARY_ROW = re.compile(
    r"^(?P<label>accuracy|macro avg|weighted avg)\s+"
    r"(?P<precision>[0-9.]+)\s+(?P<recall>[0-9.]+)\s+"
    r"(?P<f1>[0-9.]+)\s+(?P<support>\d+)$"
)
ACCURACY_ROW = re.compile(r"^accuracy\s+(?P<f1>[0-9.]+)\s+(?P<support>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a command-resolved classification report for one experiment."
    )
    parser.add_argument(
        "experiment",
        type=int,
        nargs="?",
        help="Experiment number used to replace {experiment} in each path template.",
    )
    report_group = parser.add_mutually_exclusive_group(required=True)
    report_group.add_argument(
        "--report",
        dest="report",
        help="Report JSON path. This is sufficient when command names are not needed.",
    )
    report_group.add_argument(
        "--classification-storage",
        dest="classification_storage",
        help="Classification-storage metrics JSON path.",
    )
    report_group.add_argument(
        "--report-template",
        dest="report_template",
        help="Report JSON path or template, for example reports/experiment_{experiment}.json.",
    )
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument(
        "--graph-storage",
        help="Graph storage path used to resolve label IDs to commands.",
    )
    graph_group.add_argument(
        "--graph-storage-template",
        help="Graph storage path or template used to resolve label IDs to commands.",
    )
    return parser.parse_args()


def resolve_path(template: str, experiment: int) -> pathlib.Path:
    return pathlib.Path(template.format(experiment=experiment))


def load_classification_report(path: pathlib.Path) -> str:
    raw_text = path.read_text(encoding="utf-8")
    try:
        payload: object = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text

    if isinstance(payload, dict):
        classification_report = payload.get("classification_report")
        if isinstance(classification_report, str):
            return classification_report

    raise ValueError(f"No classification_report string found in {path}")


def build_id_to_command(mapping) -> dict[int, str]:
    value_ids: dict[int, str] = {}
    for command, label in mapping.items():
        try:
            value_ids[int(label)] = str(command)
        except (TypeError, ValueError):
            value_ids = {}
            break
    if value_ids:
        return value_ids

    key_ids: dict[int, str] = {}
    for label, command in mapping.items():
        try:
            key_ids[int(label)] = str(command)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "train_subject_cmd_to_id is neither command-to-ID nor ID-to-command"
            ) from exc
    return key_ids


def load_id_to_command(path: pathlib.Path) -> dict[int, str]:
    from grasp.graph.graph_storage import load_graph_storage

    graph_storage = load_graph_storage(str(path))
    return build_id_to_command(graph_storage.train_subject_cmd_to_id)


def parse_rows(text: str, id_to_command: dict[int, str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("precision", "recall")):
            continue

        match = CLASSIFICATION_ROW.match(line)
        if match:
            label = int(match.group("label"))
            rows.append(
                [
                    str(label),
                    id_to_command.get(label, "<unknown>"),
                    f"{float(match.group('precision')):.2f}",
                    f"{float(match.group('recall')):.2f}",
                    f"{float(match.group('f1')):.2f}",
                    match.group("support"),
                ]
            )
            continue

        match = SUMMARY_ROW.match(line)
        if match:
            rows.append(
                [
                    match.group("label"),
                    "",
                    f"{float(match.group('precision')):.2f}",
                    f"{float(match.group('recall')):.2f}",
                    f"{float(match.group('f1')):.2f}",
                    match.group("support"),
                ]
            )
            continue

        match = ACCURACY_ROW.match(line)
        if match:
            rows.append(
                [
                    "accuracy",
                    "",
                    "",
                    "",
                    f"{float(match.group('f1')):.2f}",
                    match.group("support"),
                ]
            )
    return rows


def print_table(rows: list[list[str]]) -> None:
    headers = ["label", "cmd", "precision", "recall", "f1-score", "support"]
    if not rows:
        raise ValueError("No classification rows found in the report")

    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    if args.report:
        report_path = pathlib.Path(args.report)
    elif args.classification_storage:
        report_path = pathlib.Path(args.classification_storage)
    elif args.experiment is None:
        raise ValueError("An experiment number is required when using --report-template")
    else:
        report_path = resolve_path(args.report_template, args.experiment)

    if not report_path.is_file():
        raise FileNotFoundError(f"Report does not exist: {report_path}")

    id_to_command: dict[int, str] = {}
    if args.graph_storage:
        graph_storage_path = pathlib.Path(args.graph_storage)
    elif args.graph_storage_template:
        if args.experiment is None:
            raise ValueError("An experiment number is required when using --graph-storage-template")
        graph_storage_path = resolve_path(args.graph_storage_template, args.experiment)
    else:
        graph_storage_path = None

    if graph_storage_path is not None:
        if not graph_storage_path.is_file():
            raise FileNotFoundError(f"Graph storage does not exist: {graph_storage_path}")
        id_to_command = load_id_to_command(graph_storage_path)

    report_text = load_classification_report(report_path)
    print_table(parse_rows(report_text, id_to_command))


if __name__ == "__main__":
    main()
