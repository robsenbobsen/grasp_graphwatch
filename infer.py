"""Run inference on a new time range with a pretrained model bundle (see grasp/stages.py).

    python infer.py --db-name cadets_e3 \
        --inference-start "2018-04-06 00:00:00" --inference-end "2018-04-13 23:59:59"

Without --model-id the latest model trained on --db-name is used.
"""

import argparse
import logging

from grasp import stages

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRASP inference with a trained model")
    parser.add_argument("--db-name", help="Database to read events from (default: model's)")
    parser.add_argument("--model-id", help="Model bundle (default: latest for --db-name)")
    parser.add_argument("--inference-start", required=True, help='e.g. "2018-04-06 00:00:00"')
    parser.add_argument("--inference-end", required=True, help='e.g. "2018-04-13 23:59:59"')
    parser.add_argument(
        "--ground-truth-dir",
        help="Evaluate against these ground-truth CSVs, e.g. ground_truth/cadets_e3",
    )
    parser.add_argument("--run-id", help="Output name (default: <model-id>_<UTC timestamp>)")
    parser.add_argument(
        "--force-reload", action="store_true", help="Rebuild cached graphs from the DB"
    )
    args = parser.parse_args()
    if not (args.db_name or args.model_id):
        parser.error("pass --db-name and/or --model-id")
    return args


def main() -> None:
    args = parse_args()
    out_dir = stages.infer(
        inference_start=args.inference_start,
        inference_end=args.inference_end,
        db_name=args.db_name,
        model_id=args.model_id,
        ground_truth_dir=args.ground_truth_dir,
        run_id=args.run_id,
        force_reload=args.force_reload,
    )
    print(f"RUN_DIR={out_dir}")


if __name__ == "__main__":
    main()
