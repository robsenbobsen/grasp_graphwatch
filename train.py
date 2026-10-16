"""Pretrain a GRASP model bundle on one database and time range (see grasp/stages.py).

    python train.py --db-name cadets_e3 \
        --train-start "2018-04-02 00:00:00" --train-end "2018-04-05 23:59:59"
"""

import argparse
import logging

from grasp import stages

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRASP model bundle")
    parser.add_argument("--db-name", required=True, help="Database to read events from")
    parser.add_argument("--train-start", required=True, help='e.g. "2018-04-02 00:00:00"')
    parser.add_argument("--train-end", required=True, help='e.g. "2018-04-05 23:59:59"')
    parser.add_argument(
        "--experiment-config",
        "-e",
        help="Hyperparameters/settings (default: the dataset's *_default experiment)",
    )
    parser.add_argument("--model-id", help="Bundle name (default: <db-name>_<UTC timestamp>)")
    parser.add_argument(
        "--force-reload", action="store_true", help="Rebuild cached graphs from the DB"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = stages.train(
        db_name=args.db_name,
        train_start=args.train_start,
        train_end=args.train_end,
        experiment_config=args.experiment_config,
        model_id=args.model_id,
        force_reload=args.force_reload,
    )
    print(f"MODEL_ID={bundle.path.name}")


if __name__ == "__main__":
    main()
