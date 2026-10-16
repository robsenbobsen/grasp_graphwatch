import os
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv

from grasp.schema import (
    Atlasv2Operation,
    Carbanakv2Operation,
    DatasetName,
    OptcOperation,
    PathTransformerConstants,
    SpadeGraspOperation,
    SysdigOperation,
    TcOperation,
    TcOperationModified,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ENV_EXPERIMENT_CONFIG = "GRASP_EXPERIMENT_CONFIG"
DEFAULT_EXPERIMENT_FILENAME = "experiment_cadets_e3.yaml"
EXPERIMENTS_DIR = PROJECT_ROOT / "grasp" / "experiments"


def resolve_experiment_config_path(
    override: str | Path | None = None,
) -> Path:
    """Resolve the experiment config path with overrides/env/defaults."""
    if override:
        candidate = Path(override)
    else:
        env_path = os.getenv(ENV_EXPERIMENT_CONFIG)
        candidate = Path(env_path) if env_path else EXPERIMENTS_DIR / DEFAULT_EXPERIMENT_FILENAME

    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()

    return candidate


DB_USER: str = os.getenv("db_user", "default_user")
DB_PASSWORD: str = os.getenv("db_password", "default_password")
DB_HOST: str = os.getenv("db_host", "localhost")
DB_PORT: str = os.getenv("db_port", "5432")
DB_NAME_FROM_ENV: str = os.getenv("db_name", "default_db")

OPTC_OPERATIONS: dict[int, str] = {idx: op.value for idx, op in enumerate(OptcOperation)}

TC_OPERATIONS: dict[int, str] = {idx: op.value for idx, op in enumerate(TcOperation)}
TC_OPERATIONS_MODIFIED: dict[int, str] = {
    idx: op.value for idx, op in enumerate(TcOperationModified)
}

ATLASV2_EDR_OPERATIONS: dict[int, str] = {idx: op.value for idx, op in enumerate(Atlasv2Operation)}

CARBANAKV2_EDR_OPERATIONS: dict[int, str] = {
    idx: op.value for idx, op in enumerate(Carbanakv2Operation)
}

SPADE_GRASP_OPERATIONS: dict[int, str] = {
    idx: op.value for idx, op in enumerate(SpadeGraspOperation)
}

SYSDIG_OPERATIONS: dict[int, str] = {idx: op.value for idx, op in enumerate(SysdigOperation)}

EXPERIMENT_CONFIG_PATH = resolve_experiment_config_path()


def load_experiment_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load experiment config from the provided path or resolved default."""
    path_to_load = resolve_experiment_config_path(config_path)
    if path_to_load.exists():
        with open(path_to_load, "r") as f:
            return yaml.safe_load(f)
    return {}


EXPERIMENT_CONFIG: dict[str, Any] = load_experiment_config(EXPERIMENT_CONFIG_PATH)

DEFAULT_DATASET = DatasetName.CADETS_E3


RAW_DATASET_NAME: str = EXPERIMENT_CONFIG.get("dataset", {}).get(  # type: ignore
    "name", DEFAULT_DATASET.value
)

try:
    DATASET_NAME_ENUM = DatasetName(RAW_DATASET_NAME)
except ValueError as exc:
    allowed = [item.value for item in DatasetName]
    raise ValueError(
        f"Invalid dataset name: {RAW_DATASET_NAME}. Must be one of {allowed}."
    ) from exc

DATASET_NAME: str = DATASET_NAME_ENUM.value
DB_DATASET_NAME: str = DATASET_NAME_ENUM.value

if DB_NAME_FROM_ENV != "default_db":
    DB_DATASET_NAME: str = DB_NAME_FROM_ENV


EXPERIMENT_PREFIX: str = EXPERIMENT_CONFIG.get("experiment_prefix", "default")

DB_URL: str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATASET_NAME}"

ATTACK_WINDOWS_FOR_VISUALIZATION: list[tuple[str, str]] = EXPERIMENT_CONFIG.get(
    "attack_windows_for_visualization", []
)


class DetectionConfig:
    NUM_NEIGHBORS = EXPERIMENT_CONFIG.get("detection", {}).get("num_neighbors", [-1, -1])
    BATCH_SIZE = EXPERIMENT_CONFIG.get("detection", {}).get("batch_size", 64)
    EPOCHS = EXPERIMENT_CONFIG.get("detection", {}).get("epochs", 4)
    DEVICE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    LEARNING_RATE = EXPERIMENT_CONFIG.get("detection", {}).get("learning_rate", 0.01)
    HIDDEN_DIM = EXPERIMENT_CONFIG.get("detection", {}).get("hidden_dim", 128)
    HEADS = EXPERIMENT_CONFIG.get("detection", {}).get("heads", 4)
    NUM_LAYERS = EXPERIMENT_CONFIG.get("detection", {}).get("num_layers", 2)
    DROPOUT = EXPERIMENT_CONFIG.get("detection", {}).get("dropout", 0.1)
    WEIGHT_DECAY = EXPERIMENT_CONFIG.get("detection", {}).get("weight_decay", 0.0001)
    SCHEDULER_FACTOR = EXPERIMENT_CONFIG.get("detection", {}).get("scheduler_factor", 0.5)
    SCHEDULER_PATIENCE = EXPERIMENT_CONFIG.get("detection", {}).get("scheduler_patience", 5)
    SCHEDULER_MODE = EXPERIMENT_CONFIG.get("detection", {}).get("scheduler_mode", "min")
    SHUFFLE_TRAIN_PATHS = EXPERIMENT_CONFIG.get("detection", {}).get("shuffle_train_paths", False)
    SHUFFLE_TRAIN_BATCHES = EXPERIMENT_CONFIG.get("detection", {}).get(
        "shuffle_train_batches", True
    )
    USE_CLASS_WEIGHTS = EXPERIMENT_CONFIG.get("detection", {}).get("use_class_weights", False)
    SUBSET_FINETUNE_ENABLED = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_enabled", False
    )
    SUBSET_FINETUNE_EPOCHS = EXPERIMENT_CONFIG.get("detection", {}).get("subset_finetune_epochs", 1)
    SUBSET_FINETUNE_LEARNING_RATE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_learning_rate", None
    )
    SUBSET_FINETUNE_LR_FACTOR = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_lr_factor", 0.1
    )
    SUBSET_FINETUNE_FREEZE_BACKBONE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_freeze_backbone", True
    )


class LocationTransformerConfig:
    MODEL_TYPE: str = EXPERIMENT_CONFIG.get("location_transformer", {}).get(
        "model_type", "transformer"
    )
    MAX_LOCATION_LENGTH: int = EXPERIMENT_CONFIG.get("location_transformer", {}).get(
        "max_location_length", 100
    )  # type: ignore
    CHARS: str = os.getenv("CHARS", PathTransformerConstants.CHARS.value)
    PADDING_CHAR: int = int(
        os.getenv(
            "PADDING_CHAR",
            PathTransformerConstants.PADDING_CHAR.value,
        )
    )  # 0

    EMBEDDING_DIM: int = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("embedding_dim", 128)
    )  # type: ignore
    HIDDEN_DIM: int = int(EXPERIMENT_CONFIG.get("location_transformer", {}).get("hidden_dim", 256))  # type: ignore
    NUM_LAYERS: int = int(EXPERIMENT_CONFIG.get("location_transformer", {}).get("num_layers", 2))  # type: ignore
    NHEAD: int = int(EXPERIMENT_CONFIG.get("location_transformer", {}).get("nhead", 4))  # type: ignore
    NUM_EPOCHS: int = int(EXPERIMENT_CONFIG.get("location_transformer", {}).get("num_epochs", 10))  # type: ignore
    PATIENCE: int = int(EXPERIMENT_CONFIG.get("location_transformer", {}).get("patience", 3))  # type: ignore
    BATCH_SIZE: int = int(EXPERIMENT_CONFIG.get("location_transformer", {}).get("batch_size", 256))  # type: ignore
    POS_ENC_MAX_LEN: int = int(os.getenv("positional_encoding_max_len", "5000"))
    POS_ENC_DIV_TERM_HYPERPARAM: float = float(
        os.getenv("positional_encoding_div_term_hyperparam", "10000.0")
    )
    LEARNING_RATE: float = float(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("learning_rate", 0.001)
    )  # type: ignore
    WORD2VEC_WINDOW_SIZE: int = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("word2vec_window_size", 2)
    )
    WORD2VEC_MIN_COUNT: int = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("word2vec_min_count", 1)
    )
    WORD2VEC_NEGATIVE_SAMPLES: int = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("word2vec_negative_samples", 5)
    )
    WORD2VEC_LEARNING_RATE: float = float(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("word2vec_learning_rate", 0.025)
    )
    DEVICE: str = EXPERIMENT_CONFIG.get("location_transformer", {}).get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )


class GraphConfig:
    _times_cfg = EXPERIMENT_CONFIG.get("times", {})
    _storage_cfg = EXPERIMENT_CONFIG.get("storage", {})
    _graph_mgmt_cfg = EXPERIMENT_CONFIG.get("graph_management", {})

    CONTEXT_SIZE: int = int(_times_cfg.get("context_size", 120))
    STEP_SIZE: int = int(_times_cfg.get("step_size", 60))
    TRAIN_START_TIMES: list[str] = _times_cfg.get(
        "train_start_times",
        _times_cfg.get("train_start", ["2024-01-01 00:00:00"]),
    )
    TRAIN_END_TIMES: list[str] = _times_cfg.get(
        "train_end_times",
        _times_cfg.get("train_end", ["2024-01-31 23:59:59"]),
    )
    TEST_START_TIMES: list[str] = _times_cfg.get(
        "test_start_times",
        _times_cfg.get("test_start", ["2024-02-01 00:00:00"]),
    )
    TEST_END_TIMES: list[str] = _times_cfg.get(
        "test_end_times",
        _times_cfg.get("test_end", ["2024-02-15 23:59:59"]),
    )
    GRAPH_ROOT_DIR: str = _storage_cfg.get("graph_root_dir", "./data/graphs")
    FORCE_RELOAD: bool = bool(
        _storage_cfg.get("force_reload", _graph_mgmt_cfg.get("force_reload", False))
    )


class GroundTruthConfig:
    GROUND_TRUTH_BASE_DIR: str = EXPERIMENT_CONFIG.get("ground_truth", {}).get(
        "base_dir", "./data/ground_truth"
    )  # type: ignore


def set_db_name(db_name: str) -> None:
    """Point DB_URL at another database on the configured server (overrides env db_name)."""
    global DB_DATASET_NAME
    global DB_URL

    DB_DATASET_NAME = db_name
    DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATASET_NAME}"


def reload_experiment_config(config_path: str | Path | None = None) -> None:
    global EXPERIMENT_CONFIG_PATH
    global EXPERIMENT_CONFIG
    global RAW_DATASET_NAME
    global DATASET_NAME_ENUM
    global DATASET_NAME
    global DB_DATASET_NAME
    global EXPERIMENT_PREFIX
    global DB_URL
    global ATTACK_WINDOWS_FOR_VISUALIZATION

    EXPERIMENT_CONFIG_PATH = resolve_experiment_config_path(config_path)  # type: ignore
    EXPERIMENT_CONFIG = load_experiment_config(EXPERIMENT_CONFIG_PATH)

    RAW_DATASET_NAME = EXPERIMENT_CONFIG.get("dataset", {}).get(  # type: ignore
        "name", DEFAULT_DATASET.value
    )

    try:
        DATASET_NAME_ENUM = DatasetName(RAW_DATASET_NAME)
    except ValueError as exc:  # pragma: no cover - validated once here
        allowed = [item.value for item in DatasetName]
        raise ValueError(
            f"Invalid dataset name: {RAW_DATASET_NAME}. Must be one of {allowed}."
        ) from exc

    DATASET_NAME = DATASET_NAME_ENUM.value
    DB_DATASET_NAME = DATASET_NAME_ENUM.value

    if DB_NAME_FROM_ENV != "default_db":
        DB_DATASET_NAME = DB_NAME_FROM_ENV

    EXPERIMENT_PREFIX = EXPERIMENT_CONFIG.get("experiment_prefix", "default")

    DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATASET_NAME}"

    ATTACK_WINDOWS_FOR_VISUALIZATION = EXPERIMENT_CONFIG.get("attack_windows_for_visualization", [])

    # Refresh DetectionConfig
    DetectionConfig.NUM_NEIGHBORS = EXPERIMENT_CONFIG.get("detection", {}).get(
        "num_neighbors", [-1, -1]
    )
    DetectionConfig.BATCH_SIZE = EXPERIMENT_CONFIG.get("detection", {}).get("batch_size", 64)
    DetectionConfig.EPOCHS = EXPERIMENT_CONFIG.get("detection", {}).get("epochs", 4)
    DetectionConfig.DEVICE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    DetectionConfig.LEARNING_RATE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "learning_rate", 0.01
    )
    DetectionConfig.HIDDEN_DIM = EXPERIMENT_CONFIG.get("detection", {}).get("hidden_dim", 128)
    DetectionConfig.HEADS = EXPERIMENT_CONFIG.get("detection", {}).get("heads", 4)
    DetectionConfig.NUM_LAYERS = EXPERIMENT_CONFIG.get("detection", {}).get("num_layers", 2)
    DetectionConfig.DROPOUT = EXPERIMENT_CONFIG.get("detection", {}).get("dropout", 0.1)
    DetectionConfig.WEIGHT_DECAY = EXPERIMENT_CONFIG.get("detection", {}).get(
        "weight_decay", 0.0001
    )
    DetectionConfig.SCHEDULER_FACTOR = EXPERIMENT_CONFIG.get("detection", {}).get(
        "scheduler_factor", 0.5
    )
    DetectionConfig.SCHEDULER_PATIENCE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "scheduler_patience", 5
    )
    DetectionConfig.SCHEDULER_MODE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "scheduler_mode", "min"
    )
    DetectionConfig.SHUFFLE_TRAIN_PATHS = EXPERIMENT_CONFIG.get("detection", {}).get(
        "shuffle_train_paths", False
    )
    DetectionConfig.SHUFFLE_TRAIN_BATCHES = EXPERIMENT_CONFIG.get("detection", {}).get(
        "shuffle_train_batches", True
    )
    DetectionConfig.USE_CLASS_WEIGHTS = EXPERIMENT_CONFIG.get("detection", {}).get(
        "use_class_weights", False
    )
    DetectionConfig.SUBSET_FINETUNE_ENABLED = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_enabled", False
    )
    DetectionConfig.SUBSET_FINETUNE_EPOCHS = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_epochs", 1
    )
    DetectionConfig.SUBSET_FINETUNE_LEARNING_RATE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_learning_rate", None
    )
    DetectionConfig.SUBSET_FINETUNE_LR_FACTOR = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_lr_factor", 0.1
    )
    DetectionConfig.SUBSET_FINETUNE_FREEZE_BACKBONE = EXPERIMENT_CONFIG.get("detection", {}).get(
        "subset_finetune_freeze_backbone", True
    )

    # Refresh LocationTransformerConfig
    LocationTransformerConfig.MODEL_TYPE = EXPERIMENT_CONFIG.get("location_transformer", {}).get(
        "model_type", "transformer"
    )
    LocationTransformerConfig.MAX_LOCATION_LENGTH = EXPERIMENT_CONFIG.get(
        "location_transformer", {}
    ).get("max_location_length", 100)  # type: ignore
    LocationTransformerConfig.EMBEDDING_DIM = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("embedding_dim", 128)
    )
    LocationTransformerConfig.HIDDEN_DIM = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("hidden_dim", 256)
    )
    LocationTransformerConfig.NUM_LAYERS = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("num_layers", 2)
    )
    LocationTransformerConfig.NHEAD = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("nhead", 4)
    )
    LocationTransformerConfig.NUM_EPOCHS = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("num_epochs", 10)
    )
    LocationTransformerConfig.PATIENCE = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("patience", 3)
    )
    LocationTransformerConfig.BATCH_SIZE = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("batch_size", 256)
    )
    LocationTransformerConfig.LEARNING_RATE = float(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("learning_rate", 0.001)
    )
    LocationTransformerConfig.WORD2VEC_WINDOW_SIZE = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("word2vec_window_size", 2)
    )
    LocationTransformerConfig.WORD2VEC_MIN_COUNT = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("word2vec_min_count", 1)
    )
    LocationTransformerConfig.WORD2VEC_NEGATIVE_SAMPLES = int(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("word2vec_negative_samples", 5)
    )
    LocationTransformerConfig.WORD2VEC_LEARNING_RATE = float(
        EXPERIMENT_CONFIG.get("location_transformer", {}).get("word2vec_learning_rate", 0.025)
    )
    LocationTransformerConfig.DEVICE = EXPERIMENT_CONFIG.get("location_transformer", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    LocationTransformerConfig.POS_ENC_MAX_LEN = int(
        os.getenv("positional_encoding_max_len", "5000")
    )
    LocationTransformerConfig.POS_ENC_DIV_TERM_HYPERPARAM = float(
        os.getenv("positional_encoding_div_term_hyperparam", "10000.0")
    )

    # Refresh GraphConfig
    GraphConfig._times_cfg = EXPERIMENT_CONFIG.get("times", {})
    GraphConfig._storage_cfg = EXPERIMENT_CONFIG.get("storage", {})
    GraphConfig._graph_mgmt_cfg = EXPERIMENT_CONFIG.get("graph_management", {})

    GraphConfig.CONTEXT_SIZE = int(GraphConfig._times_cfg.get("context_size", 120))
    GraphConfig.STEP_SIZE = int(GraphConfig._times_cfg.get("step_size", 60))
    GraphConfig.TRAIN_START_TIMES = GraphConfig._times_cfg.get(
        "train_start_times",
        GraphConfig._times_cfg.get("train_start", ["2024-01-01 00:00:00"]),
    )
    GraphConfig.TRAIN_END_TIMES = GraphConfig._times_cfg.get(
        "train_end_times",
        GraphConfig._times_cfg.get("train_end", ["2024-01-31 23:59:59"]),
    )
    GraphConfig.TEST_START_TIMES = GraphConfig._times_cfg.get(
        "test_start_times",
        GraphConfig._times_cfg.get("test_start", ["2024-02-01 00:00:00"]),
    )
    GraphConfig.TEST_END_TIMES = GraphConfig._times_cfg.get(
        "test_end_times",
        GraphConfig._times_cfg.get("test_end", ["2024-02-15 23:59:59"]),
    )
    GraphConfig.GRAPH_ROOT_DIR = GraphConfig._storage_cfg.get("graph_root_dir", "./data/graphs")
    GraphConfig.FORCE_RELOAD = bool(
        GraphConfig._storage_cfg.get(
            "force_reload",
            GraphConfig._graph_mgmt_cfg.get("force_reload", False),
        )
    )

    # Refresh GroundTruthConfig
    GroundTruthConfig.GROUND_TRUTH_BASE_DIR = EXPERIMENT_CONFIG.get("ground_truth", {}).get(
        "base_dir", "./data/ground_truth"
    )
