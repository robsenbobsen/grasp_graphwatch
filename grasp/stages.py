"""Split GRASP pipeline: train a model bundle once, run inference with it later.

A model bundle is a directory holding everything inference needs to reproduce the
training preprocessing exactly:

    metadata.json         db/dataset, train windows, settings, metrics
    experiment.yaml       effective experiment config used for training
    location_encoder.pt   trained location encoder (transformer or word2vec)
    train_cmd_to_id.json  command -> class id mapping of the training data
    classifier.pt         GAT init kwargs + weights
    cluster_manager.pt    misclassification clusters of the training data
    graph_storage.pt      training graph paths + command mapping (for the workbench)
    train_metrics.json    training metrics incl. per-class report (for the workbench)
    extended_graphs/      cached preprocessed graphs (per bundle, since they depend on it)

Inference rebuilds raw graphs for new windows from the DB, extends them with the bundle's
location encoder and command mapping, classifies them and reports anomalies; evaluation
against ground truth is optional.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from grasp import config
from grasp.detection import classification, classification_pipeline, clustering
from grasp.detection.classification_storage import ClassificationStorage
from grasp.evaluation import evaluation_storage, ground_truth_storage
from grasp.graph.graph_management import GraphManager
from grasp.graph.graph_storage import GraphStorage, load_graph_storage
from grasp.graph.location_transformer import LocationEncoder
from grasp.reporting import detailed_report, report, visualization
from grasp.schema import DatasetName
from grasp.utils import detection_helpers
from grasp.utils.evaluation_helpers import (
    compute_metrics_multiclass,
    create_cmd_to_id_inverse,
    get_anomalies,
    get_labels_from_cmd_ids,
)
from grasp.utils.graph_helpers import load_graph_data

logger = logging.getLogger(__name__)

PROJECT_ROOT = config.PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "data" / "models"
INFERENCE_DIR = PROJECT_ROOT / "data" / "inference"
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(name: str) -> str:
    return _SAFE_NAME.sub("_", name)


def dataset_for_db(db_name: str) -> str:
    """Dataset type (operation schema, timezone, ...) of a database, e.g. sysdig_t2 -> sysdig."""
    names = sorted((d.value for d in DatasetName), key=len, reverse=True)
    for name in names:
        if db_name == name or db_name.startswith(name + "_"):
            return name
    raise ValueError(
        f"Cannot derive the dataset type from db_name '{db_name}'; "
        f"pass an experiment config whose dataset.name is one of {[d.value for d in DatasetName]}."
    )


def default_experiment_config(dataset: str) -> Path | None:
    candidate = (
        config.EXPERIMENTS_DIR / "all_experiments" / f"{dataset}_default" / f"experiment_{dataset}.yaml"
    )
    return candidate if candidate.exists() else None


def _apply_config(experiment_config: str | Path, db_name: str) -> None:
    config.reload_experiment_config(experiment_config)
    config.set_db_name(db_name)


def _graph_manager(
    db_name: str,
    train_windows: list[tuple[str, str]],
    test_windows: list[tuple[str, str]],
    extended_root_dir: Path,
    encoder_path: Path,
    force_reload: bool,
) -> GraphManager:
    ltc = config.LocationTransformerConfig
    return GraphManager(
        dataset_name=config.DATASET_NAME,
        graph_prefix=db_name,
        train_start_times=[w[0] for w in train_windows],
        train_end_times=[w[1] for w in train_windows],
        test_start_times=[w[0] for w in test_windows],
        test_end_times=[w[1] for w in test_windows],
        context_size=config.GraphConfig.CONTEXT_SIZE,
        step_size=config.GraphConfig.STEP_SIZE,
        force_reload=force_reload,
        root_dir=str(GRAPHS_DIR / _safe(db_name)),
        extended_root_dir=str(extended_root_dir),
        autoencoder_embedding_dim=ltc.EMBEDDING_DIM,
        autoencoder_hidden_dim=ltc.HIDDEN_DIM,
        autoencoder_num_epochs=ltc.NUM_EPOCHS,
        autoencoder_patience=ltc.PATIENCE,
        autoencoder_batch_size=ltc.BATCH_SIZE,
        autoencoder_num_layers=ltc.NUM_LAYERS,
        autoencoder_nhead=ltc.NHEAD,
        autoencoder_model_file_path=str(encoder_path),
        location_embedding_model_type=ltc.MODEL_TYPE,
        word2vec_window_size=ltc.WORD2VEC_WINDOW_SIZE,
        word2vec_min_count=ltc.WORD2VEC_MIN_COUNT,
        word2vec_negative_samples=ltc.WORD2VEC_NEGATIVE_SAMPLES,
        word2vec_learning_rate=ltc.WORD2VEC_LEARNING_RATE,
        autoencoder_device=ltc.DEVICE,
    )


@dataclass
class ModelBundle:
    path: Path

    @property
    def metadata_file(self) -> Path:
        return self.path / "metadata.json"

    @property
    def experiment_config(self) -> Path:
        return self.path / "experiment.yaml"

    @property
    def location_encoder(self) -> Path:
        return self.path / "location_encoder.pt"

    @property
    def train_cmd_to_id(self) -> Path:
        return self.path / "train_cmd_to_id.json"

    @property
    def classifier(self) -> Path:
        return self.path / "classifier.pt"

    @property
    def cluster_manager(self) -> Path:
        return self.path / "cluster_manager.pt"

    @property
    def extended_graphs(self) -> Path:
        return self.path / "extended_graphs"

    @property
    def graph_storage(self) -> Path:
        return self.path / "graph_storage.pt"

    @property
    def train_metrics(self) -> Path:
        return self.path / "train_metrics.json"

    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_file.read_text(encoding="utf-8"))

    def is_complete(self) -> bool:
        return self.metadata_file.exists() and self.metadata().get("status") == "complete"


def train_graph_storage(bundle: ModelBundle) -> GraphStorage:
    """Training graphs + command mapping of a bundle.

    Bundles trained before graph_storage.pt was saved are reconstructed from their
    extended training graphs and train_cmd_to_id.json.
    """
    if bundle.graph_storage.exists():
        return load_graph_storage(str(bundle.graph_storage))
    gs = GraphStorage()
    gs.extended_train_data_paths = [
        str(p) for p in sorted((bundle.extended_graphs / "processed").glob("*_train_graph_*_extended.pt"))
    ]
    gs.train_subject_cmd_to_id = json.loads(bundle.train_cmd_to_id.read_text(encoding="utf-8"))
    return gs


def list_models() -> list[dict[str, Any]]:
    models = []
    if MODELS_DIR.exists():
        for path in MODELS_DIR.iterdir():
            bundle = ModelBundle(path)
            if path.is_dir() and bundle.metadata_file.exists():
                models.append(bundle.metadata())
    return sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)


def resolve_model(model_id: str | None = None, db_name: str | None = None) -> ModelBundle:
    """A model by id, or the most recently trained complete model for db_name."""
    if model_id:
        bundle = ModelBundle(MODELS_DIR / _safe(model_id))
        if not bundle.is_complete():
            raise FileNotFoundError(f"No complete trained model '{model_id}' in {MODELS_DIR}")
        return bundle
    for meta in list_models():
        if meta.get("status") == "complete" and meta.get("db_name") == db_name:
            return ModelBundle(MODELS_DIR / meta["model_id"])
    raise FileNotFoundError(f"No trained model for db_name '{db_name}' in {MODELS_DIR}")


def train(
    db_name: str,
    train_start: str,
    train_end: str,
    experiment_config: str | Path | None = None,
    model_id: str | None = None,
    force_reload: bool = False,
) -> ModelBundle:
    """Pretrain: build training graphs, fit location encoder + classifier, save a bundle."""
    dataset = dataset_for_db(db_name)
    experiment_config = experiment_config or default_experiment_config(dataset)
    if experiment_config is None:
        raise ValueError(f"No default experiment config for dataset '{dataset}'; pass one.")
    _apply_config(experiment_config, db_name)
    if config.DATASET_NAME != dataset:
        raise ValueError(
            f"Experiment config is for dataset '{config.DATASET_NAME}', but db '{db_name}' is '{dataset}'."
        )

    model_id = _safe(model_id or f"{db_name}_{_now_id()}")
    bundle = ModelBundle(MODELS_DIR / model_id)
    if bundle.path.exists():
        raise FileExistsError(f"Model '{model_id}' already exists at {bundle.path}")
    bundle.path.mkdir(parents=True)

    metadata: dict[str, Any] = {
        "model_id": model_id,
        "status": "training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db_name": db_name,
        "dataset": config.DATASET_NAME,
        "train_start": train_start,
        "train_end": train_end,
        "context_size": config.GraphConfig.CONTEXT_SIZE,
        "step_size": config.GraphConfig.STEP_SIZE,
        "source_experiment_config": str(experiment_config),
    }
    bundle.metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    bundle.experiment_config.write_text(
        yaml.safe_dump(config.EXPERIMENT_CONFIG, sort_keys=False), encoding="utf-8"
    )
    try:
        _train_into_bundle(bundle, metadata, db_name, train_start, train_end, force_reload)
    except BaseException as exc:
        metadata.update({"status": "failed", "error": repr(exc)})
        bundle.metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    logger.info("Model bundle saved to %s", bundle.path)
    return bundle


def _train_into_bundle(
    bundle: ModelBundle,
    metadata: dict[str, Any],
    db_name: str,
    train_start: str,
    train_end: str,
    force_reload: bool,
) -> None:
    gm = _graph_manager(
        db_name=db_name,
        train_windows=[(train_start, train_end)],
        test_windows=[],
        extended_root_dir=bundle.extended_graphs,
        encoder_path=bundle.location_encoder,
        force_reload=force_reload,
    )
    gm.setup_windows()
    gm.build_train_graphs()
    gs = gm.graph_storage
    if not gs.train_data_paths:
        raise RuntimeError(f"No events in '{db_name}' between {train_start} and {train_end}.")
    gm.build_vocabulary()
    location_encoder = gm.train_location_encoder()
    # Persist the whole encoder object so inference needs no architecture parameters.
    torch.save(location_encoder, bundle.location_encoder)
    bundle.train_cmd_to_id.write_text(
        json.dumps(gs.train_subject_cmd_to_id, indent=2), encoding="utf-8"
    )
    gs.extended_train_data_paths = gm.extend_graphs(
        gs.train_data_paths, location_encoder, gs.train_subject_cmd_to_id, gm.experiment_prefix_train
    )

    dc = config.DetectionConfig
    device = torch.device(dc.DEVICE)
    model = classification_pipeline.window_based_train(
        graph_storage=gs,
        batch_size=dc.BATCH_SIZE,
        num_neighbors=dc.NUM_NEIGHBORS,
        epochs=dc.EPOCHS,
        hidden_channels=dc.HIDDEN_DIM,
        num_layers=dc.NUM_LAYERS,
        heads=dc.HEADS,
        dropout=dc.DROPOUT,
        device=device,
        learning_rate=dc.LEARNING_RATE,
        weight_decay=dc.WEIGHT_DECAY,
        shuffle_train_paths=dc.SHUFFLE_TRAIN_PATHS,
        shuffle_train_batches=dc.SHUFFLE_TRAIN_BATCHES,
        use_class_weights=dc.USE_CLASS_WEIGHTS,
        enable_subset_finetune=dc.SUBSET_FINETUNE_ENABLED,
        subset_finetune_epochs=dc.SUBSET_FINETUNE_EPOCHS,
        subset_finetune_learning_rate=dc.SUBSET_FINETUNE_LEARNING_RATE,
        subset_finetune_lr_factor=dc.SUBSET_FINETUNE_LR_FACTOR,
        subset_finetune_freeze_backbone=dc.SUBSET_FINETUNE_FREEZE_BACKBONE,
    )
    torch.save(
        {
            "init_kwargs": {
                "in_channels": model.in_channels,
                "hidden_channels": model.hidden_channels,
                "out_channels": model.out_channels,
                "num_layers": model.num_layers,
                "heads": model.heads,
                "edge_dim": model.edge_dim,
                "dropout": model.dropout,
            },
            "state_dict": model.state_dict(),
        },
        bundle.classifier,
    )

    cls_storage_train = classification_pipeline.window_based_test(
        model=model,
        graph_storage=gs,
        batch_size=dc.BATCH_SIZE,
        num_neighbors=dc.NUM_NEIGHBORS,
        device=device,
        for_train_data=True,
    )
    train_metrics = compute_metrics_multiclass(
        true_labels=cls_storage_train.y, pred_labels=cls_storage_train.y_hat
    )
    logger.info("Training metrics: %s", train_metrics)

    cluster_manager = clustering.ClusterManager(classification_storage=cls_storage_train)
    cluster_manager.create_misclassification_clusters()
    cluster_manager.save_to_disk(str(bundle.cluster_manager))
    gs.save_graph_storage(str(bundle.graph_storage))
    bundle.train_metrics.write_text(json.dumps(train_metrics, indent=2, default=str), encoding="utf-8")

    metadata.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "train_windows": len(gs.train_data_paths),
            "num_classes": len(gs.train_subject_cmd_to_id),
            "train_metrics": {
                k: v for k, v in train_metrics.items() if k != "classification_report"
            },
        }
    )
    bundle.metadata_file.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _load_classifier(bundle: ModelBundle, device: torch.device) -> classification.GAT:
    payload = torch.load(bundle.classifier, map_location=device)
    model = classification.GAT(**payload["init_kwargs"], device=device)
    model.load_state_dict(payload["state_dict"])
    return model.to(device)


def infer(
    inference_start: str,
    inference_end: str,
    db_name: str | None = None,
    model_id: str | None = None,
    ground_truth_dir: str | Path | None = None,
    run_id: str | None = None,
    force_reload: bool = False,
) -> Path:
    """Classify a new time range with a trained bundle; returns the output directory.

    db_name defaults to the model's training database; without model_id the latest model
    trained on db_name is used.
    """
    if not (db_name or model_id):
        raise ValueError("Pass db_name and/or model_id.")
    bundle = resolve_model(model_id=model_id, db_name=db_name)
    meta = bundle.metadata()
    db_name = db_name or meta["db_name"]
    if dataset_for_db(db_name) != meta["dataset"]:
        raise ValueError(
            f"Model '{meta['model_id']}' was trained on dataset '{meta['dataset']}', "
            f"db '{db_name}' is '{dataset_for_db(db_name)}'."
        )
    _apply_config(bundle.experiment_config, db_name)

    run_id = _safe(run_id or f"{meta['model_id']}_{_now_id()}")
    out_dir = INFERENCE_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Inference with model %s on %s, output %s", meta["model_id"], db_name, out_dir)

    location_encoder: LocationEncoder = torch.load(bundle.location_encoder, weights_only=False)
    train_cmd_to_id: dict[str, int] = json.loads(bundle.train_cmd_to_id.read_text(encoding="utf-8"))

    gm = _graph_manager(
        db_name=db_name,
        train_windows=[],
        test_windows=[(inference_start, inference_end)],
        extended_root_dir=bundle.extended_graphs,
        encoder_path=bundle.location_encoder,
        force_reload=force_reload,
    )
    gm.setup_windows()
    gm.build_test_graphs()
    gs: GraphStorage = gm.graph_storage
    gs.train_subject_cmd_to_id = train_cmd_to_id
    if not gs.test_data_paths:
        raise RuntimeError(f"No events in '{db_name}' between {inference_start} and {inference_end}.")
    # Same preprocessing as training: the bundle's encoder and command mapping. Commands
    # unseen in training get an all-zero label, exactly as during training.
    gs.extended_test_data_paths = gm.extend_graphs(
        gs.test_data_paths, location_encoder, train_cmd_to_id, gm.experiment_prefix_test
    )

    dc = config.DetectionConfig
    device = torch.device(dc.DEVICE)
    model = _load_classifier(bundle, device)
    first = load_graph_data(gs.extended_test_data_paths[0])
    if first.x.shape[1] != model.in_channels:  # type: ignore[union-attr]
        raise RuntimeError(
            f"Feature size {first.x.shape[1]} does not match the model ({model.in_channels})."  # type: ignore[union-attr]
        )

    cls_storage: ClassificationStorage = classification_pipeline.window_based_test(
        model=model,
        graph_storage=gs,
        batch_size=dc.BATCH_SIZE,
        num_neighbors=dc.NUM_NEIGHBORS,
        device=device,
        for_train_data=False,
    )
    cluster_manager = clustering.ClusterManager(classification_storage=ClassificationStorage())
    cluster_manager.load_from_disk(str(bundle.cluster_manager))
    detection_helpers.check_classification_with_clusters(
        cluster_manager=cluster_manager, cls_storage=cls_storage
    )
    cls_storage.save_to_file(str(out_dir / "classification.pt"))

    _write_anomalies(cls_storage, train_cmd_to_id, out_dir / "anomalies.json")

    es = evaluation_storage.EvaluationStorage(cs=cls_storage, train_subject_cmd_to_id=train_cmd_to_id)
    gts = None
    if ground_truth_dir:
        gts = ground_truth_storage.GroundTruthStorage(base_path=PROJECT_ROOT / ground_truth_dir)
        es.get_y_and_y_hat_per_gt_group(gts=gts, cs=cls_storage)
        es.get_y_and_y_hat_per_gt_file(gts=gts, cs=cls_storage)
        es.save_to_file(str(out_dir / "evaluation_storage.pt"))

    r = report.Report(
        classification_storage=cls_storage,
        evaluation_storage=es if gts else None,
        ground_truth_storage=gts,
        graph_storage=gs,
    )
    r.save(str(out_dir / "report.json"))
    detailed_report_path = out_dir / "detailed_report.json"
    detailed_report.DetailedReport(
        classification_storage=cls_storage,
        evaluation_storage=es if gts else None,
        ground_truth_storage=gts,
        graph_storage=gs,
    ).save(str(detailed_report_path))
    if gts:
        visualization.visualize_misclassification_timeframes(
            detailed_report_path=detailed_report_path,
            output_path=out_dir / "misclassifications_over_time.png",
            attack_windows=config.ATTACK_WINDOWS_FOR_VISUALIZATION,
            min_duration_seconds=0,
            to_display=False,
            x_tick_count=20,
        )

    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_id": meta["model_id"],
                "db_name": db_name,
                "inference_start": inference_start,
                "inference_end": inference_end,
                "ground_truth_dir": str(ground_truth_dir) if ground_truth_dir else None,
                "windows": len(gs.test_data_paths),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Inference results saved to %s", out_dir)
    return out_dir


def _write_anomalies(
    cls_storage: ClassificationStorage, train_cmd_to_id: dict[str, int], path: Path
) -> None:
    """Subject nodes whose observed command differs from the (cluster-corrected) prediction."""
    inverse = create_cmd_to_id_inverse(train_cmd_to_id)
    predicted = get_labels_from_cmd_ids(cls_storage.y_hat_cluster_corrected, inverse)
    rows = [
        {
            "node_uuid": cls_storage.node_uuids[i],
            "node_id": cls_storage.node_ids[i],
            "observed_cmd": cls_storage.true_cmd_labels[i],
            "predicted_cmd": predicted[i],
            "window": Path(cls_storage.window_path[i]).name,
        }
        for i in get_anomalies(cls_storage)
    ]
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    logger.info("%d anomalies written to %s", len(rows), path)

