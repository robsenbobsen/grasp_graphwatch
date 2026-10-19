import argparse
import importlib
import json
import logging
import os
from pathlib import Path

import torch
from grasp import config
from grasp.detection import (
    classification,
    classification_pipeline,
    classification_storage,
    clustering,
)
from grasp.evaluation import evaluation_storage, ground_truth_storage
from grasp.graph import graph_management, graph_storage
from grasp.reporting import detailed_report, report, visualization
from grasp.utils import detection_helpers
from grasp.utils.evaluation_helpers import compute_metrics_multiclass
from grasp.utils.file_helpers import (
    generate_experiment_file_prefix,
    generate_storage_paths,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRASP experiment")
    parser.add_argument(
        "--experiment-config",
        "-e",
        type=Path,
        required=True,
        help="Path to experiment YAML file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = args.experiment_config.expanduser().resolve()

    if not experiment_path.exists():
        logger.warning(
            "Experiment config not found at %s; falling back to defaults",
            experiment_path,
        )

    os.environ["GRASP_EXPERIMENT_CONFIG"] = str(experiment_path)
    # Reload module + config classes so CLI-provided path is fully honored
    importlib.reload(config)
    config.reload_experiment_config(experiment_path)
    config.EXPERIMENT_PREFIX = generate_experiment_file_prefix(
        base_name=config.EXPERIMENT_PREFIX,
        params={
            "dataset": config.DATASET_NAME,
            "context_size": str(config.GraphConfig.CONTEXT_SIZE),
            "step_size": str(config.GraphConfig.STEP_SIZE),
        },
    )

    logger.info("Using experiment config at %s", config.EXPERIMENT_CONFIG_PATH)
    logger.info(
        "Dataset: %s | Experiment prefix: %s",
        config.DATASET_NAME,
        config.EXPERIMENT_PREFIX,
    )

    base_path = Path(__file__).resolve().parent / "data" / config.DATASET_NAME
    storage_paths = generate_storage_paths(config.EXPERIMENT_PREFIX, base_dir=base_path)

    gm = graph_management.GraphManager(
        dataset_name=config.DATASET_NAME,
        experiment_prefix=config.EXPERIMENT_PREFIX,
        train_start_times=config.GraphConfig.TRAIN_START_TIMES,
        train_end_times=config.GraphConfig.TRAIN_END_TIMES,
        test_start_times=config.GraphConfig.TEST_START_TIMES,
        test_end_times=config.GraphConfig.TEST_END_TIMES,
        context_size=config.GraphConfig.CONTEXT_SIZE,
        step_size=config.GraphConfig.STEP_SIZE,
        force_reload=config.GraphConfig.FORCE_RELOAD,
        root_dir=config.GraphConfig.GRAPH_ROOT_DIR,
        autoencoder_embedding_dim=(config.LocationTransformerConfig.EMBEDDING_DIM),
        autoencoder_hidden_dim=config.LocationTransformerConfig.HIDDEN_DIM,
        autoencoder_num_epochs=config.LocationTransformerConfig.NUM_EPOCHS,
        autoencoder_patience=config.LocationTransformerConfig.PATIENCE,
        autoencoder_batch_size=config.LocationTransformerConfig.BATCH_SIZE,
        autoencoder_num_layers=config.LocationTransformerConfig.NUM_LAYERS,
        autoencoder_nhead=config.LocationTransformerConfig.NHEAD,
        location_embedding_model_type=(config.LocationTransformerConfig.MODEL_TYPE),
        word2vec_window_size=(config.LocationTransformerConfig.WORD2VEC_WINDOW_SIZE),
        word2vec_min_count=config.LocationTransformerConfig.WORD2VEC_MIN_COUNT,
        word2vec_negative_samples=(config.LocationTransformerConfig.WORD2VEC_NEGATIVE_SAMPLES),
        word2vec_learning_rate=(config.LocationTransformerConfig.WORD2VEC_LEARNING_RATE),
        autoencoder_device=config.LocationTransformerConfig.DEVICE,
    )
    gm.run_full_workflow()
    gm.save_graph_storage(str(storage_paths["graph_storage"]))

    gs: graph_storage.GraphStorage = gm.graph_storage

    trained_model: classification.GAT = classification_pipeline.window_based_train(
        graph_storage=gs,
        batch_size=config.DetectionConfig.BATCH_SIZE,
        num_neighbors=config.DetectionConfig.NUM_NEIGHBORS,
        epochs=config.DetectionConfig.EPOCHS,
        hidden_channels=config.DetectionConfig.HIDDEN_DIM,
        num_layers=config.DetectionConfig.NUM_LAYERS,
        heads=config.DetectionConfig.HEADS,
        dropout=config.DetectionConfig.DROPOUT,
        device=torch.device(config.DetectionConfig.DEVICE),
        learning_rate=config.DetectionConfig.LEARNING_RATE,
        weight_decay=config.DetectionConfig.WEIGHT_DECAY,
        shuffle_train_paths=config.DetectionConfig.SHUFFLE_TRAIN_PATHS,
        shuffle_train_batches=config.DetectionConfig.SHUFFLE_TRAIN_BATCHES,
        use_class_weights=config.DetectionConfig.USE_CLASS_WEIGHTS,
        enable_subset_finetune=(config.DetectionConfig.SUBSET_FINETUNE_ENABLED),
        subset_finetune_epochs=(config.DetectionConfig.SUBSET_FINETUNE_EPOCHS),
        subset_finetune_learning_rate=(config.DetectionConfig.SUBSET_FINETUNE_LEARNING_RATE),
        subset_finetune_lr_factor=(config.DetectionConfig.SUBSET_FINETUNE_LR_FACTOR),
        subset_finetune_freeze_backbone=(config.DetectionConfig.SUBSET_FINETUNE_FREEZE_BACKBONE),
    )
    trained_model.save_model(str(storage_paths["classification_model"]))

    logger.info(trained_model)  # type: ignore
    total_params = sum(p.numel() for p in trained_model.parameters())
    trainable_params = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    cls_storage_train: classification_storage.ClassificationStorage = (
        classification_pipeline.window_based_test(
            model=trained_model,
            graph_storage=gs,
            batch_size=config.DetectionConfig.BATCH_SIZE,
            num_neighbors=config.DetectionConfig.NUM_NEIGHBORS,
            device=torch.device(config.DetectionConfig.DEVICE),
            for_train_data=True,
        )
    )
    cls_storage_train.save_to_file(str(storage_paths["classification_train"]))
    cls_storage_test: classification_storage.ClassificationStorage = (
        classification_pipeline.window_based_test(
            model=trained_model,
            graph_storage=gs,
            batch_size=config.DetectionConfig.BATCH_SIZE,
            num_neighbors=config.DetectionConfig.NUM_NEIGHBORS,
            device=torch.device(config.DetectionConfig.DEVICE),
            for_train_data=False,
        )
    )
    cls_storage_test.save_to_file(str(storage_paths["classification_test"]))

    y_true = cls_storage_train.y
    y_pred = cls_storage_train.y_hat

    train_metrics = compute_metrics_multiclass(
        true_labels=y_true,
        pred_labels=y_pred,
    )
    logger.info("Training Metrics: %s", train_metrics)
    # Save train_metrics to a file
    metrics_file = (
        Path(storage_paths["classification_train"]).parent
        / f"{config.EXPERIMENT_PREFIX}_cls_storage_metrics.json"
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    logger.info("Training metrics saved to %s", metrics_file)

    cluster_manager = clustering.ClusterManager(
        classification_storage=cls_storage_train,  # Important to cluster train cls storage
    )
    cluster_manager.create_misclassification_clusters()

    detection_helpers.check_classification_with_clusters(
        cluster_manager=cluster_manager,
        cls_storage=cls_storage_test,
    )
    cluster_manager.save_to_disk(filepath=(storage_paths["cluster_manager"]))  # type: ignore

    cls_storage_test.save_to_file(str(storage_paths["classification_test"]))

    # temporary: save classification_predictions to a file for analysis
    cls_predictions_file = (
        Path(storage_paths["classification_test"]).parent
        / f"{config.EXPERIMENT_PREFIX}_classification_predictions.json"
    )
    cls_predictions_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cls_predictions_file, "w", encoding="utf-8") as f:
        json.dump(cls_storage_test.y_hat_proba, f, indent=2)

    es = evaluation_storage.EvaluationStorage(
        cs=cls_storage_test,
        train_subject_cmd_to_id=gs.train_subject_cmd_to_id,
    )
    gts = ground_truth_storage.GroundTruthStorage(
        base_path=config.GroundTruthConfig.GROUND_TRUTH_BASE_DIR
    )

    es.get_y_and_y_hat_per_gt_group(gts=gts, cs=cls_storage_test)
    es.get_y_and_y_hat_per_gt_file(gts=gts, cs=cls_storage_test)
    es.save_to_file(str(storage_paths["evaluation_storage"]))
    gts.save_to_file(str(storage_paths["ground_truth_storage"]))
    r = report.Report(
        classification_storage=cls_storage_test,
        evaluation_storage=es,
        ground_truth_storage=gts,
        graph_storage=gs,
    )
    r.save(str(storage_paths["report"]))

    dr = detailed_report.DetailedReport(
        classification_storage=cls_storage_test,
        evaluation_storage=es,
        ground_truth_storage=gts,
        graph_storage=gs,
    )
    dr.save(storage_paths["detailed_report"])  # type: ignore
    print(r.to_dict())
    visualization.visualize_misclassification_timeframes(
        detailed_report_path=storage_paths["detailed_report"],  # type: ignore
        output_path=storage_paths["visualizations_dir"]
        / f"{config.EXPERIMENT_PREFIX}_misclassifications_over_time.png",
        attack_windows=config.ATTACK_WINDOWS_FOR_VISUALIZATION,
        min_duration_seconds=0,
        to_display=False,
        x_tick_count=20,
    )


if __name__ == "__main__":
    main()
