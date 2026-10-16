import json

from grasp import stages
from grasp_contextualization import shared


def test_train_graph_storage_reconstructs_older_bundles(tmp_path):
    bundle = stages.ModelBundle(tmp_path / "model")
    processed = bundle.extended_graphs / "processed"
    processed.mkdir(parents=True)
    for name in ("db_train_graph_2_to_3_extended.pt", "db_train_graph_1_to_2_extended.pt",
                 "db_test_graph_5_to_6_extended.pt", "pre_filter.pt"):  # fmt: skip
        (processed / name).touch()
    bundle.train_cmd_to_id.write_text(json.dumps({"/bin/a": 0, "/bin/b": 1}))

    gs = stages.train_graph_storage(bundle)

    assert [p.rsplit("/", 1)[1] for p in gs.extended_train_data_paths] == [
        "db_train_graph_1_to_2_extended.pt",
        "db_train_graph_2_to_3_extended.pt",
    ]
    assert gs.train_subject_cmd_to_id == {"/bin/a": 0, "/bin/b": 1}


def test_local_path_maps_container_paths_into_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "REPO_ROOT", tmp_path)
    target = tmp_path / "data" / "models" / "m" / "g.pt"
    target.parent.mkdir(parents=True)
    target.touch()

    assert shared.local_path("/detection_service/data/models/m/g.pt") == target
    assert shared.local_path("/elsewhere/data/models/m/missing.pt").name == "missing.pt"
    assert shared.local_path(target) == target
