from graphwatch_console.api import format_detail
from graphwatch_console.page_jobs import window_label


def test_format_detail_lists_invalid_fields():
    detail = [
        {"loc": ["body", "inference_start"], "msg": "Value error, expected format"},
        {"loc": ["body"], "msg": "inference_end must be after inference_start"},
    ]
    assert format_detail(detail) == (
        "inference_start: Value error, expected format; inference_end must be after inference_start"
    )
    assert format_detail("Job 'x' not found") == "Job 'x' not found"


def test_window_label_from_graph_file_name():
    path = "/detection_service/data/models/m/extended_graphs/processed/" \
        "cadets_e3_test_graph_1522987200000000000_to_1522994400000000000_extended.pt"
    assert window_label(path) == "2018-04-06 04:00 → 06:00"
    assert window_label("other.pt") == "other.pt"
