from scripts.evaluate_yolo import match_detections, summarize_classes


def test_matching_counts_true_false_positives_and_false_negatives_by_class():
    ground_truth = [
        {"class_name": "book", "bbox": [0, 0, 10, 10]},
        {"class_name": "book", "bbox": [20, 20, 30, 30]},
        {"class_name": "laptop", "bbox": [40, 40, 50, 50]},
    ]
    predictions = [
        {"class_name": "book", "bbox": [0, 0, 10, 10]},
        {"class_name": "book", "bbox": [0, 0, 10, 10]},
        {"class_name": "book", "bbox": [40, 40, 50, 50]},
    ]

    counts = match_detections(predictions, ground_truth)

    assert counts["book"] == {"tp": 1, "fp": 2, "fn": 1}
    assert counts["laptop"] == {"tp": 0, "fp": 0, "fn": 1}


def test_per_class_metrics_distinguish_zero_from_undefined():
    metrics = summarize_classes({
        "book": {"tp": 1, "fp": 1, "fn": 3},
        "laptop": {"tp": 0, "fp": 0, "fn": 1},
        "mouse": {"tp": 0, "fp": 1, "fn": 0},
    })

    assert metrics["book"]["precision"] == 0.5
    assert metrics["book"]["recall"] == 0.25
    assert metrics["laptop"]["precision"] is None
    assert metrics["laptop"]["recall"] == 0
    assert metrics["mouse"]["precision"] == 0
    assert metrics["mouse"]["recall"] is None
