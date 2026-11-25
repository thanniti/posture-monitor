# scripts/export_model.py
import json
import sys

import joblib


def _export_tree(tree):
    return {
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "childrenLeft": tree.children_left.tolist(),
        "childrenRight": tree.children_right.tolist(),
        "value": [node_value[0].tolist() for node_value in tree.value],
    }


def export_model(model_bundle_path, output_path):
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    if type(model).__name__ != "RandomForestClassifier":
        raise ValueError(
            f"export_model only supports RandomForestClassifier, got {type(model).__name__}. "
            "If train.py selected a different model type, this export script needs updating "
            "to match (e.g. a different tree-walk, or an SVM-specific export)."
        )
    payload = {
        "classes": list(model.classes_),
        "featureOrder": bundle["feature_order"],
        "scaleReference": bundle["scale_reference"],
        "trees": [_export_tree(estimator.tree_) for estimator in model.estimators_],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"Exported {len(payload['trees'])} trees, {len(payload['classes'])} classes to {output_path}")


if __name__ == "__main__":
    model_bundle_path = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/models/classifier.joblib"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "web/public/models/classifier.json"
    export_model(model_bundle_path, output_path)
