import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from posture_monitor.calibration.calibrate import NORMALIZED_FEATURE_ORDER, feature_vector, normalize_features


def _to_matrix(sub_df, scale_reference):
    X = [feature_vector(normalize_features(row, scale_reference)) for row in sub_df.to_dict("records")]
    y = sub_df["class"].tolist()
    return X, y


def load_and_normalize(csv_path):
    df = pd.read_csv(csv_path)
    train_df = df[df.honest_split == "train"]
    test_df = df[df.honest_split == "test"]
    scale_reference = train_df["torso_length_px"].median()

    X_train, y_train = _to_matrix(train_df, scale_reference)
    X_test, y_test = _to_matrix(test_df, scale_reference)
    return X_train, y_train, X_test, y_test, scale_reference


def train_and_evaluate(csv_path, model_out_path):
    X_train, y_train, X_test, y_test, scale_reference = load_and_normalize(csv_path)

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    }

    best_name, best_model, best_score = None, None, -1.0
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
        mean_score = scores.mean()
        print(f"{name}: cv f1_macro = {mean_score:.3f}")
        if mean_score > best_score:
            best_name, best_model, best_score = name, model, mean_score

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"Selected model: {best_name}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=sorted(set(y_train))))

    joblib.dump(
        {"model": best_model, "scale_reference": scale_reference, "feature_order": NORMALIZED_FEATURE_ORDER},
        model_out_path,
    )
    return best_model


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "posture_monitor/training/features_honest.csv"
    model_out_path = sys.argv[2] if len(sys.argv) > 2 else "posture_monitor/models/classifier.joblib"
    train_and_evaluate(csv_path, model_out_path)
