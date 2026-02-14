import sys
import joblib
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


# --------------------------------------
# FAIRNESS AUDIT FUNCTION
# --------------------------------------
def run_audit(X_test, y_pred, y_true):

    df = X_test.copy()
    target_name = "target"
    df[target_name] = y_true

    ds_true = BinaryLabelDataset(
        df=df,
        label_names=[target_name],
        favorable_label=1,
        unfavorable_label=0,
        protected_attribute_names=["race"]
    )

    ds_pred = ds_true.copy()
    ds_pred.labels = y_pred.reshape(-1, 1)

    privileged_groups = [{"race": 0}]
    unprivileged_groups = [{"race": 1}]

    metric = ClassificationMetric(
        ds_true,
        ds_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    results = {
        "DI": metric.disparate_impact(),
        "EOD": metric.equal_opportunity_difference(),
        "SPD": metric.statistical_parity_difference()
    }

    return results


# --------------------------------------
# CI/CD ENTRY POINT
# --------------------------------------
if __name__ == "__main__":

    print("Running Fairness Audit...")

    # Load saved model artifacts
    artifacts = joblib.load("artifacts/model.pkl")

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Generate predictions
    y_pred = model.predict(X_test_scaled)

    # Run fairness audit
    results = run_audit(X_test, y_pred, y_test)

    print("Fairness Results:")
    print(results)

    # --------------------------------------
    # FAIRNESS GATE (CI/CD CONTROL)
    # --------------------------------------
    FAIRNESS_THRESHOLD = 0.8

    if results["DI"] < FAIRNESS_THRESHOLD:
        print("❌ Fairness threshold violated! CI should fail.")
        sys.exit(1)
    else:
        print("✅ Fairness criteria satisfied.")
