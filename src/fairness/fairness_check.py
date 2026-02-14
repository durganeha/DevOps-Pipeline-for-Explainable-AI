import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def run_audit(X_test, y_pred, y_true):
    """
    Standardized Fairness Audit using the AIF360 Toolkit.
    
    Args:
        X_test (pd.DataFrame): The feature set (must include 'race')
        y_pred (np.array): Model predictions
        y_true (np.array): Actual ground truth labels
    """
    
    # 1. Prepare the DataFrame for AIF360
    df = X_test.copy()
    
    # Handle the list requirement for label_names in newer AIF360 versions
    target_name = 'target'
    df[target_name] = y_true
    
    # 2. Create the BinaryLabelDataset object
    # Favorable = 1 (e.g., Grant Bail), Unfavorable = 0 (e.g., Deny Bail)
    ds_true = BinaryLabelDataset(
        df=df, 
        label_names=[target_name], 
        favorable_label=1, 
        unfavorable_label=0,
        protected_attribute_names=['race']
    )
    
    # 3. Create a copy for the predictions
    ds_pred = ds_true.copy()
    ds_pred.labels = y_pred.reshape(-1, 1)
    
    # 4. Define demographic groups
    # 0 = Caucasian (Privileged), 1 = African-American (Unprivileged)
    privileged_groups = [{'race': 0}]
    unprivileged_groups = [{'race': 1}]
    
    # 5. Initialize the Metric Calculator
    metric = ClassificationMetric(
        ds_true, 
        ds_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # 6. Calculate and return standard fairness metrics
    results = {
        "DI": metric.disparate_impact(),
        "EOD": metric.equal_opportunity_difference(),
        "SPD": metric.statistical_parity_difference()
    }
    
    return results