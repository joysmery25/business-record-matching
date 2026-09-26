import pandas as pd
import numpy as np
import time
import os
import psutil
import duckdb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix)

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def evaluate_model(y_true, y_pred, y_prob, name):
    res = {}
    res['name'] = name
    res['precision'] = precision_score(y_true, y_pred, zero_division=0)
    res['recall'] = recall_score(y_true, y_pred, zero_division=0)
    res['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Try ROC AUC
    try:
        res['roc_auc'] = roc_auc_score(y_true, y_prob)
    except Exception:
        res['roc_auc'] = np.nan
        
    res['pr_auc'] = average_precision_score(y_true, y_prob)
    res['cm'] = confusion_matrix(y_true, y_pred)
    return res

def main():
    print("Starting Step 5: Match Model Evaluation...")
    start_time = time.time()
    
    con = duckdb.connect()
    df = con.execute("SELECT * FROM read_parquet('intermediate/similarity_features_10k.parquet')").df()
    
    # Check nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("WARNING: Null values found in features")
    
    features = [
        'name_sim', 'addr_sim', 'exact_country', 
        'exact_name', 'exact_address', 'name_len_diff', 'addr_len_diff'
    ]
    target = 'is_match'
    group_col = 'source1_entity_id'
    
    X = df[features]
    y = df[target]
    groups = df[group_col]
    
    # Check grouped distribution
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_test, y_test, groups_test = X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]
    
    # Check positive counts
    pos_train = y_train.sum()
    pos_test = y_test.sum()
    
    if pos_train < 10 or pos_test < 5:
        print(f"WARNING: Too few positives in split! Train pos: {pos_train}, Test pos: {pos_test}")
        # Proceeding anyway as instructed, but warning
        
    # Model A: Logistic Regression
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_prob_test = lr.predict_proba(X_test)[:, 1]
    lr_pred_test = lr.predict(X_test)
    lr_eval = evaluate_model(y_test, lr_pred_test, lr_prob_test, "Logistic Regression")
    
    # Model B: Random Forest
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    rf_prob_test = rf.predict_proba(X_test)[:, 1]
    rf_pred_test = rf.predict(X_test)
    rf_eval = evaluate_model(y_test, rf_pred_test, rf_prob_test, "Random Forest")
    
    # Baseline Non-ML Rules
    # We will tune thresholds on the train set.
    # Grid of (name_sim, addr_sim) thresholds
    best_f1 = 0
    best_rule = None
    
    for n_thresh in [0.7, 0.8, 0.9]:
        for a_thresh in [0.0, 0.5, 0.7]:
            # Rule: exact_name OR (name_sim >= n_thresh AND addr_sim >= a_thresh)
            rule_pred_train = (X_train['exact_name'] == 1) | ((X_train['name_sim'] >= n_thresh) & (X_train['addr_sim'] >= a_thresh))
            f1 = f1_score(y_train, rule_pred_train, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_rule = (n_thresh, a_thresh)
                
    if best_rule:
        n_thresh, a_thresh = best_rule
        baseline_prob_test = ((X_test['exact_name'] == 1) | ((X_test['name_sim'] >= n_thresh) & (X_test['addr_sim'] >= a_thresh))).astype(float)
        baseline_pred_test = baseline_prob_test.astype(int)
        base_eval = evaluate_model(y_test, baseline_pred_test, baseline_prob_test, f"Baseline (Name>={n_thresh} & Addr>={a_thresh})")
    else:
        # Fallback if no rule produces F1 > 0
        baseline_prob_test = np.zeros(len(y_test))
        baseline_pred_test = np.zeros(len(y_test))
        base_eval = evaluate_model(y_test, baseline_pred_test, baseline_prob_test, "Baseline (All Zero)")
        
    # Generate complete predictions with best model (Random Forest is usually best, but let's compare F1)
    # Actually, we will predict for all 10,000 using Random Forest, as it is generally stronger, but let's pick the best F1 model.
    best_model_name = "Random Forest"
    best_model = rf
    if lr_eval['f1'] > rf_eval['f1']:
        best_model_name = "Logistic Regression"
        best_model = lr
        
    df['predicted_probability'] = best_model.predict_proba(X)[:, 1]
    df['predicted_label'] = best_model.predict(X)
    df['model_name'] = best_model_name
    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'
    
    # Save predictions
    out_cols = ['source1_entity_id', 'candidate_entity_id', 'is_match', 'predicted_probability', 'predicted_label', 'model_name', 'split']
    final_df = df[out_cols]
    con.register('final_df', final_df)
    con.execute("COPY final_df TO 'intermediate/model_predictions_10k.parquet' (FORMAT PARQUET)")
    
    # Write Report
    os.makedirs('reports', exist_ok=True)
    with open('reports/match_model_evaluation.txt', 'w', encoding='utf-8') as f:
        f.write("MATCH MODEL EVALUATION (10K QUICK)\n")
        f.write("==================================\n\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Train rows: {len(X_train)} (Positives: {pos_train}, Negatives: {len(X_train)-pos_train})\n")
        f.write(f"Test rows:  {len(X_test)} (Positives: {pos_test}, Negatives: {len(X_test)-pos_test})\n\n")
        
        for eval_res in [lr_eval, rf_eval, base_eval]:
            f.write(f"--- {eval_res['name']} ---\n")
            f.write(f"Precision: {eval_res['precision']:.4f}\n")
            f.write(f"Recall:    {eval_res['recall']:.4f}\n")
            f.write(f"F1-Score:  {eval_res['f1']:.4f}\n")
            f.write(f"ROC-AUC:   {eval_res['roc_auc']:.4f}\n")
            f.write(f"PR-AUC:    {eval_res['pr_auc']:.4f}\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"TN: {eval_res['cm'][0][0]} | FP: {eval_res['cm'][0][1]}\n")
            f.write(f"FN: {eval_res['cm'][1][0]} | TP: {eval_res['cm'][1][1]}\n\n")
            
        f.write("--- Feature Importance ---\n")
        f.write("Logistic Regression Coefficients:\n")
        for feat, coef in zip(features, lr.coef_[0]):
            f.write(f"  {feat}: {coef:.4f}\n")
            
        f.write("\nRandom Forest Feature Importances:\n")
        for feat, imp in zip(features, rf.feature_importances_):
            f.write(f"  {feat}: {imp:.4f}\n")
            
        f.write(f"\nRuntime: {time.time() - start_time:.2f} seconds\n")
        f.write(f"Peak RAM: {get_memory_usage():.2f} MB\n")
        
    print("Evaluation finished and saved.")
    
    # ----------------------------------------------------
    # === STEP 5 VALIDATION ===
    # ----------------------------------------------------
    print("\n=== STEP 5 VALIDATION ===")
    
    val_df = con.execute("SELECT * FROM read_parquet('intermediate/model_predictions_10k.parquet')").df()
    passed = True
    errors = []
    
    # prediction row count = 10,000
    if len(val_df) != 10000:
        passed = False
        errors.append(f"Row count is {len(val_df)}, expected 10000")
        
    # no NULL probabilities
    if val_df['predicted_probability'].isnull().any():
        passed = False
        errors.append("NULL values found in predicted_probability")
        
    # probabilities are between 0 and 1
    if val_df['predicted_probability'].min() < 0 or val_df['predicted_probability'].max() > 1:
        passed = False
        errors.append("predicted_probability out of range [0, 1]")
        
    # predicted labels are valid
    if not set(val_df['predicted_label'].unique()).issubset({0, 1}):
        passed = False
        errors.append("Invalid values in predicted_label (should be 0 or 1)")
        
    # train/test source1 IDs do not overlap
    train_ids = set(val_df[val_df['split'] == 'train']['source1_entity_id'])
    test_ids = set(val_df[val_df['split'] == 'test']['source1_entity_id'])
    overlap = train_ids.intersection(test_ids)
    if len(overlap) > 0:
        passed = False
        errors.append(f"Train/Test source1_entity_id overlap detected: {len(overlap)} IDs")
        
    if passed:
        print("PASS")
    else:
        print("FAIL")
        for err in errors:
            print(f"- {err}")

if __name__ == "__main__":
    main()
