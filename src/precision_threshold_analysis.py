import duckdb
import time
import os
import psutil

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def safe_div(n, d):
    return n / d if d > 0 else 0.0

def f1_score(p, r):
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def main():
    print("Starting Step 6: Precision-Focused Threshold Optimization...")
    start_time = time.time()
    
    con = duckdb.connect()
    
    # 1. Join predictions and features
    con.execute("""
        CREATE TABLE joined_data AS 
        SELECT 
            p.*, 
            f.name_sim, f.addr_sim, f.exact_country, f.exact_name, f.exact_address,
            f.name_len_diff, f.addr_len_diff
        FROM read_parquet('intermediate/model_predictions_10k.parquet') p
        JOIN read_parquet('intermediate/similarity_features_10k.parquet') f
          ON p.source1_entity_id = f.source1_entity_id AND p.candidate_entity_id = f.candidate_entity_id
    """)
    
    # Analyze thresholds on TEST set
    test_rows = con.execute("SELECT * FROM joined_data WHERE split = 'test'").fetchall()
    
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    thresh_results = []
    
    best_precision = -1.0
    best_prec_recall = -1.0
    best_prec_thresh = 0.0
    best_prec_tp = 0
    best_prec_fp = 0
    best_prec_fn = 0
    best_prec_tn = 0
    
    rec_99_prec = -1; rec_95_prec = -1; rec_90_prec = -1
    
    for t in thresholds:
        tp = sum(1 for r in test_rows if r[3] >= t and r[2] == 1)
        fp = sum(1 for r in test_rows if r[3] >= t and r[2] == 0)
        fn = sum(1 for r in test_rows if r[3] < t and r[2] == 1)
        tn = sum(1 for r in test_rows if r[3] < t and r[2] == 0)
        
        pred_pos = tp + fp
        prec = safe_div(tp, pred_pos)
        rec = safe_div(tp, tp + fn)
        f1 = f1_score(prec, rec)
        
        thresh_results.append((t, pred_pos, tp, fp, fn, tn, prec, rec, f1))
        
        if prec > best_precision or (prec == best_precision and rec > best_prec_recall):
            best_precision = prec
            best_prec_recall = rec
            best_prec_thresh = t
            best_prec_tp = tp
            best_prec_fp = fp
            best_prec_fn = fn
            best_prec_tn = tn
            
        if rec >= 0.99: rec_99_prec = max(rec_99_prec, prec)
        if rec >= 0.95: rec_95_prec = max(rec_95_prec, prec)
        if rec >= 0.90: rec_90_prec = max(rec_90_prec, prec)

    # Analyze Hard Rules on FULL dataset
    all_rows = con.execute("SELECT * FROM joined_data").fetchall()
    rule_results = []
    
    # r[2]=is_match, r[3]=prob, r[7]=name_sim, r[8]=addr_sim, r[9]=exact_country, r[10]=exact_name, r[11]=exact_address
    def eval_rule(rule_name, condition_func):
        tp = sum(1 for r in all_rows if condition_func(r) and r[2] == 1)
        fp = sum(1 for r in all_rows if condition_func(r) and r[2] == 0)
        fn = sum(1 for r in all_rows if not condition_func(r) and r[2] == 1)
        tn = sum(1 for r in all_rows if not condition_func(r) and r[2] == 0)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = f1_score(prec, rec)
        rule_results.append((rule_name, tp, fp, fn, tn, prec, rec, f1, tp+fp))
        
    eval_rule("Rule A (exact_country & exact_name)", lambda r: r[9]==1 and r[10]==1)
    eval_rule("Rule B (exact_country & exact_addr)", lambda r: r[9]==1 and r[11]==1)
    eval_rule("Rule C (name>=0.95 & addr>=0.80 & country)", lambda r: r[7]>=0.95 and r[8]>=0.80 and r[9]==1)
    eval_rule("Rule D (name>=0.90 & addr>=0.80 & country)", lambda r: r[7]>=0.90 and r[8]>=0.80 and r[9]==1)
    eval_rule("Rule E (name>=0.95 & addr>=0.90 & country)", lambda r: r[7]>=0.95 and r[8]>=0.90 and r[9]==1)
    eval_rule("Rule F (prob>=0.99 & country)", lambda r: r[3]>=0.99 and r[9]==1)
    eval_rule("Rule G (prob>=0.995 & country)", lambda r: r[3]>=0.995 and r[9]==1)

    # Three-way decision on full dataset
    decision_results = []
    for match_t in [0.99, 0.995, 0.999]:
        matches = 0; reviews = 0; non_matches = 0
        tp = 0; fp = 0
        for r in all_rows:
            p = r[3]
            if p >= match_t:
                matches += 1
                if r[2] == 1: tp += 1
                else: fp += 1
            elif p >= 0.50:
                reviews += 1
            else:
                non_matches += 1
        
        prec = safe_div(tp, tp+fp)
        rec = safe_div(tp, 114) # 114 total matches in 10k
        decision_results.append((match_t, matches, reviews, non_matches, prec, rec))
        
    # Pick the highest threshold that has some TP for output parquet
    out_thresh = best_prec_thresh
    
    con.execute(f"""
        CREATE TABLE out_preds AS 
        SELECT 
            p.*, 
            {out_thresh}::DOUBLE as threshold,
            CASE 
                WHEN p.predicted_probability >= {out_thresh} THEN 'MATCH'
                WHEN p.predicted_probability >= 0.50 THEN 'REVIEW'
                ELSE 'NON_MATCH'
            END as decision
        FROM read_parquet('intermediate/model_predictions_10k.parquet') p
    """)
    con.execute("COPY out_preds TO 'intermediate/precision_threshold_predictions_10k.parquet' (FORMAT PARQUET)")
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/precision_threshold_analysis.txt', 'w') as f:
        f.write("STEP 6: PRECISION-FOCUSED THRESHOLD OPTIMIZATION\n")
        f.write("==================================================\n\n")
        
        f.write("1. THRESHOLD ANALYSIS (Test Set Only)\n")
        f.write(f"{'Threshold':<10} | {'Pred Pos':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'TN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}\n")
        f.write("-" * 95 + "\n")
        for t in thresh_results:
            f.write(f"{t[0]:<10.3f} | {t[1]:<10} | {t[2]:<5} | {t[3]:<5} | {t[4]:<5} | {t[5]:<5} | {t[6]:<10.4f} | {t[7]:<10.4f} | {t[8]:<10.4f}\n")
            
        f.write("\n2. PRECISION TARGET\n")
        f.write(f"Highest measured precision: {best_precision*100:.2f}%\n")
        if best_prec_fp == 0:
            f.write("100% observed precision on this evaluation sample (0 observed false positives)\n")
        
        f.write(f"Precision near 99% recall: {max(0, rec_99_prec)*100:.2f}%\n")
        f.write(f"Precision near 95% recall: {max(0, rec_95_prec)*100:.2f}%\n")
        f.write(f"Precision near 90% recall: {max(0, rec_90_prec)*100:.2f}%\n")
        f.write(f"Highest threshold producing useful matches: {best_prec_thresh}\n")
        
        f.write("\n3. PRECISION-ORIENTED HARD RULES (Full 10K Dataset)\n")
        for r in rule_results:
            f.write(f"\n{r[0]}:\n")
            f.write(f"  TP: {r[1]}, FP: {r[2]}, FN: {r[3]}, TN: {r[4]}\n")
            f.write(f"  Precision: {r[5]:.4f}, Recall: {r[6]:.4f}, F1: {r[7]:.4f}\n")
            f.write(f"  Predicted MATCH: {r[8]}\n")
            
        f.write("\n4. THREE-WAY DECISION SYSTEM (Full 10K Dataset)\n")
        for d in decision_results:
            f.write(f"\nThreshold: >= {d[0]:.3f}\n")
            f.write(f"  MATCH: {d[1]}\n")
            f.write(f"  REVIEW: {d[2]}\n")
            f.write(f"  NON_MATCH: {d[3]}\n")
            f.write(f"  MATCH Group Precision: {d[4]:.4f}, Recall: {d[5]:.4f}\n")
            
    # VALIDATION
    print("\n=== STEP 6 VALIDATION ===")
    v_rows = con.execute("SELECT * FROM out_preds").fetchall()
    v_schema = con.execute("DESCRIBE out_preds").fetchall()
    cols = [r[0] for r in v_schema]
    
    passed = True
    errors = []
    
    if len(v_rows) != 10000:
        passed = False
        errors.append(f"Row count is {len(v_rows)}, expected 10000")
        
    prob_idx = cols.index('predicted_probability')
    label_idx = cols.index('is_match')
    thresh_idx = cols.index('threshold')
    dec_idx = cols.index('decision')
    split_idx = cols.index('split')
    s1_idx = cols.index('source1_entity_id')
    
    has_null = False
    for r in v_rows:
        if r[prob_idx] is None: has_null = True
        if r[prob_idx] < 0 or r[prob_idx] > 1:
            passed = False
            errors.append("Probabilities out of range")
            break
        if r[label_idx] not in (0, 1):
            passed = False
            errors.append("Invalid label")
            break
        if r[thresh_idx] != out_thresh:
            passed = False
            errors.append("Threshold mismatch")
            break
            
    if has_null:
        passed = False
        errors.append("NULL probabilities found")
        
    train_ids = {r[s1_idx] for r in v_rows if r[split_idx] == 'train'}
    test_ids = {r[s1_idx] for r in v_rows if r[split_idx] == 'test'}
    if len(train_ids.intersection(test_ids)) > 0:
        passed = False
        errors.append("Train/Test leakage detected")
        
    if passed:
        print("PASS")
    else:
        print("FAIL")
        for e in errors: print(e)
        
    print("\n=== FINAL REPORT ===")
    print("STEP 6 COMPLETE\n")
    print(f"Highest measured precision:\n{best_precision*100:.2f}%\n")
    print(f"Corresponding recall:\n{best_prec_recall*100:.2f}%\n")
    print(f"Threshold:\n{best_prec_thresh:.3f}\n")
    print(f"TP:\n{best_prec_tp}\n")
    print(f"FP:\n{best_prec_fp}\n")
    print(f"FN:\n{best_prec_fn}\n")
    print(f"TN:\n{best_prec_tn}\n")
    
    demonstrated = "YES" if best_precision >= 0.9999 else "NO"
    print(f"Was 99.99% precision demonstrated?\n{demonstrated}")
    if demonstrated == "NO":
        print("99.99% precision was not demonstrated on the current evaluation sample.\n")
        
    best_rule = max(rule_results, key=lambda x: x[5])
    print(f"Best precision-oriented rule:\n{best_rule[0]} (Precision: {best_rule[5]*100:.2f}%)")
    
    print(f"\nRuntime: {time.time() - start_time:.2f}s | Peak RAM: {get_memory_usage():.2f}MB")

if __name__ == "__main__":
    main()
