import duckdb
import os
import time
import psutil

def main():
    print("Starting Refined Blocking Evaluation...")
    start_time = time.time()
    con = duckdb.connect(database=':memory:')

    con.execute("PRAGMA memory_limit='10GB'")

    # 1. Prepare ground truth for the 10,000 S1 records
    con.execute("""
        CREATE TABLE s1_sample AS 
        SELECT entity_id as source1_entity_id FROM read_parquet('intermediate/source1_normalized.parquet')
        ORDER BY md5(entity_id) LIMIT 10000
    """)
    
    con.execute("""
        CREATE TABLE gt AS 
        SELECT 
            gt.source1_entity_id,
            UNNEST(string_split(gt.matched_entity_ids, ',')) as expected_match_id
        FROM read_csv('train_ground_truth.tsv', sep='\\t', header=True) gt
        JOIN s1_sample s1 ON gt.source1_entity_id = s1.source1_entity_id
    """)

    total_true = con.execute("SELECT COUNT(*) FROM gt").fetchone()[0]
    total_true_s2 = con.execute("SELECT COUNT(*) FROM gt WHERE expected_match_id LIKE 'S2-%'").fetchone()[0]
    total_true_s3 = con.execute("SELECT COUNT(*) FROM gt WHERE expected_match_id LIKE 'S3-%'").fetchone()[0]

    # 2. Load generated candidates
    con.execute("CREATE TABLE cands AS SELECT * FROM read_parquet('intermediate/blocking_refined_candidates.parquet')")

    # 3. View for strategy-level analysis
    con.execute("""
        CREATE VIEW cands_unnested AS
        SELECT 
            source1_entity_id,
            candidate_entity_id,
            candidate_source,
            UNNEST(string_split(blocking_strategies, ',')) as strategy
        FROM cands
    """)

    strategies = [s[0] for s in con.execute("SELECT DISTINCT strategy FROM cands_unnested").fetchall()]

    os.makedirs('reports', exist_ok=True)
    with open('reports/blocking_refinement_evaluation.txt', 'w', encoding='utf-8') as f:
        f.write("BLOCKING REFINEMENT EVALUATION REPORT\n")
        f.write("=====================================\n")
        f.write("Dataset/subset used: 10,000 random S1 records (SAME AS BASELINE)\n")
        f.write(f"Total Expected Matches in sample: {total_true}\n\n")

        f.write("--- Individual Strategy Performance ---\n")
        for strat in strategies:
            found = con.execute(f"""
                SELECT COUNT(*) FROM gt
                JOIN cands_unnested c ON gt.source1_entity_id = c.source1_entity_id 
                                     AND gt.expected_match_id = c.candidate_entity_id
                WHERE c.strategy = '{strat}'
            """).fetchone()[0]
            recall = found / total_true if total_true > 0 else 0

            stats = con.execute(f"""
                SELECT 
                    AVG(cnt), 
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cnt),
                    MAX(cnt)
                FROM (
                    SELECT s1.source1_entity_id, COUNT(c.candidate_entity_id) as cnt
                    FROM s1_sample s1
                    LEFT JOIN (SELECT * FROM cands_unnested WHERE strategy = '{strat}') c 
                    ON s1.source1_entity_id = c.source1_entity_id
                    GROUP BY s1.source1_entity_id
                )
            """).fetchone()

            sum_cands = con.execute(f"SELECT COUNT(*) FROM cands_unnested WHERE strategy = '{strat}'").fetchone()[0]
            zero_cands = con.execute(f"""
                SELECT COUNT(*) FROM s1_sample 
                WHERE source1_entity_id NOT IN (
                    SELECT source1_entity_id FROM cands_unnested WHERE strategy = '{strat}'
                )
            """).fetchone()[0]

            f.write(f"Strategy: {strat}\n")
            f.write(f"  Candidate Recall: {recall:.4%} ({found}/{total_true})\n")
            f.write(f"  Total Candidates: {sum_cands:,}\n")
            f.write(f"  Avg per S1:       {stats[0]:.2f}\n")
            f.write(f"  Median per S1:    {stats[1]:.1f}\n")
            f.write(f"  Max for one S1:   {stats[2]:,}\n")
            f.write(f"  S1s with 0 cands: {zero_cands:,}\n\n")

        f.write("--- UNION of All Strategies ---\n")
        union_found = con.execute("""
            SELECT COUNT(*) FROM gt
            JOIN cands c ON gt.source1_entity_id = c.source1_entity_id 
                        AND gt.expected_match_id = c.candidate_entity_id
        """).fetchone()[0]
        union_found_s2 = con.execute("""
            SELECT COUNT(*) FROM gt
            JOIN cands c ON gt.source1_entity_id = c.source1_entity_id 
                        AND gt.expected_match_id = c.candidate_entity_id
            WHERE gt.expected_match_id LIKE 'S2-%'
        """).fetchone()[0]
        union_found_s3 = con.execute("""
            SELECT COUNT(*) FROM gt
            JOIN cands c ON gt.source1_entity_id = c.source1_entity_id 
                        AND gt.expected_match_id = c.candidate_entity_id
            WHERE gt.expected_match_id LIKE 'S3-%'
        """).fetchone()[0]

        union_recall = union_found / total_true if total_true > 0 else 0
        s2_recall = union_found_s2 / total_true_s2 if total_true_s2 > 0 else 0
        s3_recall = union_found_s3 / total_true_s3 if total_true_s3 > 0 else 0

        union_stats = con.execute("""
            SELECT 
                AVG(cnt), 
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cnt),
                MAX(cnt)
            FROM (
                SELECT s1.source1_entity_id, COUNT(c.candidate_entity_id) as cnt
                FROM s1_sample s1
                LEFT JOIN cands c ON s1.source1_entity_id = c.source1_entity_id
                GROUP BY s1.source1_entity_id
            )
        """).fetchone()

        union_sum = con.execute("SELECT COUNT(*) FROM cands").fetchone()[0]
        union_zero = con.execute("""
            SELECT COUNT(*) FROM s1_sample 
            WHERE source1_entity_id NOT IN (
                SELECT source1_entity_id FROM cands
            )
        """).fetchone()[0]

        f.write(f"Union Candidate Recall: {union_recall:.4%} ({union_found}/{total_true})\n")
        f.write(f"  S1->S2 Recall: {s2_recall:.4%} ({union_found_s2}/{total_true_s2})\n")
        f.write(f"  S1->S3 Recall: {s3_recall:.4%} ({union_found_s3}/{total_true_s3})\n")
        f.write(f"Union Total Candidates: {union_sum:,}\n")
        f.write(f"Avg per S1:       {union_stats[0]:.2f}\n")
        f.write(f"Median per S1:    {union_stats[1]:.1f}\n")
        f.write(f"Max for one S1:   {union_stats[2]:,}\n")
        f.write(f"S1s with 0 cands: {union_zero:,}\n\n")

        f.write("--- Analysis & Recommendations ---\n")
        f.write("Baseline Comparison:\n")
        f.write("- Baseline Recall: 76.35%\n")
        f.write(f"- Refined Recall:  {union_recall:.4%}\n")
        f.write("- Baseline Avg Cands: 4,493.91\n")
        f.write(f"- Refined Avg Cands:  {union_stats[0]:.2f}\n")
        
        f.write(f"\nRuntime: {time.time() - start_time:.2f} seconds\n")
        f.write(f"Peak RAM: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB\n")

    print(f"Evaluation Complete. Check reports/blocking_refinement_evaluation.txt. Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
