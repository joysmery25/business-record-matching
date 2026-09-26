import duckdb
import time
import os
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def batch_cosine_sim(texts1, texts2, analyzer, ngram_range):
    # Handle NaNs / Nones
    texts1 = ["" if t is None else str(t) for t in texts1]
    texts2 = ["" if t is None else str(t) for t in texts2]
    
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    try:
        vec.fit(texts1 + texts2)
        X1 = vec.transform(texts1)
        X2 = vec.transform(texts2)
        sims = X1.multiply(X2).sum(axis=1).A1
    except ValueError:
        # Empty vocabulary
        sims = np.zeros(len(texts1))
    return sims

def main():
    print("Starting Similarity Feature Generation...")
    start_time = time.time()
    
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB'")
    
    # Load candidate pairs and sample 50,000 reproducibly
    con.execute("SELECT setseed(0.42)")
    con.execute("CREATE VIEW cands AS SELECT * FROM read_parquet('intermediate/blocking_final_candidates.parquet') ORDER BY random() LIMIT 50000")
    
    con.execute("CREATE VIEW s1_data AS SELECT * FROM read_parquet('intermediate/source1_normalized.parquet')")
    con.execute("CREATE VIEW s2_data AS SELECT * FROM read_parquet('intermediate/source2_normalized.parquet')")
    con.execute("CREATE VIEW s3_data AS SELECT * FROM read_parquet('intermediate/source3_normalized.parquet')")
    
    # Ground truth
    con.execute("""
        CREATE TABLE gt AS 
        SELECT 
            source1_entity_id,
            UNNEST(string_split(matched_entity_ids, ',')) as expected_match_id
        FROM read_csv('train_ground_truth.tsv', sep='\\t', header=True)
    """)
    
    con.execute("""
        CREATE VIEW pairs_joined AS
        SELECT 
            c.source1_entity_id,
            c.candidate_entity_id,
            c.candidate_source,
            s1.name_clean as s1_name,
            COALESCE(s2.name_clean, s3.name_clean) as s2_name,
            s1.address_clean as s1_address,
            COALESCE(s2.address_clean, s3.address_clean) as s2_address,
            s1.country_clean as s1_country,
            COALESCE(s2.country_clean, s3.country_clean) as s2_country,
            s1.name_compact as s1_name_comp,
            COALESCE(s2.name_compact, s3.name_compact) as s2_name_comp,
            s1.address_compact as s1_addr_comp,
            COALESCE(s2.address_compact, s3.address_compact) as s2_addr_comp,
            CASE WHEN gt.expected_match_id IS NOT NULL THEN 1 ELSE 0 END as is_match
        FROM cands c
        JOIN s1_data s1 ON c.source1_entity_id = s1.entity_id
        LEFT JOIN s2_data s2 ON c.candidate_entity_id = s2.entity_id AND c.candidate_source = 'S2'
        LEFT JOIN s3_data s3 ON c.candidate_entity_id = s3.entity_id AND c.candidate_source = 'S3'
        LEFT JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.candidate_entity_id = gt.expected_match_id
    """)

    con.execute("""
        CREATE TABLE sim_feats (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR,
            is_match INTEGER,
            name_sim FLOAT,
            addr_sim FLOAT,
            exact_country INTEGER,
            exact_name_comp INTEGER,
            exact_addr_comp INTEGER
        )
    """)

    total_pairs = con.execute("SELECT COUNT(*) FROM cands").fetchone()[0]
    print(f"Total pairs to process: {total_pairs}")
    
    res = con.execute("SELECT * FROM pairs_joined")
    rows = res.fetchall()
    
    texts1_name = [r[3] for r in rows]
    texts2_name = [r[4] for r in rows]
    name_sim = batch_cosine_sim(texts1_name, texts2_name, analyzer='char_wb', ngram_range=(3,4))
    
    texts1_addr = [r[5] for r in rows]
    texts2_addr = [r[6] for r in rows]
    addr_sim = batch_cosine_sim(texts1_addr, texts2_addr, analyzer='word', ngram_range=(1,2))
    
    out_rows = []
    for i, r in enumerate(rows):
        exact_country = 1 if r[7] == r[8] and r[7] is not None and r[7] != '' else 0
        exact_name_comp = 1 if r[9] == r[10] and r[9] is not None and r[9] != '' else 0
        exact_addr_comp = 1 if r[11] == r[12] and r[11] is not None and r[11] != '' else 0
        
        out_rows.append((
            str(r[0]),
            str(r[1]),
            int(r[13]),
            float(name_sim[i]),
            float(addr_sim[i]),
            exact_country,
            exact_name_comp,
            exact_addr_comp
        ))
        
    con.executemany("INSERT INTO sim_feats VALUES (?, ?, ?, ?, ?, ?, ?, ?)", out_rows)
    print(f"Processed {len(rows)} / {total_pairs} pairs | RAM: {get_memory_usage():.2f} MB")
        
    con.execute("COPY sim_feats TO 'intermediate/similarity_features_50k.parquet' (FORMAT PARQUET)")
    
    # ------------------
    # EVALUATION
    # ------------------
    print("Evaluating similarity features...")
    
    # Overall counts
    total_processed = con.execute("SELECT COUNT(*) FROM sim_feats").fetchone()[0]
    total_matches = con.execute("SELECT COUNT(*) FROM sim_feats WHERE is_match = 1").fetchone()[0]
    total_non_matches = total_processed - total_matches
    
    def get_stats(col, match_val):
        return con.execute(f"""
            SELECT 
                AVG({col}), 
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}),
                MIN({col}),
                MAX({col})
            FROM sim_feats WHERE is_match = {match_val}
        """).fetchone()

    name_match = get_stats('name_sim', 1)
    name_non = get_stats('name_sim', 0)
    
    addr_match = get_stats('addr_sim', 1)
    addr_non = get_stats('addr_sim', 0)
    
    def get_bool_stats(col):
        m = con.execute(f"SELECT SUM({col}) FROM sim_feats WHERE is_match = 1").fetchone()[0] or 0
        nm = con.execute(f"SELECT SUM({col}) FROM sim_feats WHERE is_match = 0").fetchone()[0] or 0
        return m, nm

    country_m, country_nm = get_bool_stats('exact_country')
    name_comp_m, name_comp_nm = get_bool_stats('exact_name_comp')
    addr_comp_m, addr_comp_nm = get_bool_stats('exact_addr_comp')

    os.makedirs('reports', exist_ok=True)
    with open('reports/similarity_feature_evaluation_50k.txt', 'w', encoding='utf-8') as f:
        f.write("SIMILARITY FEATURE EVALUATION\n")
        f.write("=============================\n")
        f.write(f"Total candidate pairs processed: {total_processed:,}\n")
        f.write(f"True matches: {total_matches:,}\n")
        f.write(f"Non-matches: {total_non_matches:,}\n\n")
        
        f.write("--- Name Similarity (TF-IDF Char 3-5 Grams) ---\n")
        f.write(f"Matches     -> Avg: {name_match[0]:.4f} | Median: {name_match[1]:.4f} | Min: {name_match[2]:.4f} | Max: {name_match[3]:.4f}\n")
        f.write(f"Non-matches -> Avg: {name_non[0]:.4f} | Median: {name_non[1]:.4f} | Min: {name_non[2]:.4f} | Max: {name_non[3]:.4f}\n\n")
        
        f.write("--- Address Similarity (TF-IDF Word 1-2 Grams) ---\n")
        f.write(f"Matches     -> Avg: {addr_match[0]:.4f} | Median: {addr_match[1]:.4f} | Min: {addr_match[2]:.4f} | Max: {addr_match[3]:.4f}\n")
        f.write(f"Non-matches -> Avg: {addr_non[0]:.4f} | Median: {addr_non[1]:.4f} | Min: {addr_non[2]:.4f} | Max: {addr_non[3]:.4f}\n\n")
        
        f.write("--- Exact Match Features ---\n")
        f.write("Exact Country:\n")
        f.write(f"  Matches:     {country_m:,} ({country_m/total_matches:.2%} of matches)\n")
        if total_non_matches > 0:
            f.write(f"  Non-matches: {country_nm:,} ({country_nm/total_non_matches:.2%} of non-matches)\n")
        
        f.write("Exact Name Compact:\n")
        f.write(f"  Matches:     {name_comp_m:,} ({name_comp_m/total_matches:.2%} of matches)\n")
        if total_non_matches > 0:
            f.write(f"  Non-matches: {name_comp_nm:,} ({name_comp_nm/total_non_matches:.2%} of non-matches)\n")
        
        f.write("Exact Address Compact:\n")
        f.write(f"  Matches:     {addr_comp_m:,} ({addr_comp_m/total_matches:.2%} of matches)\n")
        if total_non_matches > 0:
            f.write(f"  Non-matches: {addr_comp_nm:,} ({addr_comp_nm/total_non_matches:.2%} of non-matches)\n\n")
        
        f.write(f"Generation & Evaluation Runtime: {time.time() - start_time:.2f} seconds\n")
        f.write(f"Peak RAM: {get_memory_usage():.2f} MB\n")

    print(f"Done. Runtime: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
