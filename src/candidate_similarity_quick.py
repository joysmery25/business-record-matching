import duckdb
import time
import os
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gc

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def batch_cosine_sim(texts1, texts2, analyzer, ngram_range):
    texts1 = ["" if t is None else str(t) for t in texts1]
    texts2 = ["" if t is None else str(t) for t in texts2]
    
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    try:
        vec.fit(texts1 + texts2)
        X1 = vec.transform(texts1)
        X2 = vec.transform(texts2)
        sims = X1.multiply(X2).sum(axis=1).A1
    except ValueError:
        sims = np.zeros(len(texts1))
    return sims

def main():
    print("Starting Similarity Feature Generation (Quick 10K)...")
    start_time = time.time()
    
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB'")
    
    # 1. Load exactly 10,000 reproducible candidate pairs into a materialized temp table
    con.execute("SELECT setseed(0.42)")
    con.execute("""
        CREATE TABLE sample_10k AS 
        SELECT source1_entity_id, candidate_entity_id, candidate_source
        FROM read_parquet('intermediate/blocking_final_candidates.parquet')
        ORDER BY random() 
        LIMIT 10000
    """)
    
    # 2. Extract needed entity IDs to filter Parquet reads
    con.execute("""
        CREATE TABLE needed_s1 AS SELECT DISTINCT source1_entity_id as entity_id FROM sample_10k
    """)
    con.execute("""
        CREATE TABLE needed_s2 AS SELECT DISTINCT candidate_entity_id as entity_id FROM sample_10k WHERE candidate_source = 'S2'
    """)
    con.execute("""
        CREATE TABLE needed_s3 AS SELECT DISTINCT candidate_entity_id as entity_id FROM sample_10k WHERE candidate_source = 'S3'
    """)
    
    # 3. Read only necessary rows from normalized files
    con.execute("""
        CREATE TABLE s1_data AS 
        SELECT p.* FROM read_parquet('intermediate/source1_normalized.parquet') p
        JOIN needed_s1 n ON p.entity_id = n.entity_id
    """)
    
    con.execute("""
        CREATE TABLE s2_data AS 
        SELECT p.* FROM read_parquet('intermediate/source2_normalized.parquet') p
        JOIN needed_s2 n ON p.entity_id = n.entity_id
    """)
    
    con.execute("""
        CREATE TABLE s3_data AS 
        SELECT p.* FROM read_parquet('intermediate/source3_normalized.parquet') p
        JOIN needed_s3 n ON p.entity_id = n.entity_id
    """)
    
    # Ground truth for labels
    con.execute("""
        CREATE TABLE gt AS 
        SELECT 
            source1_entity_id,
            UNNEST(string_split(matched_entity_ids, ',')) as expected_match_id
        FROM read_csv('train_ground_truth.tsv', sep='\\t', header=True)
    """)
    
    # Join everything together
    con.execute("""
        CREATE TABLE pairs_joined AS
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
        FROM sample_10k c
        LEFT JOIN s1_data s1 ON c.source1_entity_id = s1.entity_id
        LEFT JOIN s2_data s2 ON c.candidate_entity_id = s2.entity_id AND c.candidate_source = 'S2'
        LEFT JOIN s3_data s3 ON c.candidate_entity_id = s3.entity_id AND c.candidate_source = 'S3'
        LEFT JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.candidate_entity_id = gt.expected_match_id
    """)

    res = con.execute("SELECT * FROM pairs_joined")
    rows = res.fetchall()
    
    print(f"Loaded {len(rows)} rows into memory. Doing TF-IDF...")
    
    # Extract columns for vectorization
    texts1_name = [r[3] for r in rows]
    texts2_name = [r[4] for r in rows]
    
    texts1_addr = [r[5] for r in rows]
    texts2_addr = [r[6] for r in rows]
    
    # Calculate cosines
    name_sim = batch_cosine_sim(texts1_name, texts2_name, analyzer='char_wb', ngram_range=(3,4))
    addr_sim = batch_cosine_sim(texts1_addr, texts2_addr, analyzer='word', ngram_range=(1,2))
    
    con.execute("""
        CREATE TABLE sim_feats (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR,
            is_match INTEGER,
            name_sim FLOAT,
            addr_sim FLOAT,
            exact_country INTEGER,
            exact_name INTEGER,
            exact_address INTEGER,
            name_len_diff INTEGER,
            addr_len_diff INTEGER
        )
    """)
    
    out_rows = []
    for i, r in enumerate(rows):
        n1 = r[3] if r[3] is not None else ""
        n2 = r[4] if r[4] is not None else ""
        a1 = r[5] if r[5] is not None else ""
        a2 = r[6] if r[6] is not None else ""
        
        exact_country = 1 if r[7] == r[8] and r[7] is not None and r[7] != '' else 0
        exact_name_comp = 1 if r[9] == r[10] and r[9] is not None and r[9] != '' else 0
        exact_addr_comp = 1 if r[11] == r[12] and r[11] is not None and r[11] != '' else 0
        
        name_len_diff = abs(len(n1) - len(n2))
        addr_len_diff = abs(len(a1) - len(a2))
        
        out_rows.append((
            str(r[0]),
            str(r[1]),
            int(r[13]),
            float(name_sim[i]),
            float(addr_sim[i]),
            exact_country,
            exact_name_comp,
            exact_addr_comp,
            name_len_diff,
            addr_len_diff
        ))
        
    con.executemany("INSERT INTO sim_feats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", out_rows)
    print(f"Processed {len(rows)} pairs. RAM: {get_memory_usage():.2f} MB")
        
    con.execute("COPY sim_feats TO 'intermediate/similarity_features_10k.parquet' (FORMAT PARQUET)")
    
    # Evaluate
    total_processed = con.execute("SELECT COUNT(*) FROM sim_feats").fetchone()[0]
    total_matches = con.execute("SELECT COUNT(*) FROM sim_feats WHERE is_match = 1").fetchone()[0]
    total_non_matches = total_processed - total_matches
    
    def get_stats(col, match_val):
        return con.execute(f"""
            SELECT 
                MIN({col}),
                MAX({col}),
                AVG({col}), 
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col})
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
    name_comp_m, name_comp_nm = get_bool_stats('exact_name')
    addr_comp_m, addr_comp_nm = get_bool_stats('exact_address')

    os.makedirs('reports', exist_ok=True)
    with open('reports/similarity_feature_evaluation_10k.txt', 'w', encoding='utf-8') as f:
        f.write("SIMILARITY FEATURE EVALUATION (10K QUICK)\n")
        f.write("=========================================\n")
        f.write(f"Total candidate pairs processed: {total_processed:,}\n")
        f.write(f"True matches: {total_matches:,}\n")
        f.write(f"Non-matches: {total_non_matches:,}\n\n")
        
        f.write("--- Name Similarity (TF-IDF Char 3-4 Grams) ---\n")
        f.write(f"Matches     -> Min: {name_match[0]:.4f} | Max: {name_match[1]:.4f} | Mean: {name_match[2]:.4f} | Median: {name_match[3]:.4f}\n")
        if total_non_matches > 0:
            f.write(f"Non-matches -> Min: {name_non[0]:.4f} | Max: {name_non[1]:.4f} | Mean: {name_non[2]:.4f} | Median: {name_non[3]:.4f}\n\n")
        
        f.write("--- Address Similarity (TF-IDF Word 1-2 Grams) ---\n")
        f.write(f"Matches     -> Min: {addr_match[0]:.4f} | Max: {addr_match[1]:.4f} | Mean: {addr_match[2]:.4f} | Median: {addr_match[3]:.4f}\n")
        if total_non_matches > 0:
            f.write(f"Non-matches -> Min: {addr_non[0]:.4f} | Max: {addr_non[1]:.4f} | Mean: {addr_non[2]:.4f} | Median: {addr_non[3]:.4f}\n\n")
        
        f.write("--- Exact Match Features ---\n")
        f.write("Exact Country:\n")
        f.write(f"  Matches:     {country_m:,}\n")
        if total_non_matches > 0:
            f.write(f"  Non-matches: {country_nm:,}\n")
        
        f.write("Exact Name Compact:\n")
        f.write(f"  Matches:     {name_comp_m:,}\n")
        if total_non_matches > 0:
            f.write(f"  Non-matches: {name_comp_nm:,}\n")
        
        f.write("Exact Address Compact:\n")
        f.write(f"  Matches:     {addr_comp_m:,}\n")
        if total_non_matches > 0:
            f.write(f"  Non-matches: {addr_comp_nm:,}\n\n")
        
        f.write(f"Generation & Evaluation Runtime: {time.time() - start_time:.2f} seconds\n")
        f.write(f"Peak RAM: {get_memory_usage():.2f} MB\n")

    print(f"Done. Runtime: {time.time() - start_time:.2f}s, Peak RAM: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    main()
