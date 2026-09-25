import duckdb
import time
import psutil
import os
from blocking import get_blocking_queries

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    print("Starting Candidate Generation on 10,000 S1 Sample...")
    start_time = time.time()
    
    # 16 GB constraint: limit memory usage safely
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB'")

    print("Sampling S1...")
    # Get exactly 10,000 deterministic random records
    con.execute("""
        CREATE TABLE s1_sample AS 
        SELECT * FROM read_parquet('intermediate/source1_normalized.parquet')
        ORDER BY md5(entity_id) LIMIT 10000
    """)

    con.execute("CREATE VIEW s2_data AS SELECT * FROM read_parquet('intermediate/source2_normalized.parquet')")
    con.execute("CREATE VIEW s3_data AS SELECT * FROM read_parquet('intermediate/source3_normalized.parquet')")

    con.execute("""
        CREATE TABLE candidates (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR,
            candidate_source VARCHAR,
            blocking_strategy VARCHAR
        )
    """)

    print("Executing strategies for S2...")
    for name, q in get_blocking_queries("s1_sample", "s2_data", "S2"):
        print(f"  Running: {name} on S2")
        t0 = time.time()
        con.execute(f"INSERT INTO candidates {q}")
        print(f"    Done in {time.time()-t0:.2f}s")

    print("Executing strategies for S3...")
    for name, q in get_blocking_queries("s1_sample", "s3_data", "S3"):
        print(f"  Running: {name} on S3")
        t0 = time.time()
        con.execute(f"INSERT INTO candidates {q}")
        print(f"    Done in {time.time()-t0:.2f}s")

    print("Deduplicating candidates...")
    con.execute("""
        CREATE TABLE dedup_candidates AS
        SELECT 
            source1_entity_id, 
            candidate_entity_id, 
            candidate_source,
            string_agg(DISTINCT blocking_strategy, ',') as blocking_strategies
        FROM candidates
        GROUP BY source1_entity_id, candidate_entity_id, candidate_source
    """)

    print("Exporting candidates...")
    con.execute("COPY dedup_candidates TO 'intermediate/blocking_sample_candidates.parquet' (FORMAT PARQUET)")
    
    print(f"Peak RAM: {get_memory_usage():.2f} MB")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
