import duckdb
import time
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    print("Starting Final Candidate Generation...")
    start_time = time.time()
    
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='10GB'")

    print("Loading data...")
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

    print("Precomputing frequencies for S2+S3 (this takes a moment)...")
    con.execute("""
        CREATE TABLE name_tokens_freq AS
        SELECT token, COUNT(*) as freq
        FROM (
            SELECT UNNEST(string_split(name_clean, ' ')) as token FROM s2_data WHERE name_clean != ''
            UNION ALL
            SELECT UNNEST(string_split(name_clean, ' ')) as token FROM s3_data WHERE name_clean != ''
        )
        WHERE length(token) >= 4
        GROUP BY token
    """)
    
    con.execute("""
        CREATE TABLE addr_tokens_freq AS
        SELECT token, COUNT(*) as freq
        FROM (
            SELECT UNNEST(string_split(address_clean, ' ')) as token FROM s2_data WHERE address_clean != ''
            UNION ALL
            SELECT UNNEST(string_split(address_clean, ' ')) as token FROM s3_data WHERE address_clean != ''
        )
        WHERE length(token) >= 4
        GROUP BY token
    """)
    
    con.execute("""
        CREATE TABLE prefix8_freq AS
        SELECT prefix, COUNT(*) as freq
        FROM (
            SELECT substring(name_compact, 1, 8) as prefix FROM s2_data WHERE length(name_compact) >= 8
            UNION ALL
            SELECT substring(name_compact, 1, 8) as prefix FROM s3_data WHERE length(name_compact) >= 8
        )
        GROUP BY prefix
    """)

    print("Creating tokenized tables for fast joins...")
    # S1
    con.execute("""
        CREATE TABLE s1_nt AS 
        SELECT entity_id, country_clean, 
               regexp_extract(address_clean, '\\b\\d{5,6}\\b', 0) as postal,
               regexp_extract(address_clean, '^\\d+', 0) as house_num,
               substring(name_compact, 1, 5) as prefix5,
               token
        FROM (SELECT *, UNNEST(string_split(name_clean, ' ')) as token FROM s1_sample)
    """)
    con.execute("CREATE TABLE s1_at AS SELECT entity_id, token FROM (SELECT entity_id, UNNEST(string_split(address_clean, ' ')) as token FROM s1_sample)")

    # S2
    con.execute("""
        CREATE TABLE s2_nt AS 
        SELECT entity_id, country_clean, 
               regexp_extract(address_clean, '\\b\\d{5,6}\\b', 0) as postal,
               regexp_extract(address_clean, '^\\d+', 0) as house_num,
               substring(name_compact, 1, 5) as prefix5,
               token
        FROM (SELECT *, UNNEST(string_split(name_clean, ' ')) as token FROM s2_data)
        WHERE token IN (SELECT token FROM name_tokens_freq WHERE freq < 20000)
    """)
    con.execute("CREATE TABLE s2_at AS SELECT entity_id, token FROM (SELECT entity_id, UNNEST(string_split(address_clean, ' ')) as token FROM s2_data) WHERE token IN (SELECT token FROM addr_tokens_freq WHERE freq < 10000)")

    # S3
    con.execute("""
        CREATE TABLE s3_nt AS 
        SELECT entity_id, country_clean, 
               regexp_extract(address_clean, '\\b\\d{5,6}\\b', 0) as postal,
               regexp_extract(address_clean, '^\\d+', 0) as house_num,
               substring(name_compact, 1, 5) as prefix5,
               token
        FROM (SELECT *, UNNEST(string_split(name_clean, ' ')) as token FROM s3_data)
        WHERE token IN (SELECT token FROM name_tokens_freq WHERE freq < 20000)
    """)
    con.execute("CREATE TABLE s3_at AS SELECT entity_id, token FROM (SELECT entity_id, UNNEST(string_split(address_clean, ' ')) as token FROM s3_data) WHERE token IN (SELECT token FROM addr_tokens_freq WHERE freq < 10000)")

    def get_queries(s2_table, s2_nt, s2_at, source_name):
        return [
            ("exact_name", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'exact_name' FROM s1_sample s1 JOIN {s2_table} s2 ON s1.name_compact = s2.name_compact WHERE length(s1.name_compact) >= 5"),
            ("exact_address_country", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'exact_address_country' FROM s1_sample s1 JOIN {s2_table} s2 ON s1.address_compact = s2.address_compact AND s1.country_clean = s2.country_clean WHERE length(s1.address_compact) >= 8"),
            ("house_num_name_prefix5", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'house_num_name_prefix5' FROM s1_sample s1 JOIN {s2_table} s2 ON substring(s1.name_compact, 1, 5) = substring(s2.name_compact, 1, 5) AND regexp_extract(s1.address_clean, '^\\d+', 0) = regexp_extract(s2.address_clean, '^\\d+', 0) WHERE regexp_extract(s1.address_clean, '^\\d+', 0) != '' AND length(s1.name_compact) >= 5"),
            ("first_word_name_address", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'first_word_name_address' FROM s1_sample s1 JOIN {s2_table} s2 ON split_part(s1.name_clean, ' ', 1) = split_part(s2.name_clean, ' ', 1) AND split_part(s1.address_clean, ' ', 1) = split_part(s2.address_clean, ' ', 1) WHERE length(split_part(s1.name_clean, ' ', 1)) >= 4 AND length(split_part(s1.address_clean, ' ', 1)) >= 4"),
            ("postal_name_prefix4", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'postal_name_prefix4' FROM s1_sample s1 JOIN {s2_table} s2 ON substring(s1.name_compact, 1, 4) = substring(s2.name_compact, 1, 4) AND regexp_extract(s1.address_clean, '\\b\\d{{5,6}}\\b', 0) = regexp_extract(s2.address_clean, '\\b\\d{{5,6}}\\b', 0) WHERE regexp_extract(s1.address_clean, '\\b\\d{{5,6}}\\b', 0) != '' AND length(s1.name_compact) >= 4"),
            # REMOVED rare_name_token_country entirely.
            ("rare_name_prefix8_country", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'rare_name_prefix8_country' FROM s1_sample s1 JOIN {s2_table} s2 ON substring(s1.name_compact, 1, 8) = substring(s2.name_compact, 1, 8) AND s1.country_clean = s2.country_clean WHERE length(s1.name_compact) >= 8 AND length(s2.name_compact) >= 8 AND substring(s1.name_compact, 1, 8) IN (SELECT prefix FROM prefix8_freq WHERE freq < 5000)"),
            ("rare_name_token_postal", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'rare_name_token_postal' FROM s1_nt s1 JOIN {s2_nt} s2 ON s1.token = s2.token AND s1.postal = s2.postal WHERE s1.postal != ''"),
            ("name_prefix5_postal", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'name_prefix5_postal' FROM s1_nt s1 JOIN {s2_nt} s2 ON s1.prefix5 = s2.prefix5 AND s1.postal = s2.postal WHERE s1.postal != '' AND length(s1.prefix5) = 5"),
            ("rare_name_token_rare_addr_token", f"SELECT s1n.entity_id, s2n.entity_id, '{source_name}', 'rare_name_token_rare_addr_token' FROM s1_nt s1n JOIN {s2_nt} s2n ON s1n.token = s2n.token JOIN s1_at s1a ON s1n.entity_id = s1a.entity_id JOIN {s2_at} s2a ON s2n.entity_id = s2a.entity_id AND s1a.token = s2a.token"),
            ("house_num_rare_name_token", f"SELECT s1.entity_id, s2.entity_id, '{source_name}', 'house_num_rare_name_token' FROM s1_nt s1 JOIN {s2_nt} s2 ON s1.token = s2.token AND s1.house_num = s2.house_num WHERE s1.house_num != ''")
        ]

    for name, q in get_queries("s2_data", "s2_nt", "s2_at", "S2"):
        print(f"Running {name} on S2")
        con.execute(f"INSERT INTO candidates {q}")
    for name, q in get_queries("s3_data", "s3_nt", "s3_at", "S3"):
        print(f"Running {name} on S3")
        con.execute(f"INSERT INTO candidates {q}")

    print("Deduplicating candidates...")
    con.execute("""
        CREATE TABLE dedup_candidates AS
        SELECT 
            source1_entity_id, candidate_entity_id, candidate_source,
            string_agg(DISTINCT blocking_strategy, ',') as blocking_strategies
        FROM candidates
        GROUP BY source1_entity_id, candidate_entity_id, candidate_source
    """)

    con.execute("COPY dedup_candidates TO 'intermediate/blocking_final_candidates.parquet' (FORMAT PARQUET)")
    print(f"Peak RAM: {get_memory_usage():.2f} MB")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
