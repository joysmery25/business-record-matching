import os
import duckdb
import argparse
import time
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def normalize_dataset(source_name: str, input_file: str, output_file: str, sample: bool = False):
    start_time = time.time()
    
    if os.path.exists(output_file) and not sample:
        print(f"Output file {output_file} already exists. Skipping.")
        return
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    con = duckdb.connect(database=':memory:')
    
    # Read the data, optionally limiting for tests
    limit_clause = "LIMIT 1000" if sample else ""
    
    con.execute(f"""
        CREATE VIEW source_data AS 
        SELECT * FROM read_csv('{input_file}', sep='\\t', header=True, all_varchar=True, null_padding=True)
        {limit_clause}
    """)
    
    orig_rows = con.execute("SELECT COUNT(*) FROM source_data").fetchone()[0]
    orig_unique_ids = con.execute("SELECT COUNT(DISTINCT entity_id) FROM source_data").fetchone()[0]
    
    print(f"Normalizing {input_file} (Sample: {sample})")
    
    # We define a macro-like string for the clean logic
    def clean_expr(col):
        # 1. Lowercase
        # 2. Replace punctuation and control chars with space
        # 3. Replace multiple spaces with a single space
        # 4. Trim
        base = f"LOWER({col})"
        no_punct = f"REGEXP_REPLACE({base}, '[[:punct:][:cntrl:]]+', ' ', 'g')"
        no_multi_space = f"REGEXP_REPLACE({no_punct}, '\\s+', ' ', 'g')"
        return f"TRIM({no_multi_space})"

    def compact_expr(col):
        # Remove all spaces from the clean expression
        return f"REPLACE({clean_expr(col)}, ' ', '')"

    def token_count_expr(col):
        # Count tokens by counting spaces in the clean string + 1 (if not empty)
        c_expr = clean_expr(col)
        return f"""
            CASE 
                WHEN {col} IS NULL THEN 0
                WHEN {c_expr} = '' THEN 0 
                ELSE LENGTH({c_expr}) - LENGTH(REPLACE({c_expr}, ' ', '')) + 1 
            END
        """
        
    sql = f"""
    COPY (
        SELECT 
            entity_id,
            business_name,
            business_address,
            country,
            
            -- Name Representations
            LOWER(TRIM(business_name)) AS name_lower,
            {clean_expr('business_name')} AS name_clean,
            {compact_expr('business_name')} AS name_compact,
            {token_count_expr('business_name')} AS name_token_count,
            
            -- Address Representations
            LOWER(TRIM(business_address)) AS address_lower,
            {clean_expr('business_address')} AS address_clean,
            {compact_expr('business_address')} AS address_compact,
            {token_count_expr('business_address')} AS address_token_count,
            
            -- Country
            TRIM(REGEXP_REPLACE(LOWER(country), '\\s+', ' ', 'g')) AS country_clean,
            
            -- Additional Features
            LENGTH(business_name) AS name_length,
            LENGTH({clean_expr('business_name')}) AS name_clean_length,
            LENGTH(business_address) AS address_length,
            LENGTH({clean_expr('business_address')}) AS address_clean_length,
            
            CASE WHEN TRIM(business_name) != '' AND business_name IS NOT NULL THEN 1 ELSE 0 END AS has_name,
            CASE WHEN TRIM(business_address) != '' AND business_address IS NOT NULL THEN 1 ELSE 0 END AS has_address,
            CASE WHEN TRIM(country) != '' AND country IS NOT NULL THEN 1 ELSE 0 END AS has_country
            
        FROM source_data
    ) TO '{output_file}' (FORMAT PARQUET);
    """
    
    con.execute(sql)
    
    # Validation after
    con.execute(f"CREATE VIEW result_data AS SELECT * FROM read_parquet('{output_file}')")
    new_rows = con.execute("SELECT COUNT(*) FROM result_data").fetchone()[0]
    new_unique_ids = con.execute("SELECT COUNT(DISTINCT entity_id) FROM result_data").fetchone()[0]
    
    file_size = os.path.getsize(output_file)
    runtime = time.time() - start_time
    mem_usage = get_memory_usage()
    
    print("-" * 50)
    print(f"Validation for {source_name}")
    print(f"Source | Original rows | Normalized rows | Unique IDs before | Unique IDs after")
    print(f"{source_name} | {orig_rows} | {new_rows} | {orig_unique_ids} | {new_unique_ids}")
    print(f"Output File Size: {file_size / (1024*1024):.2f} MB")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Peak RAM: ~{mem_usage:.2f} MB")
    print("-" * 50)
    
    if sample:
        print("\n--- 10 Examples ---")
        examples = con.execute("""
            SELECT 
                business_name, name_lower, name_clean, name_compact, name_token_count,
                business_address, address_lower, address_clean, address_compact, address_token_count,
                country, country_clean
            FROM result_data 
            WHERE business_name IS NOT NULL AND business_address IS NOT NULL
            LIMIT 10
        """).fetchall()
        for idx, ex in enumerate(examples):
            print(f"Example {idx+1}:")
            print(f"  Name Orig   : {ex[0]}")
            print(f"  Name Lower  : {ex[1]}")
            print(f"  Name Clean  : {ex[2]}")
            print(f"  Name Compact: {ex[3]}")
            print(f"  Name Tokens : {ex[4]}")
            print(f"  Addr Orig   : {ex[5]}")
            print(f"  Addr Lower  : {ex[6]}")
            print(f"  Addr Clean  : {ex[7]}")
            print(f"  Addr Compact: {ex[8]}")
            print(f"  Addr Tokens : {ex[9]}")
            print(f"  Country Orig: {ex[10]}")
            print(f"  Country Cln : {ex[11]}")
            print("")

def main():
    parser = argparse.ArgumentParser(description="Normalize business entity resolution TSV datasets.")
    parser.add_argument("--source", help="Source ID (e.g. 1, 2, 3)", required=False)
    parser.add_argument("--input", help="Input TSV file", required=False)
    parser.add_argument("--output", help="Output Parquet file", required=False)
    parser.add_argument("--sample", action="store_true", help="Process only 1000 rows for testing")
    parser.add_argument("--all", action="store_true", help="Process all three sources")
    args = parser.parse_args()
    
    # We must explicitly force utf-8 for stdout just in case it prints non-ascii chars to the windows console
    import sys
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    if args.all:
        sources = [
            ("Source 1", "train_source1.tsv", "intermediate/source1_normalized.parquet"),
            ("Source 2", "train_source2.tsv", "intermediate/source2_normalized.parquet"),
            ("Source 3", "train_source3.tsv", "intermediate/source3_normalized.parquet")
        ]
        for src_name, in_f, out_f in sources:
            normalize_dataset(src_name, in_f, out_f, args.sample)
    else:
        if not args.input or not args.output:
            print("Must specify --input and --output if not using --all")
            return
        src_name = f"Source {args.source}" if args.source else "Source X"
        normalize_dataset(src_name, args.input, args.output, args.sample)

if __name__ == "__main__":
    main()
