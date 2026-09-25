import os
import duckdb
import argparse
import time
import psutil

def profile_dataset(file_path: str, output_dir: str):
    start_time = time.time()
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    report_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_profile.txt")
    
    print(f"Profiling {file_path} ...")
    
    # Initialize DuckDB
    # Setting some memory limits if desired, but default DuckDB is quite good
    con = duckdb.connect(database=':memory:')
    
    # We use all_varchar to avoid casting issues with malformed lines for the first pass
    try:
        con.execute(f"""
            CREATE VIEW source_data AS 
            SELECT * FROM read_csv('{file_path}', sep='\\t', header=True, all_varchar=True, null_padding=True, auto_detect=True)
        """)
    except Exception as e:
        print(f"Error loading {file_path} into DuckDB: {e}")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Failed to read file: {e}\n")
        return
    
    columns_info = con.execute("DESCRIBE source_data").fetchall()
    column_names = [col[0] for col in columns_info]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Profiling Report for {file_name} ===\n")
        f.write(f"Exact file size on disk: {file_size:,} bytes\n")
        
        # Total rows
        try:
            total_rows = con.execute("SELECT COUNT(*) FROM source_data").fetchone()[0]
        except Exception as e:
            f.write(f"Error counting rows (possible malformed data): {e}\n")
            return
            
        f.write(f"Total rows: {total_rows:,}\n\n")
        
        f.write("--- Columns & Initial DuckDB Inferred Types ---\n")
        # Note: we read as all_varchar, so everything is VARCHAR initially in the view
        # We can look at the raw file again if we wanted to detect types, or just stick to standard
        # The user wants Data types. Since we enforced all_varchar, we might not get correct types.
        # Let's drop the all_varchar view and just try a regular read_csv, which gives types.
        # If it fails, we fall back to all_varchar.
        
        con_types = duckdb.connect(database=':memory:')
        try:
            con_types.execute(f"CREATE VIEW types_view AS SELECT * FROM read_csv('{file_path}', sep='\\t', header=True, sample_size=10000)")
            types_info = con_types.execute("DESCRIBE types_view").fetchall()
            type_dict = {row[0]: row[1] for row in types_info}
        except Exception:
            type_dict = {col: 'VARCHAR (fallback)' for col in column_names}
            
        for col in column_names:
            f.write(f"- {col}: {type_dict.get(col, 'VARCHAR')}\n")
        f.write("\n")
        
        # Missing values
        f.write("--- Missing Values & Null-like Values ---\n")
        null_like_expr = "UPPER(CAST({col} AS VARCHAR)) IN ('NULL', 'N/A', 'NA', '-', '')"
        for col in column_names:
            res = con.execute(f"""
                SELECT 
                    COUNT(CASE WHEN {col} IS NULL THEN 1 END),
                    COUNT(CASE WHEN {null_like_expr.format(col=col)} THEN 1 END)
                FROM source_data
            """).fetchone()
            actual_nulls = res[0]
            null_likes = res[1]
            total_missing = actual_nulls + null_likes
            perc = (total_missing / total_rows * 100) if total_rows > 0 else 0
            f.write(f"{col}: {total_missing:,} missing/null-like ({perc:.2f}%)\n")
        
        # entity_id stats
        f.write("\n--- Entity ID Stats ---\n")
        if 'entity_id' in column_names:
            unique_entities = con.execute("SELECT COUNT(DISTINCT entity_id) FROM source_data WHERE entity_id IS NOT NULL").fetchone()[0]
            dup_entities = con.execute("""
                SELECT COALESCE(SUM(cnt), 0) FROM (
                    SELECT COUNT(*) as cnt FROM source_data 
                    WHERE entity_id IS NOT NULL 
                    GROUP BY entity_id HAVING COUNT(*) > 1
                )
            """).fetchone()[0]
            f.write(f"Unique entity_id values: {unique_entities:,}\n")
            f.write(f"Rows with duplicate entity_id values: {dup_entities:,}\n")
        else:
            f.write("No 'entity_id' column found.\n")
            
        # business_name stats
        f.write("\n--- Business Name Stats ---\n")
        if 'business_name' in column_names:
            unique_names = con.execute("SELECT COUNT(DISTINCT business_name) FROM source_data WHERE business_name IS NOT NULL").fetchone()[0]
            f.write(f"Unique business names: {unique_names:,}\n")
            
            lengths = con.execute("""
                SELECT 
                    MIN(LENGTH(trim(CAST(business_name AS VARCHAR)))), 
                    MAX(LENGTH(trim(CAST(business_name AS VARCHAR)))), 
                    AVG(LENGTH(trim(CAST(business_name AS VARCHAR)))),
                    COUNT(CASE WHEN LENGTH(trim(CAST(business_name AS VARCHAR))) <= 2 THEN 1 END)
                FROM source_data 
                WHERE business_name IS NOT NULL
            """).fetchone()
            
            f.write(f"Min length: {lengths[0]}\n")
            f.write(f"Max length: {lengths[1]}\n")
            f.write(f"Average length: {lengths[2]:.2f}\n")
            
            short_names = lengths[3]
            short_perc = (short_names / total_rows * 100) if total_rows > 0 else 0
            f.write(f"Empty or extremely short names (<=2 chars): {short_names:,} ({short_perc:.2f}%)\n")
        else:
            f.write("No 'business_name' column found.\n")
        
        # business_address stats
        f.write("\n--- Business Address Stats ---\n")
        if 'business_address' in column_names:
            lengths_addr = con.execute("""
                SELECT 
                    MIN(LENGTH(trim(CAST(business_address AS VARCHAR)))), 
                    MAX(LENGTH(trim(CAST(business_address AS VARCHAR)))), 
                    AVG(LENGTH(trim(CAST(business_address AS VARCHAR)))),
                    COUNT(CASE WHEN LENGTH(trim(CAST(business_address AS VARCHAR))) <= 5 THEN 1 END)
                FROM source_data 
                WHERE business_address IS NOT NULL
            """).fetchone()
            
            f.write(f"Min length: {lengths_addr[0]}\n")
            f.write(f"Max length: {lengths_addr[1]}\n")
            f.write(f"Average length: {lengths_addr[2] if lengths_addr[2] else 0:.2f}\n")
            
            short_addrs = lengths_addr[3]
            short_addr_perc = (short_addrs / total_rows * 100) if total_rows > 0 else 0
            f.write(f"Empty or extremely short addresses (<=5 chars): {short_addrs:,} ({short_addr_perc:.2f}%)\n")
        else:
             f.write("No 'business_address' column found.\n")
            
        # country stats
        f.write("\n--- Country Stats ---\n")
        if 'country' in column_names:
            unique_countries = con.execute("SELECT COUNT(DISTINCT country) FROM source_data WHERE country IS NOT NULL").fetchone()[0]
            f.write(f"Unique countries: {unique_countries:,}\n")
            
            f.write("Country distribution (Top 20):\n")
            dist = con.execute("SELECT country, COUNT(*) as cnt FROM source_data GROUP BY country ORDER BY cnt DESC LIMIT 20").fetchall()
            for row in dist:
                f.write(f"  {row[0]}: {row[1]:,}\n")
        else:
            f.write("No 'country' column found.\n")
        
        # Whitespace checks
        f.write("\n--- Whitespace Issues ---\n")
        for col in ['business_name', 'business_address', 'country']:
            if col in column_names:
                leading_trailing = con.execute(f"""
                    SELECT COUNT(*) FROM source_data 
                    WHERE CAST({col} AS VARCHAR) != trim(CAST({col} AS VARCHAR)) AND {col} IS NOT NULL
                """).fetchone()[0]
                f.write(f"{col} with leading/trailing whitespace: {leading_trailing:,}\n")
                
        # Safe examples
        f.write("\n--- Safe Examples (Raw Structure) ---\n")
        examples = con.execute("SELECT * FROM source_data LIMIT 3").fetchall()
        for idx, ex in enumerate(examples):
            f.write(f"Row {idx+1}:\n")
            for i, val in enumerate(ex):
                f.write(f"  {column_names[i]}: {repr(val)}\n")
            f.write("\n")
            
        runtime = time.time() - start_time
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024)
        
        f.write("\n--- Performance ---\n")
        f.write(f"Runtime: {runtime:.2f} seconds\n")
        f.write(f"Approximate RAM usage: {ram_usage:.2f} MB\n")
        
    print(f"Profile saved to {report_path}")

def combine_reports(output_dir: str):
    combined_path = os.path.join(output_dir, "combined_profile.txt")
    print(f"Creating {combined_path} ...")
    with open(combined_path, 'w', encoding='utf-8') as cout:
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith("_profile.txt") and fname != "combined_profile.txt":
                cout.write(f"\\n{'='*50}\\n")
                cout.write(f"REPORT: {fname}\\n")
                cout.write(f"{'='*50}\\n\\n")
                with open(os.path.join(output_dir, fname), 'r', encoding='utf-8') as fin:
                    cout.write(fin.read())
                    cout.write("\\n")

def main():
    parser = argparse.ArgumentParser(description="Profile business entity resolution TSV datasets.")
    parser.add_argument("--files", nargs='+', help="List of TSV files to profile", required=True)
    parser.add_argument("--outdir", default="reports", help="Output directory for reports")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Will profile files: {args.files}")
    for f in args.files:
        profile_dataset(f, args.outdir)
        
    combine_reports(args.outdir)

if __name__ == "__main__":
    main()
