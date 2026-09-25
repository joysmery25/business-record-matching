import os
import duckdb
import time
import psutil
import codecs
import sys

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def profile_parquet(source_name: str, file_path: str, f_out):
    start_time = time.time()
    con = duckdb.connect(database=':memory:')
    con.execute(f"CREATE VIEW pq_data AS SELECT * FROM read_parquet('{file_path}')")
    
    f_out.write(f"==================================================\n")
    f_out.write(f"PROFILING: {source_name}\n")
    f_out.write(f"==================================================\n\n")
    
    # 1. Total rows & 2. Unique entity_id
    rows, uniq_ids = con.execute("SELECT COUNT(*), COUNT(DISTINCT entity_id) FROM pq_data").fetchone()
    f_out.write(f"1. Total rows: {rows:,}\n")
    f_out.write(f"2. Unique entity_id: {uniq_ids:,}\n")
    
    # 3. Unique name_clean & 4. Unique name_compact
    uniq_nc, uniq_ncomp = con.execute("SELECT COUNT(DISTINCT name_clean), COUNT(DISTINCT name_compact) FROM pq_data").fetchone()
    f_out.write(f"3. Unique name_clean: {uniq_nc:,}\n")
    f_out.write(f"4. Unique name_compact: {uniq_ncomp:,}\n")
    
    # 5. Unique address_clean & 6. Unique address_compact
    uniq_ac, uniq_acomp = con.execute("SELECT COUNT(DISTINCT address_clean), COUNT(DISTINCT address_compact) FROM pq_data").fetchone()
    f_out.write(f"5. Unique address_clean: {uniq_ac:,}\n")
    f_out.write(f"6. Unique address_compact: {uniq_acomp:,}\n\n")
    
    # 7. Country distribution
    f_out.write("7. Country Distribution:\n")
    countries = con.execute("SELECT country_clean, COUNT(*) as cnt FROM pq_data GROUP BY country_clean ORDER BY cnt DESC").fetchall()
    for c, cnt in countries:
        f_out.write(f"   {c}: {cnt:,}\n")
    f_out.write("\n")
    
    # 8. Missing/empty address & 9. Missing/empty name
    miss_addr = con.execute("SELECT COUNT(*) FROM pq_data WHERE address_clean IS NULL OR address_clean = ''").fetchone()[0]
    miss_name = con.execute("SELECT COUNT(*) FROM pq_data WHERE name_clean IS NULL OR name_clean = ''").fetchone()[0]
    f_out.write(f"8. Missing/empty address counts: {miss_addr:,}\n")
    f_out.write(f"9. Missing/empty name counts: {miss_name:,}\n\n")
    
    # Distributions
    def get_dist(col):
        return con.execute(f"""
            SELECT 
                MIN({col}), MAX({col}), AVG({col}),
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {col})
            FROM pq_data WHERE {col} > 0
        """).fetchone()

    dist_nl = get_dist("name_clean_length")
    f_out.write(f"10. Name length distribution (Min/Max/Avg/Median): {dist_nl[0]} / {dist_nl[1]} / {dist_nl[2]:.2f} / {dist_nl[3]:.1f}\n")
    
    dist_al = get_dist("address_clean_length")
    f_out.write(f"11. Address length distribution (Min/Max/Avg/Median): {dist_al[0]} / {dist_al[1]} / {dist_al[2]:.2f} / {dist_al[3]:.1f}\n")
    
    dist_nt = get_dist("name_token_count")
    f_out.write(f"12. Name token-count distribution (Min/Max/Avg/Median): {dist_nt[0]} / {dist_nt[1]} / {dist_nt[2]:.2f} / {dist_nt[3]:.1f}\n")
    
    dist_at = get_dist("address_token_count")
    f_out.write(f"13. Address token-count distribution (Min/Max/Avg/Median): {dist_at[0]} / {dist_at[1]} / {dist_at[2]:.2f} / {dist_at[3]:.1f}\n\n")
    
    # Frequency Stats Function
    def print_freq_stats(field, label):
        f_out.write(f"--- {label} Frequency Statistics ---\n")
        bins = con.execute(f"""
            SELECT 
                COUNT(CASE WHEN cnt = 1 THEN 1 END),
                COUNT(CASE WHEN cnt BETWEEN 2 AND 5 THEN 1 END),
                COUNT(CASE WHEN cnt BETWEEN 6 AND 20 THEN 1 END),
                COUNT(CASE WHEN cnt BETWEEN 21 AND 100 THEN 1 END),
                COUNT(CASE WHEN cnt > 100 THEN 1 END)
            FROM (
                SELECT {field}, COUNT(*) as cnt 
                FROM pq_data 
                WHERE {field} IS NOT NULL AND {field} != '' 
                GROUP BY {field}
            )
        """).fetchone()
        f_out.write(f"Appearing once: {bins[0]:,}\n")
        f_out.write(f"Appearing 2-5 times: {bins[1]:,}\n")
        f_out.write(f"Appearing 6-20 times: {bins[2]:,}\n")
        f_out.write(f"Appearing 21-100 times: {bins[3]:,}\n")
        f_out.write(f"Appearing >100 times: {bins[4]:,}\n\n")
        
        f_out.write(f"Top 50 Most Frequent {label}s:\n")
        top_50 = con.execute(f"""
            SELECT {field}, COUNT(*) as cnt 
            FROM pq_data 
            WHERE {field} IS NOT NULL AND {field} != '' 
            GROUP BY {field} 
            ORDER BY cnt DESC 
            LIMIT 50
        """).fetchall()
        for v, cnt in top_50:
            f_out.write(f"  {cnt:,} | {v}\n")
        f_out.write("\n")

    print_freq_stats("name_clean", "Name")
    print_freq_stats("address_clean", "Address")
    
    f_out.write(f"Execution time for {source_name}: {time.time() - start_time:.2f} seconds\n")
    f_out.write(f"Peak RAM during {source_name}: {get_memory_usage():.2f} MB\n\n")

def main():
    start_time = time.time()
    os.makedirs('reports', exist_ok=True)
    report_file = 'reports/normalized_data_profile.txt'
    
    sources = [
        ("Source 1", "intermediate/source1_normalized.parquet"),
        ("Source 2", "intermediate/source2_normalized.parquet"),
        ("Source 3", "intermediate/source3_normalized.parquet")
    ]
    
    with codecs.open(report_file, 'w', encoding='utf-8') as f:
        for name, path in sources:
            if os.path.exists(path):
                print(f"Profiling {name}...")
                profile_parquet(name, path, f)
            else:
                print(f"{path} not found.")
                f.write(f"File not found: {path}\n")

    total_time = time.time() - start_time
    peak_ram = get_memory_usage()
    
    # Calculate disk usage for the intermediate parquet files
    total_size = sum(os.path.getsize(path) for _, path in sources if os.path.exists(path))
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\nProfiling complete! Report saved to {report_file}")
    print(f"Total Runtime: {total_time:.2f} seconds")
    print(f"Final Peak RAM: ~{peak_ram:.2f} MB")
    print(f"Total Disk Usage for Parquet files: {total_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
