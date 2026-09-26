import duckdb
import os

def main():
    con = duckdb.connect()
    file = "intermediate/similarity_features_10k.parquet"
    
    if not os.path.exists(file):
        print("=== FINAL VALIDATION ===")
        print(f"FAIL: File '{file}' does not exist.")
        return

    print("=== SCHEMA ===")
    schema = con.execute(f"DESCRIBE SELECT * FROM '{file}'").fetchall()
    for col in schema:
        print(col)

    print("\n=== ROW COUNT ===")
    count = con.execute(f"SELECT COUNT(*) FROM '{file}'").fetchone()[0]
    print(count)

    print("\n=== FIRST 5 ROWS ===")
    rows = con.execute(f"SELECT * FROM '{file}' LIMIT 5").fetchall()
    for row in rows:
        print(row)

    print("\n=== NULL CHECK ===")
    nulls = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE name_sim IS NULL) AS null_name_sim,
            COUNT(*) FILTER (WHERE addr_sim IS NULL) AS null_addr_sim,
            COUNT(*) FILTER (WHERE exact_country IS NULL) AS null_exact_country,
            COUNT(*) FILTER (WHERE exact_name IS NULL) AS null_exact_name,
            COUNT(*) FILTER (WHERE exact_address IS NULL) AS null_exact_address,
            COUNT(*) FILTER (WHERE name_len_diff IS NULL) AS null_name_len,
            COUNT(*) FILTER (WHERE addr_len_diff IS NULL) AS null_addr_len
        FROM '{file}'
    """).fetchone()
    print(f"null_name_sim: {nulls[0]}")
    print(f"null_addr_sim: {nulls[1]}")
    print(f"null_exact_country: {nulls[2]}")
    print(f"null_exact_name: {nulls[3]}")
    print(f"null_exact_address: {nulls[4]}")
    print(f"null_name_len_diff: {nulls[5]}")
    print(f"null_addr_len_diff: {nulls[6]}")

    print("\n=== MATCH COUNTS ===")
    matches = con.execute(f"""
        SELECT is_match, COUNT(*)
        FROM '{file}'
        GROUP BY is_match
        ORDER BY is_match
    """).fetchall()
    for m in matches:
        print(f"is_match={m[0]}: {m[1]}")

    print("\n=== SIMILARITY RANGE CHECK ===")
    ranges = con.execute(f"""
        SELECT 
            MIN(name_sim), MAX(name_sim),
            MIN(addr_sim), MAX(addr_sim)
        FROM '{file}'
    """).fetchone()
    min_name, max_name, min_addr, max_addr = ranges
    print(f"MIN(name_sim): {min_name}")
    print(f"MAX(name_sim): {max_name}")
    print(f"MIN(addr_sim): {min_addr}")
    print(f"MAX(addr_sim): {max_addr}")

    print("\n=== FEATURE SUMMARY ===")
    summary = con.execute(f"""
        SELECT
            AVG(name_sim) FILTER (WHERE is_match = 1) AS avg_name_sim_match,
            AVG(name_sim) FILTER (WHERE is_match = 0) AS avg_name_sim_non_match,
            AVG(addr_sim) FILTER (WHERE is_match = 1) AS avg_addr_sim_match,
            AVG(addr_sim) FILTER (WHERE is_match = 0) AS avg_addr_sim_non_match,
            SUM(exact_country) AS sum_exact_country,
            SUM(exact_name) AS sum_exact_name,
            SUM(exact_address) AS sum_exact_address
        FROM '{file}'
    """).fetchone()
    print(f"average name_sim for matches: {summary[0]:.4f}")
    print(f"average name_sim for non-matches: {summary[1]:.4f}")
    print(f"average addr_sim for matches: {summary[2]:.4f}")
    print(f"average addr_sim for non-matches: {summary[3]:.4f}")
    print(f"count of exact_country = 1: {summary[4]}")
    print(f"count of exact_name = 1: {summary[5]}")
    print(f"count of exact_address = 1: {summary[6]}")

    # Validate
    passed = True
    errors = []
    if count != 10000:
        passed = False
        errors.append(f"Row count is {count}, expected 10000")
        
    expected_cols = {'source1_entity_id', 'candidate_entity_id', 'is_match', 'name_sim', 'addr_sim', 'exact_country', 'exact_name', 'exact_address', 'name_len_diff', 'addr_len_diff'}
    actual_cols = {col[0] for col in schema}
    missing = expected_cols - actual_cols
    if missing:
        passed = False
        errors.append(f"Missing columns: {missing}")

    total_nulls = sum(nulls)
    if total_nulls > 0:
        passed = False
        errors.append(f"Found {total_nulls} NULLs across feature columns")
        
    if min_name < 0 or max_name > 1 or min_addr < 0 or max_addr > 1:
        passed = False
        errors.append("Cosine similarity out of range [0, 1]")

    print("\n=== FINAL VALIDATION ===")
    if passed:
        print("PASS")
    else:
        print("FAIL")
        for err in errors:
            print(f"- {err}")

    con.close()

if __name__ == "__main__":
    main()