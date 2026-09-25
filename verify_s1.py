import duckdb
import os
import time

out_file = 'intermediate/source1_normalized.parquet'
if os.path.exists(out_file):
    print(f"Verifying existing file: {out_file}")
    con = duckdb.connect()
    start = time.time()
    con.execute(f"CREATE VIEW res AS SELECT * FROM read_parquet('{out_file}')")
    rows = con.execute("SELECT COUNT(*) FROM res").fetchone()[0]
    unique_ids = con.execute("SELECT COUNT(DISTINCT entity_id) FROM res").fetchone()[0]
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"Output rows: {rows}")
    print(f"Unique IDs: {unique_ids}")
    print(f"Output File Size: {size_mb:.2f} MB")
    print(f"Verification runtime: {time.time() - start:.2f} sec")
else:
    print("File does not exist.")
