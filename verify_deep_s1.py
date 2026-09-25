import duckdb
import time
import psutil
import os
import sys
import codecs

# Force stdout to utf-8 for unicode chars
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

start_time = time.time()
con = duckdb.connect(database=':memory:')

tsv_path = 'train_source1.tsv'
pq_path = 'intermediate/source1_normalized.parquet'

con.execute(f"CREATE VIEW tsv_data AS SELECT * FROM read_csv('{tsv_path}', sep='\\t', header=True, all_varchar=True, null_padding=True)")
con.execute(f"CREATE VIEW pq_data AS SELECT * FROM read_parquet('{pq_path}')")

print("--- 1. Row and Unique ID Counts ---")
tsv_rows = con.execute("SELECT COUNT(*) FROM tsv_data").fetchone()[0]
pq_rows = con.execute("SELECT COUNT(*) FROM pq_data").fetchone()[0]
tsv_unique = con.execute("SELECT COUNT(DISTINCT entity_id) FROM tsv_data").fetchone()[0]
pq_unique = con.execute("SELECT COUNT(DISTINCT entity_id) FROM pq_data").fetchone()[0]

print(f"TSV Rows: {tsv_rows} | Parquet Rows: {pq_rows}")
print(f"TSV Unique IDs: {tsv_unique} | Parquet Unique IDs: {pq_unique}")

print("\n--- 2. Exact Preservation of Original Columns ---")
mismatch_query = """
SELECT COUNT(*) FROM tsv_data t
JOIN pq_data p ON t.entity_id = p.entity_id
WHERE 
   COALESCE(t.business_name, '') != COALESCE(p.business_name, '')
   OR COALESCE(t.business_address, '') != COALESCE(p.business_address, '')
   OR COALESCE(t.country, '') != COALESCE(p.country, '')
"""
mismatches = con.execute(mismatch_query).fetchone()[0]
print(f"Number of mismatched original fields: {mismatches}")

print("\n--- 3. Verify Expected Normalized Columns Exist ---")
columns = [c[0] for c in con.execute("DESCRIBE pq_data").fetchall()]
expected = [
    'name_lower', 'name_clean', 'name_compact', 'name_token_count', 'name_length', 'name_clean_length', 'has_name',
    'address_lower', 'address_clean', 'address_compact', 'address_token_count', 'address_length', 'address_clean_length', 'has_address',
    'country_clean', 'has_country'
]
missing_cols = [c for c in expected if c not in columns]
print(f"Missing expected columns: {missing_cols if missing_cols else 'None. All exist!'}")

print("\n--- 4. NULL / Empty Counts ---")
for col in ['name_lower', 'name_clean', 'name_compact', 'address_lower', 'address_clean', 'address_compact', 'country_clean']:
    cnt = con.execute(f"SELECT COUNT(*) FROM pq_data WHERE {col} IS NULL OR {col} = ''").fetchone()[0]
    print(f"{col} empty/NULL count: {cnt}")

print("\n--- 5. Random 20 Records Sample ---")
# Using random() in order by for a random sample
sample = con.execute("""
    SELECT 
        business_name, name_lower, name_clean, name_compact,
        business_address, address_lower, address_clean, address_compact,
        country, country_clean
    FROM pq_data
    WHERE business_address IS NOT NULL AND business_name IS NOT NULL
    USING SAMPLE 20
""").fetchall()

for i, row in enumerate(sample):
    print(f"--- Record {i+1} ---")
    print(f"Name (Orig)    : {row[0]}")
    print(f"Name (Lower)   : {row[1]}")
    print(f"Name (Clean)   : {row[2]}")
    print(f"Name (Compact) : {row[3]}")
    print(f"Addr (Orig)    : {row[4]}")
    print(f"Addr (Lower)   : {row[5]}")
    print(f"Addr (Clean)   : {row[6]}")
    print(f"Addr (Compact) : {row[7]}")
    print(f"Country (Orig) : {row[8]}")
    print(f"Country (Clean): {row[9]}")
    
exec_time = time.time() - start_time
peak_ram = get_memory_usage()
print("\n--- 7. Execution Stats ---")
print(f"Execution Time: {exec_time:.2f} seconds")
print(f"Peak RAM: {peak_ram:.2f} MB")
