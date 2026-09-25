import duckdb

con = duckdb.connect()
con.execute("CREATE VIEW tsv AS SELECT * FROM read_csv('train_source2.tsv', sep='\\t', header=True, all_varchar=True, null_padding=True)")

print("--- SQL NULL Count ---")
null_addr = con.execute("SELECT COUNT(*) FROM tsv WHERE business_address IS NULL").fetchone()[0]
print(f"SQL NULL addresses: {null_addr}")

print("--- Literal '<NULL>' Count ---")
literal_bracket_null = con.execute("SELECT COUNT(*) FROM tsv WHERE business_address LIKE '%<NULL>%'").fetchone()[0]
print(f"Literal '<NULL>' addresses: {literal_bracket_null}")

print("--- Literal 'NULL' or 'null' (without brackets) Count ---")
literal_null = con.execute("SELECT COUNT(*) FROM tsv WHERE LOWER(business_address) LIKE '%null%' AND business_address NOT LIKE '%<NULL>%'").fetchone()[0]
print(f"Literal 'null' addresses: {literal_null}")

print("\n--- 5 Examples: SQL NULL ---")
ex1 = con.execute("SELECT business_address FROM tsv WHERE business_address IS NULL LIMIT 5").fetchall()
for e in ex1: print(repr(e[0]))

print("\n--- 5 Examples: Literal '<NULL>' ---")
ex2 = con.execute("SELECT business_address FROM tsv WHERE business_address LIKE '%<NULL>%' LIMIT 5").fetchall()
for e in ex2: print(repr(e[0]))

print("\n--- 5 Examples: Literal 'null' (no brackets) ---")
ex3 = con.execute("SELECT business_address FROM tsv WHERE LOWER(business_address) LIKE '%null%' AND business_address NOT LIKE '%<NULL>%' LIMIT 5").fetchall()
for e in ex3: print(repr(e[0]))

# Check business_name
print("\n--- Checking business_name ---")
name_sql_null = con.execute("SELECT COUNT(*) FROM tsv WHERE business_name IS NULL").fetchone()[0]
name_lit_bracket = con.execute("SELECT COUNT(*) FROM tsv WHERE business_name LIKE '%<NULL>%'").fetchone()[0]
name_lit_null = con.execute("SELECT COUNT(*) FROM tsv WHERE LOWER(business_name) LIKE '%null%' AND business_name NOT LIKE '%<NULL>%'").fetchone()[0]

print(f"SQL NULL names: {name_sql_null}")
print(f"Literal '<NULL>' names: {name_lit_bracket}")
print(f"Literal 'null' names: {name_lit_null}")
