import duckdb
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

con = duckdb.connect()
con.execute("CREATE VIEW tsv AS SELECT * FROM read_csv('train_source2.tsv', sep='\\t', header=True, all_varchar=True, null_padding=True)")

print("\n--- Checking business_name ---")
name_sql_null = con.execute("SELECT COUNT(*) FROM tsv WHERE business_name IS NULL").fetchone()[0]
name_lit_bracket = con.execute("SELECT COUNT(*) FROM tsv WHERE business_name LIKE '%<NULL>%'").fetchone()[0]
name_lit_null = con.execute("SELECT COUNT(*) FROM tsv WHERE LOWER(business_name) LIKE '%null%' AND business_name NOT LIKE '%<NULL>%'").fetchone()[0]

print(f"SQL NULL names: {name_sql_null}")
print(f"Literal '<NULL>' names: {name_lit_bracket}")
print(f"Literal 'null' names (no brackets): {name_lit_null}")

print("\n--- 5 Examples: Literal '<NULL>' ---")
ex2 = con.execute("SELECT business_name FROM tsv WHERE business_name LIKE '%<NULL>%' LIMIT 5").fetchall()
for e in ex2: print(repr(e[0]))

print("\n--- 5 Examples: Literal 'null' (no brackets) ---")
ex3 = con.execute("SELECT business_name FROM tsv WHERE LOWER(business_name) LIKE '%null%' AND business_name NOT LIKE '%<NULL>%' LIMIT 5").fetchall()
for e in ex3: print(repr(e[0]))
