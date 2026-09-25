import duckdb
con = duckdb.connect()

for src in ['source2_test.parquet', 'source3_test.parquet']:
    print(f"Stats for {src}:")
    con.execute(f"CREATE OR REPLACE VIEW res AS SELECT * FROM read_parquet('intermediate/{src}')")
    c1 = con.execute("SELECT COUNT(*) FROM res WHERE name_clean = '' AND business_name != ''").fetchone()[0]
    c2 = con.execute("SELECT COUNT(*) FROM res WHERE address_clean = '' AND business_address != ''").fetchone()[0]
    print(f'  Empty names: {c1}, Empty addrs: {c2}')
