import duckdb
sql = "SELECT REGEXP_REPLACE('Hello, राम! 123+', '[[:punct:][:cntrl:]]+', ' ', 'g')"
print(duckdb.query(sql).fetchone()[0])
