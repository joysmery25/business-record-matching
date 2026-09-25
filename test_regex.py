import duckdb
sql = "SELECT REGEXP_REPLACE('Hello, राम! 123', '[^\\\\p{L}\\\\p{N}]+', ' ', 'g')"
print(duckdb.query(sql).fetchone()[0])
