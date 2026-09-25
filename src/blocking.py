def get_blocking_queries(s1_table, s2_table, source_name):
    queries = []
    
    # 1. Exact Name + Country
    queries.append(("exact_name_country", f"""
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, '{source_name}' as candidate_source, 'exact_name_country' as blocking_strategy
        FROM {s1_table} s1
        INNER JOIN {s2_table} s2 ON s1.name_compact = s2.name_compact AND s1.country_clean = s2.country_clean
        WHERE length(s1.name_compact) >= 2
    """))

    # 2. Exact Name
    queries.append(("exact_name", f"""
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, '{source_name}' as candidate_source, 'exact_name' as blocking_strategy
        FROM {s1_table} s1
        INNER JOIN {s2_table} s2 ON s1.name_compact = s2.name_compact
        WHERE length(s1.name_compact) >= 5
    """))

    # 3. Name Prefix (first 8 chars) + Country
    queries.append(("name_prefix8_country", f"""
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, '{source_name}' as candidate_source, 'name_prefix8_country' as blocking_strategy
        FROM {s1_table} s1
        INNER JOIN {s2_table} s2 
            ON substring(s1.name_compact, 1, 8) = substring(s2.name_compact, 1, 8) 
            AND s1.country_clean = s2.country_clean
        WHERE length(s1.name_compact) >= 8 AND length(s2.name_compact) >= 8
    """))

    # 4. First Word of Name + Country (Token 1)
    queries.append(("name_first_word_country", f"""
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, '{source_name}' as candidate_source, 'name_first_word_country' as blocking_strategy
        FROM {s1_table} s1
        INNER JOIN {s2_table} s2 
            ON split_part(s1.name_clean, ' ', 1) = split_part(s2.name_clean, ' ', 1) 
            AND s1.country_clean = s2.country_clean
        WHERE length(split_part(s1.name_clean, ' ', 1)) >= 5
    """))

    # 5. Postal/PIN Code (5 or 6 digits) + Name Prefix (4 chars)
    queries.append(("postal_name_prefix4", f"""
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, '{source_name}' as candidate_source, 'postal_name_prefix4' as blocking_strategy
        FROM {s1_table} s1
        INNER JOIN {s2_table} s2 
            ON substring(s1.name_compact, 1, 4) = substring(s2.name_compact, 1, 4)
            AND regexp_extract(s1.address_clean, '\\b\\d{{5,6}}\\b', 0) = regexp_extract(s2.address_clean, '\\b\\d{{5,6}}\\b', 0)
        WHERE regexp_extract(s1.address_clean, '\\b\\d{{5,6}}\\b', 0) != ''
          AND length(s1.name_compact) >= 4
    """))

    # 6. Exact Address + Country
    queries.append(("exact_address_country", f"""
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, '{source_name}' as candidate_source, 'exact_address_country' as blocking_strategy
        FROM {s1_table} s1
        INNER JOIN {s2_table} s2 ON s1.address_compact = s2.address_compact AND s1.country_clean = s2.country_clean
        WHERE length(s1.address_compact) >= 8
    """))

    # 7. House Number + Name Prefix (5 chars)
    queries.append(("house_num_name_prefix5", f"""
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, '{source_name}' as candidate_source, 'house_num_name_prefix5' as blocking_strategy
        FROM {s1_table} s1
        INNER JOIN {s2_table} s2 
            ON substring(s1.name_compact, 1, 5) = substring(s2.name_compact, 1, 5)
            AND regexp_extract(s1.address_clean, '^\\d+', 0) = regexp_extract(s2.address_clean, '^\\d+', 0)
        WHERE regexp_extract(s1.address_clean, '^\\d+', 0) != ''
          AND length(s1.name_compact) >= 5
    """))

    # 8. First Word Name + First Word Address
    queries.append(("first_word_name_address", f"""
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, '{source_name}' as candidate_source, 'first_word_name_address' as blocking_strategy
        FROM {s1_table} s1
        INNER JOIN {s2_table} s2 
            ON split_part(s1.name_clean, ' ', 1) = split_part(s2.name_clean, ' ', 1) 
            AND split_part(s1.address_clean, ' ', 1) = split_part(s2.address_clean, ' ', 1)
        WHERE length(split_part(s1.name_clean, ' ', 1)) >= 4
          AND length(split_part(s1.address_clean, ' ', 1)) >= 4
    """))

    return queries
