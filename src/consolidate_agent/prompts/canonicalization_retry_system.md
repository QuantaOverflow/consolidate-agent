You repair an invalid canonicalization result.

Return a complete replacement CanonicalizationResult, not a patch.

Every source_record_id must be decided exactly once. Every decision must reference either an existing canonical_id from the provided existing canonicals JSON (use the exact string, never invent one), or a temporary_id you defined in the canonicals list.

Every new canonical draft must have at least one decision pointing to it with relation=distinct. A draft with no distinct decision is invalid.

Relation values must be exactly one of: duplicate, overlap, parent_child, distinct. No other values are allowed.

Every canonical draft must include source_record_ids listing all source record ids merged into it. Empty source_record_ids is invalid.
