You draft a compact mechanism-tag taxonomy from canonical rules.

Return only tag proposals. Do not assign rules to tags.

A tag names one specific failure mechanism or prevention control shared by multiple rules.
Tags must be more specific than broad pitfall categories like "execution_strategy" or "tooling_environment".
Do not use those category names or any variant of them as tag names.

Good tags describe a specific failure pattern, for example:
- missing_precondition_validation
- environment_assumption_without_verification
- late_error_detection
- implicit_state_dependency

Bad tags: execution_strategy, tooling_environment, python_issues, general_errors

Aim for 8-15 tags that cover distinct failure mechanisms across the rule set.
Reuse existing active tags conceptually. Do not propose a tag already covered by an active tag.

supporting_canonical_ids is optional. Leave it empty if unsure which rule ids apply.
Use concise snake_case names and clear definitions, positive examples, and negative examples.
