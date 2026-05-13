"""Golden 80 record labels — manually labeled by planner BEFORE running v2 LLM.

Each entry: record_id → {difficulty, expected fields, optional alternatives, notes}.
Vocab terms come from outputs/vocab_v2.json. Pattern tags optional (can be []).
lesson_type: anti_pattern / discovery / best_practice (required, exactly 1).
matter_tags: 1-2 from Matter facet.
activity_tag: exactly 1 from Activity facet.
pattern_tags: 0-2 from Pattern facet.

To convert: `python tests/fixtures/golden_80_labels.py > tests/fixtures/golden_80.jsonl`
"""
from __future__ import annotations

import json
import sys

# Pre-loaded record metadata (title trimmed for reference; not part of label)
LABELS: list[dict] = [
    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 1: easy_anti_pattern (10 records)
    # Clear "must not / cannot / breaks" pattern, single dominant Matter
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_86ae84eb08bc",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "MemorySaver does not support cross-process resumption",
        "expected": {
            "matter_tags": ["langgraph_state"],
            "activity_tag": "designing",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "anti_pattern",
        },
        "notes": "in-memory checkpointer cannot persist; anti-pattern of expecting it to",
    },
    {
        "record_id": "knowledge_2e7cc17bada1",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "Structured output LLM calls must filter out empty-content messages",
        "expected": {
            "matter_tags": ["llm_agent_runtime"],
            "activity_tag": "integrating",
            "pattern_tags": ["early_validation"],
            "lesson_type": "anti_pattern",
        },
        "notes": "must filter empty AIMessages before with_structured_output",
    },
    {
        "record_id": "knowledge_240861da4612",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "Thread-local SQLite connections prevent InterfaceError in concurrent writes",
        "expected": {
            "matter_tags": ["persistence_db", "async_concurrency"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "anti_pattern",
        },
        "notes": "shared SQLite connection is wrong; thread-local mandatory",
    },
    {
        "record_id": "knowledge_9b63e87376a4",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "Date-based pagination requires explicit early-termination logic",
        "expected": {
            "matter_tags": ["http_api", "datetime_handling"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "anti_pattern",
        },
        "notes": "cannot rely on response emptiness or 404 for pagination termination",
    },
    {
        "record_id": "knowledge_afc8ba2c8764",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "OpenAI Realtime API requires official keys, cannot be proxied via OpenRouter",
        "expected": {
            "matter_tags": ["llm_agent_runtime", "config_env"],
            "activity_tag": "integrating",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "anti_pattern",
        },
        "notes": "cannot proxy OpenAI Realtime through third-party gateways",
    },
    {
        "record_id": "knowledge_df9cad451a82",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "Telegram bot message editing must throttle edits to avoid BadRequest",
        "expected": {
            "matter_tags": ["http_api"],
            "activity_tag": "integrating",
            "pattern_tags": ["fallback_strategy"],
            "lesson_type": "anti_pattern",
        },
        "notes": "Telegram rate-limits; client must throttle",
    },
    {
        "record_id": "knowledge_1971fbfb9e44",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "Astro catch-all routes are bypassed by direct .md page matches",
        "expected": {
            "matter_tags": ["frontend_browser", "filesystem_path"],
            "activity_tag": "debugging",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "anti_pattern",
        },
        "notes": "expecting catch-all to handle .md when file-based routing wins silently",
    },
    {
        "record_id": "knowledge_a188c8b023a4",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "Astro.glob is unavailable in getStaticPaths in Astro 6",
        "expected": {
            "matter_tags": ["frontend_browser", "build_deployment"],
            "activity_tag": "migrating",
            "pattern_tags": [],
            "lesson_type": "anti_pattern",
        },
        "notes": "API removed at build time in Astro 6; migration trap",
    },
    {
        "record_id": "knowledge_8e6008b967b8",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "LangGraph recursion limit prevents fallback logic from executing",
        "expected": {
            "matter_tags": ["langgraph_state"],
            "activity_tag": "debugging",
            "pattern_tags": ["fallback_strategy"],
            "lesson_type": "anti_pattern",
        },
        "notes": "try/except cannot catch recursion exhaustion before state update",
    },
    {
        "record_id": "knowledge_836290ec5638",
        "difficulty": "easy",
        "cell": "easy_anti_pattern",
        "_title": "HTTP client environment variables can break local service calls",
        "expected": {
            "matter_tags": ["http_api", "config_env"],
            "activity_tag": "debugging",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "anti_pattern",
        },
        "notes": "HTTP_PROXY env routes localhost through proxy and 503s",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 2: easy_discovery (10 records)
    # Surprising/non-obvious behaviors documented
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_067e16ed2a91",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "Database UPDATE operations must validate rowcount to detect silent no-op",
        "expected": {
            "matter_tags": ["persistence_db"],
            "activity_tag": "designing",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "discovery",
        },
        "notes": "0-row UPDATE silently succeeds; need cursor.rowcount check",
    },
    {
        "record_id": "knowledge_4ab96d78fa14",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "Subprocess environment inheritance requires explicit env propagation",
        "expected": {
            "matter_tags": ["config_env", "cli_argv"],
            "activity_tag": "integrating",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "notes": "MCP stdio servers don't inherit env; must pass explicitly",
    },
    {
        "record_id": "knowledge_bfc96a0648c1",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "Network-dependent data-fetching scripts fail silently without error handling",
        "expected": {
            "matter_tags": ["http_api", "cli_argv"],
            "activity_tag": "debugging",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "discovery",
        },
        "notes": "AkShare DNS/network failures exit silently",
    },
    {
        "record_id": "knowledge_5e2ea32dd051",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "Workflow health must be actively verified, not assumed",
        "expected": {
            "matter_tags": ["filesystem_path", "build_deployment"],
            "activity_tag": "testing",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "best_practice",
        },
        "notes": "skill drift detection via active healthcheck; 'must verify' tone is best_practice (codex audit 2026-05-13 caught this)",
    },
    {
        "record_id": "knowledge_43ad81c39896",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "Date filtering for pandas DataFrames must match string representation",
        "expected": {
            "matter_tags": ["persistence_db", "datetime_handling"],
            "activity_tag": "debugging",
            "pattern_tags": [],
            "lesson_type": "discovery",
        },
        "notes": "DataFrame date column was string, not datetime",
    },
    {
        "record_id": "knowledge_93c5cf0577dd",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "CDP window management requires context-level session, not page-level",
        "expected": {
            "matter_tags": ["frontend_browser"],
            "activity_tag": "debugging",
            "pattern_tags": ["abstraction_leak"],
            "lesson_type": "discovery",
        },
        "notes": "page.new_cdp_session silently fails for Browser-level commands",
    },
    {
        "record_id": "knowledge_739adf87836d",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "TypedDict.get() does not provide safe fallbacks",
        "expected": {
            "matter_tags": ["langgraph_state"],
            "activity_tag": "debugging",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "discovery",
        },
        "notes": "TypedDict.get returns None ignoring default arg",
    },
    {
        "record_id": "knowledge_ae0c2d324146",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "String-based monkeypatch targets must be updated alongside import paths",
        "expected": {
            "matter_tags": ["testing_framework"],
            "activity_tag": "refactoring",
            "pattern_tags": ["single_source_of_truth"],
            "lesson_type": "discovery",
        },
        "notes": "string-path monkeypatch invisible to import refactor tools",
    },
    {
        "record_id": "knowledge_c621ab8a4bbe",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "Decorator registration requires explicit import triggering",
        "expected": {
            "matter_tags": ["package_dependency"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "notes": "@register relies on import side effects, must be triggered",
    },
    {
        "record_id": "knowledge_6f993dbb7177",
        "difficulty": "easy",
        "cell": "easy_discovery",
        "_title": "Astro layout validation requires explicit type alignment",
        "expected": {
            "matter_tags": ["frontend_browser", "json_serialization"],
            "activity_tag": "debugging",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "notes": "string vs Date prop type drift in Astro layouts",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 3: easy_best_practice (10 records)
    # Recommended approaches with clear "should / prefer" tone
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_d226b615e6ba",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "Test isolation using FakeEmbeddings with query capture",
        "expected": {
            "matter_tags": ["testing_framework", "llm_agent_runtime"],
            "activity_tag": "testing",
            "pattern_tags": ["separation_of_concerns"],
            "lesson_type": "best_practice",
        },
        "notes": "FakeEmbeddings preferred over mocking LLM/Chroma layers",
    },
    {
        "record_id": "knowledge_eabeb97d43d7",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "Atomic write + JSON serialization must be reused, not reimplemented",
        "expected": {
            "matter_tags": ["filesystem_path", "json_serialization"],
            "activity_tag": "refactoring",
            "pattern_tags": ["single_source_of_truth"],
            "lesson_type": "best_practice",
        },
        "notes": "reuse atomic_write_json helper, don't duplicate",
    },
    {
        "record_id": "knowledge_793936000408",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "macOS default directories should be treated as immutable system interfaces",
        "expected": {
            "matter_tags": ["filesystem_path"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "best_practice",
        },
        "notes": "macOS system dirs are contracts, not generic storage",
    },
    {
        "record_id": "knowledge_686df7302848",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "Budget initialization must be idempotent and immutable after first use",
        "expected": {
            "matter_tags": ["langgraph_state"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract", "single_source_of_truth"],
            "lesson_type": "best_practice",
        },
        "notes": "budget locked after init; subsequent calls only consume",
    },
    {
        "record_id": "knowledge_e0274d24636c",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "Utility class naming should reflect visual intent, not layout role",
        "expected": {
            "matter_tags": ["frontend_browser"],
            "activity_tag": "designing",
            "pattern_tags": ["separation_of_concerns"],
            "lesson_type": "best_practice",
        },
        "notes": "CSS utility named by visual intent, not role overload",
    },
    {
        "record_id": "knowledge_f8ae1ffc2053",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "Conditional field inclusion in JSON persistence",
        "expected": {
            "matter_tags": ["json_serialization"],
            "activity_tag": "refactoring",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "best_practice",
        },
        "notes": "key-presence check beats null-fill for optional fields",
    },
    {
        "record_id": "knowledge_152007abdc02",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "Orchestration-layer source dispatching preserves graph isolation",
        "expected": {
            "matter_tags": ["cli_argv", "langgraph_state"],
            "activity_tag": "designing",
            "pattern_tags": ["separation_of_concerns"],
            "lesson_type": "best_practice",
        },
        "notes": "multi-source handled at orchestration layer, not coupled in graph",
    },
    {
        "record_id": "knowledge_711fca268c5f",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "Node-level failure should not cascade to field-level data loss",
        "expected": {
            "matter_tags": ["langgraph_state"],
            "activity_tag": "designing",
            "pattern_tags": ["fallback_strategy"],
            "lesson_type": "best_practice",
        },
        "notes": "field-granular fallback (None+warning) preserves valid fields",
    },
    {
        "record_id": "knowledge_4303a5837faf",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "Storage state injection should be conditional on validity",
        "expected": {
            "matter_tags": ["frontend_browser"],
            "activity_tag": "designing",
            "pattern_tags": ["early_validation"],
            "lesson_type": "best_practice",
        },
        "notes": "validate cookie storage before injecting into Playwright context",
    },
    {
        "record_id": "knowledge_c84d6b048d16",
        "difficulty": "easy",
        "cell": "easy_best_practice",
        "_title": "CLI command dispatch should separate sync and async execution paths",
        "expected": {
            "matter_tags": ["cli_argv", "async_concurrency"],
            "activity_tag": "designing",
            "pattern_tags": ["separation_of_concerns"],
            "lesson_type": "best_practice",
        },
        "notes": "bifurcate CLI entry: don't asyncio.run() everything",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 4: medium_cross_matter (10 records)
    # Records spanning 2-3 matter domains — picking 1-2 best matter tags requires judgment
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_ba3d4d3908bc",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "Directory creation must be explicit before file write in CLI doc workflow",
        "expected": {
            "matter_tags": ["filesystem_path", "cli_argv"],
            "activity_tag": "documenting",
            "pattern_tags": ["early_validation"],
            "lesson_type": "anti_pattern",
        },
        "notes": "matter primary=fs, secondary=cli (shell context); not deployment",
    },
    {
        "record_id": "knowledge_4f6f03d0a68d",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "EnsembleRetriever import path varies by LangChain version",
        "expected": {
            "matter_tags": ["package_dependency", "llm_agent_runtime"],
            "activity_tag": "integrating",
            "pattern_tags": ["fallback_strategy"],
            "lesson_type": "anti_pattern",
        },
        "notes": "LangChain dependency version + LLM ecosystem; not raw fs",
    },
    {
        "record_id": "knowledge_99e105aa1c2b",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "Frontend fuzzy search should dynamically select data source",
        "expected": {
            "matter_tags": ["frontend_browser", "persistence_db"],
            "activity_tag": "designing",
            "pattern_tags": ["fallback_strategy"],
            "lesson_type": "best_practice",
        },
        "notes": "frontend UX + db source selection; async incidental",
    },
    {
        "record_id": "knowledge_3748828c8653",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "Async wrapper for blocking I/O calls in asyncio",
        "expected": {
            "matter_tags": ["async_concurrency", "llm_agent_runtime"],
            "activity_tag": "integrating",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "best_practice",
        },
        "notes": "asyncio.to_thread for blocking SDK calls (DashScope sync)",
    },
    {
        "record_id": "knowledge_a84a710b3853",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "BS4 absence must be handled at import-time, not call-time",
        "expected": {
            "matter_tags": ["package_dependency", "frontend_browser"],
            "activity_tag": "designing",
            "pattern_tags": ["fallback_strategy", "early_validation"],
            "lesson_type": "anti_pattern",
        },
        "notes": "BeautifulSoup is HTML parsing → frontend; deps is the dependency mgmt aspect",
    },
    {
        "record_id": "knowledge_5a6a0ae41cd1",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "Hot-path schema validation should be avoided in CRUD methods",
        "expected": {
            "matter_tags": ["persistence_db", "llm_agent_runtime"],
            "activity_tag": "refactoring",
            "pattern_tags": ["early_validation"],
            "lesson_type": "best_practice",
        },
        "notes": "embedding CRUD on db; LLM embedding is the use context",
    },
    {
        "record_id": "knowledge_c173015685a4",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "Unit test failures after schema removal indicate test coupling to implementation",
        "expected": {
            "matter_tags": ["testing_framework", "persistence_db"],
            "activity_tag": "refactoring",
            "pattern_tags": ["separation_of_concerns", "explicit_contract"],
            "lesson_type": "anti_pattern",
        },
        "notes": "test framework + schema; llm context incidental",
    },
    {
        "record_id": "knowledge_97049ae940da",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "Thread-id–scoped checkpoint isolation must handle config drift explicitly",
        "expected": {
            "matter_tags": ["langgraph_state", "config_env"],
            "activity_tag": "designing",
            "pattern_tags": ["early_validation"],
            "lesson_type": "anti_pattern",
        },
        "notes": "LangGraph checkpoint + config drift; async is incidental",
    },
    {
        "record_id": "knowledge_a0d5e1b62028",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "Payload structure consistency across error paths required",
        "expected": {
            "matter_tags": ["http_api", "json_serialization"],
            "activity_tag": "refactoring",
            "pattern_tags": ["single_source_of_truth", "explicit_contract"],
            "lesson_type": "anti_pattern",
        },
        "notes": "HTTP response shapes; not log-as-matter (log is the symptom, not subject)",
    },
    {
        "record_id": "knowledge_33f4612ccb4a",
        "difficulty": "medium",
        "cell": "medium_cross_matter",
        "_title": "Frontend overlay visibility should be driven by structured log patterns",
        "expected": {
            "matter_tags": ["frontend_browser", "logging_observability"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "best_practice",
        },
        "notes": "frontend UI driven by structured logs; cross-matter genuine",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 5: medium_activity_ambiguous (10 records)
    # Activity tag could plausibly be 2 different values — picking one is judgment
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_8861aca32288",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "pytest -m e2e reports deselected tests, not skipped ones",
        "expected": {
            "matter_tags": ["testing_framework"],
            "activity_tag": "testing",
            "pattern_tags": [],
            "lesson_type": "discovery",
        },
        "acceptable_activity_alternatives": ["debugging"],
        "notes": "could be testing (in-test context) or debugging (investigation)",
    },
    {
        "record_id": "knowledge_a3e08dadd3f0",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "TDD mandates test file tracking",
        "expected": {
            "matter_tags": ["testing_framework", "git_vcs"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "best_practice",
        },
        "acceptable_activity_alternatives": ["testing", "documenting"],
        "notes": "TDD policy is a design decision; could be argued as testing or documenting policy",
    },
    {
        "record_id": "knowledge_e6d4b82c0601",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "DerivedView must be used alongside existing hydrate_meta_data",
        "expected": {
            "matter_tags": ["persistence_db"],
            "activity_tag": "migrating",
            "pattern_tags": ["fallback_strategy"],
            "lesson_type": "best_practice",
        },
        "acceptable_activity_alternatives": ["refactoring", "designing"],
        "notes": "coexisting layer migration; could also be refactoring",
    },
    {
        "record_id": "knowledge_7268f1490c02",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "Delayed akshare import avoids hard dependency at module load time",
        "expected": {
            "matter_tags": ["package_dependency"],
            "activity_tag": "designing",
            "pattern_tags": ["fallback_strategy", "early_validation"],
            "lesson_type": "best_practice",
        },
        "acceptable_activity_alternatives": ["configuring", "refactoring"],
        "notes": "design choice for dependency handling",
    },
    {
        "record_id": "knowledge_cea3af2b6472",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "Use ls -la to detect symbolic links",
        "expected": {
            "matter_tags": ["filesystem_path"],
            "activity_tag": "debugging",
            "pattern_tags": [],
            "lesson_type": "discovery",
        },
        "acceptable_activity_alternatives": ["documenting"],
        "notes": "lookup-trick lesson; could be doc or debug",
    },
    {
        "record_id": "knowledge_4701efff35c6",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "pytest testpaths overrides default collection",
        "expected": {
            "matter_tags": ["testing_framework"],
            "activity_tag": "configuring",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "acceptable_activity_alternatives": ["testing"],
        "notes": "configuring pytest behavior",
    },
    {
        "record_id": "knowledge_3fa3761176fc",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "Contributor guides must be grounded in observed tooling and structure",
        "expected": {
            "matter_tags": ["filesystem_path"],
            "activity_tag": "documenting",
            "pattern_tags": ["single_source_of_truth", "explicit_contract"],
            "lesson_type": "best_practice",
        },
        "acceptable_activity_alternatives": ["designing"],
        "notes": "doc-authoring activity; primary matter is project files inspected",
    },
    {
        "record_id": "knowledge_b3d064070c44",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "Backend field filtering requires explicit model/schema changes",
        "expected": {
            "matter_tags": ["http_api", "json_serialization"],
            "activity_tag": "debugging",
            "pattern_tags": ["silent_failure", "explicit_contract"],
            "lesson_type": "discovery",
        },
        "acceptable_activity_alternatives": ["integrating"],
        "notes": "discovered while debugging REST API; could also frame as integrating",
    },
    {
        "record_id": "knowledge_afec634394be",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "Pytest plugin auto-loading can silently break integration tests",
        "expected": {
            "matter_tags": ["testing_framework", "package_dependency"],
            "activity_tag": "configuring",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "discovery",
        },
        "acceptable_activity_alternatives": ["debugging", "testing"],
        "notes": "PYTEST_DISABLE_PLUGIN_AUTOLOAD is configuring; surfaced during testing/debug",
    },
    {
        "record_id": "knowledge_03b5bb094d8b",
        "difficulty": "medium",
        "cell": "medium_activity_ambiguous",
        "_title": "Prompt key removal requires explicit code-path validation",
        "expected": {
            "matter_tags": ["llm_agent_runtime"],
            "activity_tag": "refactoring",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "anti_pattern",
        },
        "acceptable_activity_alternatives": ["migrating"],
        "notes": "prompt YAML cleanup is refactoring or migration",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 6: medium_pattern_boundary (10 records)
    # Pattern tag boundary case — multiple plausible patterns
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_daa56723cdd4",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "Mock boundary dictates function visibility",
        "expected": {
            "matter_tags": ["testing_framework"],
            "activity_tag": "testing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "acceptable_pattern_alternatives": [["separation_of_concerns"], ["silent_failure"]],
        "notes": "mock visibility requires explicit module-level; boundary between explicit_contract and separation_of_concerns",
    },
    {
        "record_id": "knowledge_417e4e91d984",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "Conventional Commits should reflect behavioral impact, not just file changes",
        "expected": {
            "matter_tags": ["git_vcs"],
            "activity_tag": "documenting",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "best_practice",
        },
        "acceptable_pattern_alternatives": [["single_source_of_truth"]],
        "notes": "commit message is contract; could be SSoT of intent",
    },
    {
        "record_id": "knowledge_ec46b50ae423",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "Browser-side fetch requires explicit timeout via AbortController",
        "expected": {
            "matter_tags": ["frontend_browser", "http_api"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract", "fallback_strategy"],
            "lesson_type": "best_practice",
        },
        "notes": "explicit timeout + fallback; both apply",
    },
    {
        "record_id": "knowledge_30b62543436d",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "Astro static site deployment requires explicit output directory",
        "expected": {
            "matter_tags": ["build_deployment", "config_env"],
            "activity_tag": "deploying",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "acceptable_pattern_alternatives": [["fallback_strategy"]],
        "notes": "Vercel config not browser rendering; matter[1]=config_env (codex audit 2026-05-13)",
    },
    {
        "record_id": "knowledge_9a2e43794997",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "pytest does not auto-load .env files",
        "expected": {
            "matter_tags": ["testing_framework", "config_env"],
            "activity_tag": "configuring",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "discovery",
        },
        "acceptable_pattern_alternatives": [["explicit_contract"]],
        "notes": "silent_failure or explicit_contract — boundary case",
    },
    {
        "record_id": "knowledge_3249f7b6b663",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "Session normalization should treat non-string source values as skip candidates",
        "expected": {
            "matter_tags": ["json_serialization"],
            "activity_tag": "designing",
            "pattern_tags": ["early_validation"],
            "lesson_type": "best_practice",
        },
        "acceptable_pattern_alternatives": [["separation_of_concerns"]],
        "notes": "validation OR separation between pipeline stages",
    },
    {
        "record_id": "knowledge_ec2468c3c7f6",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "File-writing tests must verify both filesystem presence and content",
        "expected": {
            "matter_tags": ["testing_framework", "filesystem_path"],
            "activity_tag": "testing",
            "pattern_tags": ["early_validation"],
            "lesson_type": "best_practice",
        },
        "acceptable_pattern_alternatives": [["silent_failure"]],
        "notes": "test goes beyond presence; validation vs silent_failure both apply",
    },
    {
        "record_id": "knowledge_6ceec7eabf6d",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "Recall@5 evaluation requires consistent top-k across strategies",
        "expected": {
            "matter_tags": ["testing_framework"],
            "activity_tag": "testing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "best_practice",
        },
        "acceptable_pattern_alternatives": [["single_source_of_truth"]],
        "notes": "evaluation methodology requires explicit param fix",
    },
    {
        "record_id": "knowledge_f32572eca388",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "Strategy composition via shared intermediate outputs",
        "expected": {
            "matter_tags": ["llm_agent_runtime"],
            "activity_tag": "refactoring",
            "pattern_tags": ["single_source_of_truth"],
            "lesson_type": "best_practice",
        },
        "acceptable_pattern_alternatives": [["separation_of_concerns"]],
        "notes": "shared cache for cross-strategy; ssot or separation",
    },
    {
        "record_id": "knowledge_b500217e9b90",
        "difficulty": "medium",
        "cell": "medium_pattern_boundary",
        "_title": "Batch event scanning is necessary to avoid O((N+M)×E) complexity",
        "expected": {
            "matter_tags": ["persistence_db"],
            "activity_tag": "refactoring",
            "pattern_tags": ["separation_of_concerns"],
            "lesson_type": "anti_pattern",
        },
        "acceptable_pattern_alternatives": [["single_source_of_truth"]],
        "notes": "index reuse anti-pattern; separation of indexing concern",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 7: hard_v1_orphan (10 records)
    # Records that were "missing" in v1 vocab. Test whether v2 catches them.
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_b9d08f3d6dd9",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Quality thresholds are heuristic defaults, not scientific constants",
        "expected": {
            "matter_tags": ["testing_framework"],
            "activity_tag": "documenting",
            "pattern_tags": [],
            "lesson_type": "discovery",
        },
        "notes": "v1 orphan; matter is weak — likely still hard for v2 (general epistemics)",
    },
    {
        "record_id": "knowledge_b1157c951433",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Plugin systems can be purely declarative",
        "expected": {
            "matter_tags": ["package_dependency", "json_serialization"],
            "activity_tag": "designing",
            "pattern_tags": ["separation_of_concerns"],
            "lesson_type": "discovery",
        },
        "notes": "v1 orphan; declarative plugin = config-driven dispatch",
    },
    {
        "record_id": "knowledge_cc9a3c96843c",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Fallback streaming for HTTP backends should yield full response as single token",
        "expected": {
            "matter_tags": ["http_api", "llm_agent_runtime"],
            "activity_tag": "integrating",
            "pattern_tags": ["fallback_strategy"],
            "lesson_type": "best_practice",
        },
        "acceptable_matter_alternatives": [["http_api", "async_concurrency"]],
        "notes": "v1 orphan; non-streaming fallback. AsyncIterator framing also valid (codex audit 2026-05-13)",
    },
    {
        "record_id": "knowledge_e18f99cfe785",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Flood control limits Telegram's perceived streaming fidelity",
        "expected": {
            "matter_tags": ["http_api"],
            "activity_tag": "integrating",
            "pattern_tags": ["abstraction_leak"],
            "lesson_type": "discovery",
        },
        "notes": "v1 orphan; bot API limits leak through abstraction",
    },
    {
        "record_id": "knowledge_8ccb4089dcb8",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Testing streaming behavior requires mocking at HTTP transport layer",
        "expected": {
            "matter_tags": ["testing_framework", "http_api"],
            "activity_tag": "testing",
            "pattern_tags": ["separation_of_concerns"],
            "lesson_type": "best_practice",
        },
        "notes": "v1 orphan; transport-layer mocking concept",
    },
    {
        "record_id": "knowledge_99da17277019",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Git ignore rules only apply to untracked files",
        "expected": {
            "matter_tags": ["git_vcs"],
            "activity_tag": "debugging",
            "pattern_tags": [],
            "lesson_type": "discovery",
        },
        "notes": "v1 orphan; pure git semantic discovery",
    },
    {
        "record_id": "knowledge_74d09754be37",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Conventional Commits require scope disambiguation for multi-domain features",
        "expected": {
            "matter_tags": ["git_vcs"],
            "activity_tag": "documenting",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "best_practice",
        },
        "notes": "v1 orphan; commit hygiene concept",
    },
    {
        "record_id": "knowledge_3c26c6ee7ef0",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "WAF detection can occur without visible UI",
        "expected": {
            "matter_tags": ["frontend_browser", "http_api"],
            "activity_tag": "debugging",
            "pattern_tags": ["silent_failure"],
            "lesson_type": "anti_pattern",
        },
        "notes": "v1 orphan; behavioral WAF triggers without DOM signals",
    },
    {
        "record_id": "knowledge_40ef56b2da6f",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Test framework choice dictates validation strategy",
        "expected": {
            "matter_tags": ["testing_framework", "async_concurrency"],
            "activity_tag": "designing",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "anti_pattern",
        },
        "notes": "v1 orphan; pytest vs unittest interaction with async",
    },
    {
        "record_id": "knowledge_26fe66c12d70",
        "difficulty": "hard",
        "cell": "hard_v1_orphan",
        "_title": "Fallback to symbol-as-name is pragmatic heuristic for company name resolution",
        "expected": {
            "matter_tags": ["http_api"],
            "activity_tag": "designing",
            "pattern_tags": ["fallback_strategy"],
            "lesson_type": "best_practice",
        },
        "notes": "v1 orphan; graceful degradation pattern",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 8: hard_multi_matter (5 records)
    # 3+ tech domains; picking which 1-2 are PRIMARY is genuinely judgment
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_1765bb86b8b8",
        "difficulty": "hard",
        "cell": "hard_multi_matter",
        "_title": "Environment variables require explicit loading in uv run subprocesses",
        "expected": {
            "matter_tags": ["config_env", "package_dependency"],
            "activity_tag": "configuring",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "acceptable_matter_alternatives": [["config_env", "cli_argv"]],
        "notes": "primary=config_env (env vars), secondary=deps (uv); cli is incidental",
    },
    {
        "record_id": "knowledge_27200befe33e",
        "difficulty": "hard",
        "cell": "hard_multi_matter",
        "_title": "Git status ' D' means working-tree deletion only",
        "expected": {
            "matter_tags": ["git_vcs"],
            "activity_tag": "debugging",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "notes": "git is the primary subject; fs is just the deleted file",
    },
    {
        "record_id": "knowledge_945136fcca2c",
        "difficulty": "hard",
        "cell": "hard_multi_matter",
        "_title": "MCP servers can be managed via CLI without editing config files directly",
        "expected": {
            "matter_tags": ["cli_argv", "config_env"],
            "activity_tag": "configuring",
            "pattern_tags": [],
            "lesson_type": "best_practice",
        },
        "acceptable_matter_alternatives": [["cli_argv", "llm_agent_runtime"]],
        "notes": "claude mcp CLI subcommand; LLM context but config mgmt focus",
    },
    {
        "record_id": "knowledge_8b834930060e",
        "difficulty": "hard",
        "cell": "hard_multi_matter",
        "_title": "Async data loading in frontend init should be non-blocking and error-resilient",
        "expected": {
            "matter_tags": ["frontend_browser", "async_concurrency"],
            "activity_tag": "designing",
            "pattern_tags": ["fallback_strategy", "early_validation"],
            "lesson_type": "best_practice",
        },
        "notes": "frontend init pattern; async is necessary attribute",
    },
    {
        "record_id": "knowledge_19e24948d9a3",
        "difficulty": "hard",
        "cell": "hard_multi_matter",
        "_title": "Toolchain inference requires cross-referencing multiple config and source files",
        "expected": {
            "matter_tags": ["package_dependency", "filesystem_path"],
            "activity_tag": "documenting",
            "pattern_tags": ["single_source_of_truth"],
            "lesson_type": "best_practice",
        },
        "acceptable_matter_alternatives": [["package_dependency", "cli_argv"]],
        "notes": "deps cross-checked across files; documenting tooling",
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CELL 9: hard_lesson_type_ambiguous (5 records)
    # Multiple lesson_type signals fire; picking one is judgment
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "record_id": "knowledge_86b61bd8c976",
        "difficulty": "hard",
        "cell": "hard_lesson_type_ambiguous",
        "_title": "Git remote origin must be unique per hostname to avoid routing conflicts",
        "expected": {
            "matter_tags": ["git_vcs"],
            "activity_tag": "configuring",
            "pattern_tags": ["single_source_of_truth"],
            "lesson_type": "anti_pattern",
        },
        "acceptable_lesson_type_alternatives": ["best_practice"],
        "notes": "multiple A records is anti, single = best practice; primary tone is must_not",
    },
    {
        "record_id": "knowledge_45158126087b",
        "difficulty": "hard",
        "cell": "hard_lesson_type_ambiguous",
        "_title": "Gap detection in reflective search agents relies on shallow heuristic signals",
        "expected": {
            "matter_tags": ["llm_agent_runtime"],
            "activity_tag": "designing",
            "pattern_tags": ["early_validation"],
            "lesson_type": "discovery",
        },
        "acceptable_lesson_type_alternatives": ["anti_pattern"],
        "notes": "discovery of brittleness with caveat tone — could be anti-pattern framing",
    },
    {
        "record_id": "knowledge_2c393cca469e",
        "difficulty": "hard",
        "cell": "hard_lesson_type_ambiguous",
        "_title": "Integration tests should accept flexible status indicators",
        "expected": {
            "matter_tags": ["testing_framework", "json_serialization"],
            "activity_tag": "testing",
            "pattern_tags": ["early_validation"],
            "lesson_type": "best_practice",
        },
        "acceptable_lesson_type_alternatives": ["anti_pattern"],
        "notes": "hardcoding numeric status is anti, flexible is best — record framed as best",
    },
    {
        "record_id": "knowledge_eafec4141dc2",
        "difficulty": "hard",
        "cell": "hard_lesson_type_ambiguous",
        "_title": "Defaulting to force-update requires explicit overwriting logic, not just flag removal",
        "expected": {
            "matter_tags": ["git_vcs"],
            "activity_tag": "refactoring",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "anti_pattern",
        },
        "acceptable_lesson_type_alternatives": ["discovery"],
        "notes": "deleting flag without rewriting merge logic is anti-pattern; discovery of the gap",
    },
    {
        "record_id": "knowledge_f1c6f055f700",
        "difficulty": "hard",
        "cell": "hard_lesson_type_ambiguous",
        "_title": "Strategy behavior requires explicit business rule documentation beyond I/O contracts",
        "expected": {
            "matter_tags": ["llm_agent_runtime"],
            "activity_tag": "documenting",
            "pattern_tags": ["explicit_contract"],
            "lesson_type": "discovery",
        },
        "acceptable_lesson_type_alternatives": ["best_practice"],
        "notes": "discovery of LLM behavior gap; could frame as best_practice",
    },
]


def main():
    for entry in LABELS:
        # Strip private _title from output
        out = {k: v for k, v in entry.items() if not k.startswith("_")}
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
