# ADR 0002: A small dbt-style runner on DuckDB instead of dbt itself

- Status: accepted
- Date: 2026-11-19

## Context

The analytics layer should look like what a GTM/analytics engineer ships: layered SQL
models (staging -> intermediate -> marts), declared dependencies and data tests.
dbt-core + dbt-duckdb would do this but adds a heavy dependency tree, a profiles file
and a second CLI for a project whose SQL fits in ~20 files, and it would hide the DAG
logic that this project wants to show.

## Decision

Write a ~250-line runner (`gal.warehouse.runner`) that keeps dbt's conventions so the SQL
is portable to dbt unchanged:

- folder = schema; file name = model name;
- `{{ ref('x') }}`, `{{ source('raw', 't') }}`, `{{ var('k') }}` Jinja-like placeholders,
  resolved with regular expressions (no Jinja dependency);
- `-- materialized: view|table` header;
- deterministic Kahn topological sort with cycle and unknown-ref detection;
- `schema.yml` tests: `not_null`, `unique`, `accepted_values`, `relationships`,
  `expression_is_true`, each compiled to a "count failing rows" query.

A layering test asserts staging models read only sources and marts never read raw data.

## Consequences

- Zero extra dependencies; the full build of 21 models + 59 tests runs in well under a
  second on DuckDB.
- No Jinja control flow (loops/macros). If the project outgrows that, the SQL and
  `schema.yml` can move to dbt-duckdb with only the `-- materialized` header changing.
