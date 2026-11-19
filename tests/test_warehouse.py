from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from gal.config import SimConfig
from gal.pipeline import PipelineRun
from gal.warehouse.build import DataTestError, build_warehouse, default_models_dir
from gal.warehouse.runner import (
    DataTest,
    Model,
    ModelError,
    compile_sql,
    discover_models,
    load_tests,
    parse_model,
    run_models,
    run_tests,
    topo_sort,
)


def _write(root: Path, rel: str, sql: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sql)


def test_topo_sort_orders_dependencies() -> None:
    order = topo_sort({"c": ("a", "b"), "b": ("a",), "a": ()})
    assert order == ["a", "b", "c"]


def test_topo_sort_detects_cycle() -> None:
    with pytest.raises(ModelError, match="cycle"):
        topo_sort({"a": ("b",), "b": ("a",)})


def test_topo_sort_unknown_ref() -> None:
    with pytest.raises(ModelError, match="unknown"):
        topo_sort({"a": ("missing",)})


def test_parse_refs_sources_and_materialization(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "staging/x.sql",
        "-- materialized: view\nselect * from {{ source('raw', 't') }} "
        "join {{ ref('y') }} using (id) join {{ ref( \"y\" ) }} using (id)",
    )
    m = parse_model(tmp_path / "staging" / "x.sql", "staging")
    assert m.refs == ("y",)
    assert m.sources == (("raw", "t"),)
    assert m.materialized == "view"


def test_compile_substitutes_refs_sources_and_vars(tmp_path: Path) -> None:
    m = Model(
        "x",
        "marts",
        tmp_path,
        "select {{ var('n') }} from {{ ref('y') }}, {{ source('raw','t') }}",
        ("y",),
        (("raw", "t"),),
        "table",
    )
    y = Model("y", "staging", tmp_path, "select 1", (), (), "view")
    sql = compile_sql(m, {"x": m, "y": y}, {"n": 7})
    assert sql == "select 7 from staging.y, raw.t"
    with pytest.raises(ModelError, match="undefined var"):
        compile_sql(m, {"x": m, "y": y}, {})


def test_runner_end_to_end_on_toy_project(tmp_path: Path) -> None:
    _write(tmp_path, "marts/m2.sql", "select id, v * 2 as v2 from {{ ref('m1') }}")
    _write(
        tmp_path, "staging/m1.sql", "-- materialized: view\nselect * from {{ source('raw','t') }}"
    )
    _write(
        tmp_path,
        "schema.yml",
        "models:\n  - name: m2\n    columns:\n      - name: id\n        tests: [not_null, unique]\n"
        "      - name: v2\n        tests:\n          - accepted_values:\n"
        "              values: [2, 4]\n",
    )
    con = duckdb.connect()
    con.execute(
        "create schema raw; create table raw.t as select * from (values (1, 1), (2, 2)) v(id, v)"
    )
    res = run_models(con, tmp_path)
    assert res.executed == ["m1", "m2"]
    results = run_tests(con, tmp_path)
    assert len(results) == 3 and all(r.passed for r in results)
    con.execute("insert into raw.t values (2, 5)")
    run_models(con, tmp_path)
    failed = {r.test.kind for r in run_tests(con, tmp_path) if not r.passed}
    assert failed == {"unique", "accepted_values"}


def test_relationship_and_expression_tests(tmp_path: Path) -> None:
    _write(tmp_path, "s/parent.sql", "select * from (values (1), (2)) v(id)")
    _write(tmp_path, "s/child.sql", "select * from (values (1, 5), (3, -1)) v(pid, amt)")
    _write(
        tmp_path,
        "schema.yml",
        "models:\n  - name: child\n    tests:\n      - expression_is_true:\n"
        "          expression: amt > 0\n    columns:\n      - name: pid\n        tests:\n"
        "          - relationships: {to: parent, field: id}\n",
    )
    con = duckdb.connect()
    run_models(con, tmp_path)
    res = {r.test.kind: r.failures for r in run_tests(con, tmp_path)}
    assert res == {"expression_is_true": 1, "relationships": 1}


def test_duplicate_model_names_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "a/m.sql", "select 1")
    _write(tmp_path, "b/m.sql", "select 1")
    with pytest.raises(ModelError, match="duplicate"):
        discover_models(tmp_path)


def test_unknown_test_kind(tmp_path: Path) -> None:
    from gal.warehouse.runner import compile_test_sql

    m = Model("m", "s", tmp_path, "select 1", (), (), "table")
    with pytest.raises(ModelError, match="unknown test"):
        compile_test_sql(DataTest("m", "c", "is_awesome"), {"m": m})


def test_project_models_are_layered() -> None:
    models = discover_models(default_models_dir())
    layers = {"staging": 0, "intermediate": 1, "marts": 2}
    for m in models.values():
        for r in m.refs:
            assert layers[models[r].schema] <= layers[m.schema], (m.name, r)
        if m.schema == "staging":
            assert not m.refs and m.sources
        else:
            assert not m.sources, f"{m.name} must read raw data through staging"
    assert len(load_tests(default_models_dir())) >= 40


def test_full_build_passes_all_data_tests(small_run: PipelineRun) -> None:
    tests = small_run.build.tests
    assert tests and all(t.passed for t in tests)
    assert "fct_attribution_paths" in small_run.build.executed


def test_sessionization(small_run: PipelineRun) -> None:
    con = small_run.con
    raw = con.execute("select count(*) from staging.stg_web_events").fetchone()[0]
    sessions = con.execute(
        "select count(*), sum(pageviews) from intermediate.int_sessions"
    ).fetchone()
    assert sessions[0] < raw and sessions[1] == raw
    internal = con.execute(
        "select count(*) from intermediate.int_sessions where landing_channel = 'internal'"
    ).fetchone()[0]
    assert internal == 0


def test_attribution_paths_respect_window(small_run: PipelineRun, small_cfg: SimConfig) -> None:
    df = small_run.con.execute("select * from marts.fct_attribution_paths").df()
    assert (df.days_before_conversion >= 0).all()
    assert (df.days_before_conversion <= small_cfg.attribution.window_days).all()
    assert (df.touch_ts <= df.conversion_at).all()


def test_identity_gaps_drop_touches(small_run: PipelineRun) -> None:
    cov = small_run.con.execute("select * from marts.mart_identity_coverage").df().iloc[0]
    assert 0 < cov.identified_sessions < cov.web_sessions
    assert cov.contacts_domain_matched > 0 and cov.contacts_unresolved > 0


def test_build_raises_on_broken_data(small_run: PipelineRun, small_cfg: SimConfig) -> None:
    con = duckdb.connect()
    tables = {k: v.copy() for k, v in small_run.data.tables.items()}
    tables["accounts"] = pd.concat([tables["accounts"], tables["accounts"].head(1)])
    from gal.warehouse.build import load_raw_frames

    load_raw_frames(con, tables)
    with pytest.raises(DataTestError, match=r"unique:stg_accounts\.account_id"):
        build_warehouse(con, small_cfg)
