"""A tiny dbt-style model runner for DuckDB.

* Models are ``.sql`` files under ``models/<layer>/``; the folder is the DuckDB schema.
* ``{{ ref('model') }}`` and ``{{ source('raw', 'table') }}`` declare dependencies;
  ``{{ var('name') }}`` substitutes run-time variables.
* An optional first-line header ``-- materialized: view|table`` picks materialisation.
* Models run in topological order (Kahn's algorithm; cycles are an error).
* ``schema.yml`` files declare column tests: ``not_null``, ``unique``,
  ``accepted_values``, ``relationships`` and model-level ``expression_is_true``.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REF_RE = re.compile(r"\{\{\s*ref\(\s*['\"]([A-Za-z0-9_]+)['\"]\s*\)\s*\}\}")
SOURCE_RE = re.compile(
    r"\{\{\s*source\(\s*['\"]([A-Za-z0-9_]+)['\"]\s*,\s*['\"]([A-Za-z0-9_]+)['\"]\s*\)\s*\}\}"
)
VAR_RE = re.compile(r"\{\{\s*var\(\s*['\"]([A-Za-z0-9_]+)['\"]\s*\)\s*\}\}")
MATERIALIZED_RE = re.compile(r"^\s*--\s*materialized:\s*(view|table)\s*$", re.MULTILINE)


class ModelError(RuntimeError):
    """Raised for unknown refs, dependency cycles or missing variables."""


@dataclass(frozen=True)
class Model:
    name: str
    schema: str
    path: Path
    sql: str
    refs: tuple[str, ...]
    sources: tuple[tuple[str, str], ...]
    materialized: str

    @property
    def relation(self) -> str:
        return f"{self.schema}.{self.name}"


@dataclass(frozen=True)
class DataTest:
    model: str
    column: str | None
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        col = f".{self.column}" if self.column else ""
        return f"{self.kind}:{self.model}{col}"


@dataclass(frozen=True)
class DataTestResult:
    test: DataTest
    failures: int

    @property
    def passed(self) -> bool:
        return self.failures == 0


@dataclass
class RunResult:
    executed: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)
    tests: list[DataTestResult] = field(default_factory=list)

    @property
    def failed_tests(self) -> list[DataTestResult]:
        return [t for t in self.tests if not t.passed]


def parse_model(path: Path, schema: str) -> Model:
    sql = path.read_text(encoding="utf-8")
    refs = tuple(dict.fromkeys(REF_RE.findall(sql)))
    sources = tuple(dict.fromkeys(SOURCE_RE.findall(sql)))
    m = MATERIALIZED_RE.search(sql)
    return Model(
        name=path.stem,
        schema=schema,
        path=path,
        sql=sql,
        refs=refs,
        sources=sources,
        materialized=m.group(1) if m else "table",
    )


def discover_models(models_dir: str | Path) -> dict[str, Model]:
    """Find every ``models/<layer>/*.sql`` file. Model names must be globally unique."""
    root = Path(models_dir)
    models: dict[str, Model] = {}
    for path in sorted(root.glob("*/*.sql")):
        model = parse_model(path, schema=path.parent.name)
        if model.name in models:
            raise ModelError(f"duplicate model name {model.name!r}")
        models[model.name] = model
    if not models:
        raise ModelError(f"no models found under {root}")
    return models


def topo_sort(deps: dict[str, tuple[str, ...]]) -> list[str]:
    """Deterministic Kahn topological sort. ``deps[x]`` = nodes ``x`` depends on."""
    for node, parents in deps.items():
        for p in parents:
            if p not in deps:
                raise ModelError(f"{node} references unknown model {p!r}")
    indegree = {n: len(set(ps)) for n, ps in deps.items()}
    children: dict[str, list[str]] = {n: [] for n in deps}
    for n, ps in deps.items():
        for p in set(ps):
            children[p].append(n)
    ready = deque(sorted(n for n, d in indegree.items() if d == 0))
    order: list[str] = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for c in sorted(children[n]):
            indegree[c] -= 1
            if indegree[c] == 0:
                ready.append(c)
    if len(order) != len(deps):
        cyclic = sorted(n for n, d in indegree.items() if d > 0)
        raise ModelError(f"dependency cycle among models: {cyclic}")
    return order


def compile_sql(model: Model, models: dict[str, Model], variables: dict[str, Any]) -> str:
    def _ref(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in models:
            raise ModelError(f"{model.name} references unknown model {name!r}")
        return models[name].relation

    def _var(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in variables:
            raise ModelError(f"{model.name} uses undefined var {key!r}")
        return str(variables[key])

    sql = REF_RE.sub(_ref, model.sql)
    sql = SOURCE_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}", sql)
    sql = VAR_RE.sub(_var, sql)
    return sql.strip().rstrip(";")


def run_models(
    con: Any,
    models_dir: str | Path,
    variables: dict[str, Any] | None = None,
) -> RunResult:
    """Materialise every model in dependency order and return timings."""
    variables = variables or {}
    models = discover_models(models_dir)
    order = topo_sort({name: m.refs for name, m in models.items()})
    result = RunResult()
    for schema in sorted({m.schema for m in models.values()}):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    for name in order:
        model = models[name]
        sql = compile_sql(model, models, variables)
        kind = "VIEW" if model.materialized == "view" else "TABLE"
        start = time.perf_counter()
        con.execute(f"CREATE OR REPLACE {kind} {model.relation} AS {sql}")
        result.seconds[name] = time.perf_counter() - start
        result.executed.append(name)
    return result


def load_tests(models_dir: str | Path) -> list[DataTest]:
    """Parse every ``schema.yml`` under ``models_dir``."""
    tests: list[DataTest] = []
    for path in sorted(Path(models_dir).rglob("schema.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for model in doc.get("models", []):
            mname = model["name"]
            for t in model.get("tests", []):
                kind, params = _split_test(t)
                tests.append(DataTest(model=mname, column=None, kind=kind, params=params))
            for col in model.get("columns", []):
                for t in col.get("tests", []):
                    kind, params = _split_test(t)
                    tests.append(
                        DataTest(model=mname, column=col["name"], kind=kind, params=params)
                    )
    return tests


def _split_test(spec: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict) and len(spec) == 1:
        kind, params = next(iter(spec.items()))
        return str(kind), dict(params or {})
    raise ModelError(f"cannot parse test spec {spec!r}")


def _quote(v: Any) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int | float):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def compile_test_sql(test: DataTest, models: dict[str, Model]) -> str:
    """Compile a data test to a query returning the number of failing rows."""
    if test.model not in models:
        raise ModelError(f"test {test.name} targets unknown model")
    rel = models[test.model].relation
    col = test.column
    if test.kind == "not_null":
        return f"SELECT count(*) FROM {rel} WHERE {col} IS NULL"
    if test.kind == "unique":
        return (
            f"SELECT count(*) FROM (SELECT {col} FROM {rel} WHERE {col} IS NOT NULL "
            f"GROUP BY {col} HAVING count(*) > 1)"
        )
    if test.kind == "accepted_values":
        values = ", ".join(_quote(v) for v in test.params["values"])
        return f"SELECT count(*) FROM {rel} WHERE {col} IS NOT NULL AND {col} NOT IN ({values})"
    if test.kind == "relationships":
        to = test.params["to"]
        if to not in models:
            raise ModelError(f"relationships test points to unknown model {to!r}")
        field_ = test.params["field"]
        parent = models[to].relation
        return (
            f"SELECT count(*) FROM {rel} c LEFT JOIN {parent} p ON c.{col} = p.{field_} "
            f"WHERE c.{col} IS NOT NULL AND p.{field_} IS NULL"
        )
    if test.kind == "expression_is_true":
        return f"SELECT count(*) FROM {rel} WHERE NOT ({test.params['expression']})"
    raise ModelError(f"unknown test kind {test.kind!r}")


def run_tests(
    con: Any, models_dir: str | Path, variables: dict[str, Any] | None = None
) -> list[DataTestResult]:
    """Execute every declared data test; a test passes when it returns zero failing rows."""
    variables = variables or {}
    models = discover_models(models_dir)
    results: list[DataTestResult] = []
    for t in load_tests(models_dir):
        sql = VAR_RE.sub(lambda m: str(variables[m.group(1)]), compile_test_sql(t, models))
        failures = int(con.execute(sql).fetchone()[0])
        results.append(DataTestResult(test=t, failures=failures))
    return results
