# Refresh From Scratch (Post-Schema-Change Runbook)

Use this when schema/model changes are large and you want a clean rebuild.

## 1) Pull latest code

```powershell
git checkout master
git pull --ff-only
```

## 2) Recreate Python env

```powershell
if (Test-Path .venv) { Remove-Item .venv -Recurse -Force }
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e .[dev,docs]
.\.venv\Scripts\python -m spacy download en_core_web_sm
```

## 3) Reset generated artifacts (keep source texts/seeds)

```powershell
$toClear = @('data\output','data\exports','data\corpus','data\style','data\voice','data\worldbible')
foreach($d in $toClear){
  if(Test-Path $d){ Remove-Item "$d\*" -Recurse -Force -ErrorAction SilentlyContinue }
  else { New-Item -ItemType Directory -Path $d | Out-Null }
  if(-not (Test-Path (Join-Path $d '.gitkeep'))){ New-Item -ItemType File -Path (Join-Path $d '.gitkeep') -Force | Out-Null }
}
if(Test-Path 'data\bga_analytics.duckdb'){ Remove-Item 'data\bga_analytics.duckdb' -Force }
```

## 4) Reset Neo4j completely

```powershell
docker compose down -v
docker compose up -d
```

> Note: container health may show `unhealthy` while startup check is noisy, but Bolt can still be reachable. Verify with `bga status` and Cypher checks below.

## 5) Rebuild core graph from canonical books

(Using pipeline full to guarantee graph writes)

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\bga.exe pipeline full data\texts\lotr-corpus\hobbit.txt -t "The Hobbit" -a "J.R.R. Tolkien" -o data\output\refresh
.\.venv\Scripts\bga.exe pipeline full data\texts\lotr-corpus\fellowship.txt -t "The Fellowship of the Ring" -a "J.R.R. Tolkien" -o data\output\refresh
.\.venv\Scripts\bga.exe pipeline full data\texts\lotr-corpus\silmarillion.txt -t "The Silmarillion" -a "J.R.R. Tolkien" -o data\output\refresh
```

## 6) Verify database contents

```powershell
docker exec bga-neo4j cypher-shell -u neo4j -p bookgraph123 "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS c ORDER BY c DESC LIMIT 20;"
docker exec bga-neo4j cypher-shell -u neo4j -p bookgraph123 "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS c ORDER BY c DESC LIMIT 20;"
```

## 7) Smoke tests + docs build

```powershell
.\.venv\Scripts\python -m pytest -q tests/test_worldbuilding_kickoff.py tests/test_genealogy.py
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\mkdocs build --strict -q
```

## Known gotchas

- `bga corpus process` can complete corpus extraction but may not fully populate entity/relationship graph by itself.
- For guaranteed graph refresh after schema changes, use `pipeline full` per canonical text as above.
- If MkDocs fails on Windows with unicode output, set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`.
