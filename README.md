
## Install

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Run with real FIRMS data

```bash
python -m xfire --modis data/modis/*.csv --viirs data/viirs/*.csv --out output
```
## Run a self test

```bash
python -m xfire --self-test
```


## Outputs

- `output/results.json` 
- `output/dashboard.html` 
- `output/quality_report.json`

## Project structure

```text
XFiRE/
├── config/
│   └── regions.json
├── data/
│   ├── modis/
│   └── viirs/
├── output/
├── src/
│   └── xfire/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── ingest.py
│       ├── harmonize.py
│       ├── intelligence.py
│       ├── pipeline.py
│       └── dashboard.py
├── requirements.txt
└── README.md
```
