## Data Version Control (DVC)

### Setup
```bash
pip install dvc
dvc init
dvc remote add -d localstorage /path/to/local/storage
```

### Reproduce Data Pipeline
```bash
dvc pull   # download tracked data
dvc push   # upload data to remote
```

### Data Versions
- `data/MachineLearningRating_v3.txt` — raw dataset (1M rows)
- `data/insurance_cleaned.csv` — cleaned and typed dataset