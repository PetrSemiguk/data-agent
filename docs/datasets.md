# Datasets

## Test datasets (Kaggle)

`python scripts/download_data.py` downloads them into `data/` (not stored in git).

| Dataset | What it is | Used for |
|---|---|---|
| [Superstore](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) | sales, profit, dates | development (`dev` splits) |
| [Titanic](https://www.kaggle.com/datasets/yasserh/titanic-dataset) | small, with missing values | development |
| [Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) | large (1M rows, 90 MB) and messy: returns, negative quantities | development |
| [IBM HR Analytics Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) | employees who left or stayed | hold-out: evaluation only |
| [Daily Climate (Delhi)](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) | daily weather measurements | hold-out: evaluation only |
| [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | clinical measurements | final test: run once at the end |
| [House Sales in King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) | home sales with prices and dates | final test: run once at the end |

Hold-out datasets were never used when writing prompts or skills, so they show
whether the agent's methods generalize to data it was not tuned on. The final test
datasets are stricter: the eval ran on them once, after all changes.

## Demo dataset

`demo/demo_sales.csv` is synthetic sales data made by `scripts/make_demo_data.py`
with a fixed seed, so the file is always the same. It is invented rather than taken
from Kaggle because the test datasets' licenses do not clearly allow shipping them
with an app. It contains the traps a careful analyst should notice: exact duplicate
rows, an incomplete last month, service rows that are not products (shipping fees,
discount adjustments), a tiny region that tops any ranking by average, and skewed
amounts.
