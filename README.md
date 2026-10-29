# data-agent

## Test datasets

- [Superstore](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final): sales, profit, dates; for revenue and seasonality questions.
- [Titanic](https://www.kaggle.com/datasets/yasserh/titanic-dataset): small dataset with missing values.
- [Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci): large and messy; returns, negative quantities.
- [IBM HR Analytics Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset): employees who left or stayed; for comparing groups. Hold-out: used only for evaluation.
- [Daily Climate (Delhi)](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data): daily weather measurements; for time series questions. Hold-out: used only for evaluation.
- [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction): clinical measurements of patients with and without heart disease. Final test: run once at the very end.
- [House Sales in King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction): home sales with prices, sizes and sale dates. Final test: run once at the very end.

Hold-out datasets are never used when writing prompts or skills, so they show whether
the agent's methods generalize to data it was not tuned on.
Final test datasets are stricter: the eval runs on them only once, after all changes.

Run `python scripts/download_data.py` to download them into the data/ folder.

## Run with Docker

Needs `.env` with `ANTHROPIC_API_KEY` (see `.env.example`) and the datasets in `data/`.

```
docker compose build
docker compose up -d                 # web app on http://localhost:8501
docker compose run --rm app python scripts/run_agent.py --file "data/Titanic-Dataset.csv" --question "How many passengers survived?"
docker compose down
```

The app runs as a non-root user on a read-only file system; `data/` is mounted
read-only, and the analysis kernel gets no API key or other secrets.
