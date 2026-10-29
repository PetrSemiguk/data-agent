"""Reference answers: one function per eval question, computed with pandas.

Each function returns {check key: expected value}. A number check gets a float,
a label check a string, and a ranking check the list of items in order. build_expected.py writes the
values into questions.json.
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


@cache
def load(name: str) -> pd.DataFrame:
    """The raw dataset, read once; functions must not change it."""
    try:
        return pd.read_csv(DATA_DIR / name)
    except UnicodeDecodeError:
        return pd.read_csv(DATA_DIR / name, encoding="latin-1")


def titanic() -> pd.DataFrame:
    return load("Titanic-Dataset.csv")


def superstore() -> pd.DataFrame:
    data = load("Sample - Superstore.csv").copy()
    data["Order Date"] = pd.to_datetime(data["Order Date"], format="%m/%d/%Y")
    return data


def retail() -> pd.DataFrame:
    data = load("online_retail_II.csv").copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["Revenue"] = data["Quantity"] * data["Price"]
    return data


def hr() -> pd.DataFrame:
    return load("WA_Fn-UseC_-HR-Employee-Attrition.csv")


def climate() -> pd.DataFrame:
    data = load("DailyDelhiClimateTrain.csv").copy()
    data["date"] = pd.to_datetime(data["date"])
    return data


def heart() -> pd.DataFrame:
    return load("heart.csv")


def houses() -> pd.DataFrame:
    data = load("kc_house_data.csv").copy()
    data["date"] = pd.to_datetime(data["date"], format="%Y%m%dT%H%M%S")
    return data


# ---------------------------------------------------------------- dev: Titanic

def d01_titanic_counts() -> dict[str, Any]:
    data = titanic()
    return {"passengers": float(len(data)), "survivors": float(data["Survived"].sum())}


def d02_titanic_survival_by_sex() -> dict[str, Any]:
    rates = titanic().groupby("Sex")["Survived"].mean() * 100
    return {"female_rate": float(rates["female"]), "male_rate": float(rates["male"])}


def d03_titanic_average_age() -> dict[str, Any]:
    ages = titanic()["Age"]
    return {"mean_age": float(ages.mean()), "missing_ages": float(ages.isna().sum())}


def d04_titanic_fare_by_survival() -> dict[str, Any]:
    medians = titanic().groupby("Survived")["Fare"].median()
    return {"median_fare_survived": float(medians[1]), "median_fare_died": float(medians[0])}


# ------------------------------------------------------------- dev: Superstore

def d05_superstore_sales_by_category() -> dict[str, Any]:
    totals = superstore().groupby("Category")["Sales"].sum().sort_values(ascending=False)
    return {"category_ranking": list(totals.index), "technology_sales": float(totals["Technology"])}


def d06_superstore_top_months_2017() -> dict[str, Any]:
    data = superstore()
    year = data[data["Order Date"].dt.year == 2017]
    months = year.groupby(year["Order Date"].dt.month)["Sales"].sum().sort_values(ascending=False)
    return {
        "top_months": [MONTHS[m - 1] for m in months.index[:3]],
        "november_sales": float(months[11]),
    }


def d07_superstore_margin_by_category() -> dict[str, Any]:
    totals = superstore().groupby("Category")[["Profit", "Sales"]].sum()
    margin = (totals["Profit"] / totals["Sales"] * 100).sort_values(ascending=False)
    return {"margin_ranking": list(margin.index), "technology_margin": float(margin["Technology"])}


def d08_superstore_weakest_region() -> dict[str, Any]:
    data = superstore()
    totals = data.groupby("Region")[["Profit", "Sales"]].sum()
    margin = (totals["Profit"] / totals["Sales"] * 100).sort_values()
    region = margin.index[0]
    losses = data[data["Region"] == region].groupby("Sub-Category")["Profit"].sum().sort_values()
    return {
        "weakest_region": region,
        "weakest_region_margin": float(margin.iloc[0]),
        "biggest_loss_subcategory": losses.index[0],
        "biggest_loss": float(losses.iloc[0]),
    }


# -------------------------------------------------------- dev: Online Retail II

def d09_retail_purchasing_customers() -> dict[str, Any]:
    data = retail()
    purchases = data[~data["Invoice"].astype(str).str.startswith("C")]
    return {"customers": float(purchases["Customer ID"].dropna().nunique())}


def d10_retail_growth_2011() -> dict[str, Any]:
    data = retail().drop_duplicates()
    last_day = data["InvoiceDate"].max().normalize()
    end_2011 = last_day + pd.Timedelta(days=1)
    end_2010 = end_2011 - pd.DateOffset(years=1)
    same_span = {
        year: data[(data["InvoiceDate"] >= f"{year}-01-01") & (data["InvoiceDate"] < end)]["Revenue"].sum()
        for year, end in ((2010, end_2010), (2011, end_2011))
    }
    return {
        "growth_same_span_pct": float((same_span[2011] / same_span[2010] - 1) * 100),
        "last_day": float(last_day.day),  # informational: the last day of December 2011 in the data
    }


def d11_retail_net_revenue_2010() -> dict[str, Any]:
    data = retail()
    deduplicated = data.drop_duplicates()
    in_2010 = deduplicated[deduplicated["InvoiceDate"].dt.year == 2010]
    return {
        "net_revenue_2010": float(in_2010["Revenue"].sum()),
        "duplicates_removed": float(len(data) - len(deduplicated)),
    }


# ---------------------------------------------------------------- holdout: HR

def h01_hr_income_by_attrition() -> dict[str, Any]:
    data = hr()
    groups = data.groupby("Attrition")["MonthlyIncome"]
    medians, sizes = groups.median(), groups.size()
    return {
        "median_income_left": float(medians["Yes"]),
        "median_income_stayed": float(medians["No"]),
        "n_left": float(sizes["Yes"]),
        "n_stayed": float(sizes["No"]),
    }


def h02_hr_attrition_by_department() -> dict[str, Any]:
    data = hr()
    rate = (data["Attrition"] == "Yes").groupby(data["Department"]).mean().mul(100).sort_values(ascending=False)
    return {"department_ranking": list(rate.index), "top_department_rate": float(rate.iloc[0])}


def h03_hr_attrition_by_overtime() -> dict[str, Any]:
    data = hr()
    rate = (data["Attrition"] == "Yes").groupby(data["OverTime"]).mean() * 100
    return {"rate_overtime": float(rate["Yes"]), "rate_no_overtime": float(rate["No"])}


# ----------------------------------------------------------- holdout: Climate

def h04_climate_warmest_year() -> dict[str, Any]:
    data = climate()
    days = data.groupby(data["date"].dt.year).size()
    complete = days[days >= 365].index
    means = data[data["date"].dt.year.isin(complete)].groupby(data["date"].dt.year)["meantemp"].mean()
    return {"warmest_year": str(int(means.idxmax())), "warmest_year_mean": float(means.max())}


def h05_climate_average_pressure() -> dict[str, Any]:
    pressure = climate()["meanpressure"]
    # Physically plausible sea-level pressure; the rest are recording errors.
    plausible = pressure[(pressure >= 900) & (pressure <= 1100)]
    return {"mean_pressure_clean": float(plausible.mean()), "implausible_values": float(len(pressure) - len(plausible))}


# --------------------------------------------------------------- final: Heart

def f01_heart_cholesterol() -> dict[str, Any]:
    data = heart()
    # Cholesterol 0 is a missing value, not a measurement.
    measured = data[data["Cholesterol"] > 0]
    medians = measured.groupby("HeartDisease")["Cholesterol"].median()
    return {
        "median_chol_disease": float(medians[1]),
        "median_chol_no_disease": float(medians[0]),
        "zero_cholesterol_rows": float((data["Cholesterol"] == 0).sum()),
    }


def f02_heart_disease_by_sex() -> dict[str, Any]:
    rate = heart().groupby("Sex")["HeartDisease"].mean() * 100
    return {"rate_male": float(rate["M"]), "rate_female": float(rate["F"])}


# -------------------------------------------------------------- final: Houses

def f03_houses_may_sales() -> dict[str, Any]:
    data = houses()
    last_day = data["date"].max().day
    may = {year: data[(data["date"].dt.year == year) & (data["date"].dt.month == 5)] for year in (2014, 2015)}
    same_days = {year: frame[frame["date"].dt.day <= last_day] for year, frame in may.items()}
    return {
        "sales_may_2014_same_days": float(len(same_days[2014])),
        "sales_may_2015": float(len(may[2015])),
        "last_day_may_2015": float(last_day),
    }


def f04_houses_typical_price() -> dict[str, Any]:
    price = houses()["price"]
    return {"median_price": float(price.median()), "mean_price": float(price.mean())}


def f05_houses_price_per_sqft_zipcodes() -> dict[str, Any]:
    data = houses()
    per_sqft = data["price"] / data["sqft_living"]
    by_zip = per_sqft.groupby(data["zipcode"])
    stats = pd.DataFrame({"sales": by_zip.size(), "median": by_zip.median()})
    top = stats[stats["sales"] >= 100].sort_values("median", ascending=False).head(3)
    return {"top_zipcodes": [str(z) for z in top.index], "top_zip_median_per_sqft": float(top["median"].iloc[0])}


REFERENCES: dict[str, Callable[[], dict[str, Any]]] = {
    name: function for name, function in globals().items()
    if callable(function) and name[:1] in "dhf" and name[1:3].isdigit()
}
