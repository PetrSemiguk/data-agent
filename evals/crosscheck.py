"""Second, independent computation of every expected value in questions.json.

reference.py uses pandas; this file reads the CSV files with the csv module and
computes with plain Python (dicts, statistics), so a bug in one is unlikely to
repeat in the other. Values of the final split are not printed unless
--show-final is given; only OK / MISMATCH is shown for them.

Run from the project root:  .venv\\Scripts\\python.exe evals\\crosscheck.py
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
DATA_DIR = EVALS_DIR.parent / "data"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def rows(name: str, encoding: str = "utf-8") -> Iterator[dict[str, str]]:
    with open(DATA_DIR / name, newline="", encoding=encoding) as file:
        yield from csv.DictReader(file)


def rate(hits: dict[str, int], totals: dict[str, int], group: str) -> float:
    return hits[group] / totals[group] * 100


# ---------------------------------------------------------------- dev: Titanic

def d01() -> dict[str, Any]:
    data = list(rows("Titanic-Dataset.csv"))
    return {"passengers": len(data), "survivors": sum(int(r["Survived"]) for r in data)}


def d02() -> dict[str, Any]:
    hits, totals = defaultdict(int), defaultdict(int)
    for r in rows("Titanic-Dataset.csv"):
        totals[r["Sex"]] += 1
        hits[r["Sex"]] += int(r["Survived"])
    return {"female_rate": rate(hits, totals, "female"), "male_rate": rate(hits, totals, "male")}


def d03() -> dict[str, Any]:
    ages = [r["Age"] for r in rows("Titanic-Dataset.csv")]
    known = [float(a) for a in ages if a.strip()]
    return {"mean_age": statistics.fmean(known), "missing_ages": len(ages) - len(known)}


def d04() -> dict[str, Any]:
    fares = defaultdict(list)
    for r in rows("Titanic-Dataset.csv"):
        fares[r["Survived"]].append(float(r["Fare"]))
    return {"median_fare_survived": statistics.median(fares["1"]), "median_fare_died": statistics.median(fares["0"])}


# ------------------------------------------------------------- dev: Superstore

def superstore() -> list[dict[str, str]]:
    return list(rows("Sample - Superstore.csv", encoding="latin-1"))


def d05() -> dict[str, Any]:
    sales = defaultdict(float)
    for r in superstore():
        sales[r["Category"]] += float(r["Sales"])
    return {"category_ranking": sorted(sales, key=sales.get, reverse=True), "technology_sales": sales["Technology"]}


def d06() -> dict[str, Any]:
    sales = defaultdict(float)
    for r in superstore():
        month, _, year = r["Order Date"].split("/")  # M/D/YYYY
        if year == "2017":
            sales[MONTHS[int(month) - 1]] += float(r["Sales"])
    return {"top_months": sorted(sales, key=sales.get, reverse=True)[:3], "november_sales": sales["November"]}


def margins(key: str, data: list[dict[str, str]]) -> dict[str, float]:
    profit, sales = defaultdict(float), defaultdict(float)
    for r in data:
        profit[r[key]] += float(r["Profit"])
        sales[r[key]] += float(r["Sales"])
    return {k: profit[k] / sales[k] * 100 for k in sales}


def d07() -> dict[str, Any]:
    margin = margins("Category", superstore())
    return {"margin_ranking": sorted(margin, key=margin.get, reverse=True), "technology_margin": margin["Technology"]}


def d08() -> dict[str, Any]:
    data = superstore()
    margin = margins("Region", data)
    region = min(margin, key=margin.get)
    losses = defaultdict(float)
    for r in data:
        if r["Region"] == region:
            losses[r["Sub-Category"]] += float(r["Profit"])
    worst = min(losses, key=losses.get)
    return {"weakest_region": region, "weakest_region_margin": margin[region],
            "biggest_loss_subcategory": worst, "biggest_loss": losses[worst]}


# -------------------------------------------------------- dev: Online Retail II

def retail_unique_rows() -> tuple[list[tuple[str, ...]], int, list[str]]:
    """Distinct raw rows (as text) in first-seen order, the row count and the header."""
    with open(DATA_DIR / "online_retail_II.csv", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        seen: dict[tuple[str, ...], None] = {}
        total = 0
        for row in reader:
            total += 1
            seen.setdefault(tuple(row), None)
    return list(seen), total, header


def d09() -> dict[str, Any]:
    customers = {r["Customer ID"] for r in rows("online_retail_II.csv")
                 if not r["Invoice"].startswith("C") and r["Customer ID"].strip()}
    return {"customers": len(customers)}


def revenue_between(unique: list[tuple[str, ...]], header: list[str], start: str, end: str) -> float:
    q, p, d = header.index("Quantity"), header.index("Price"), header.index("InvoiceDate")
    return sum(float(r[q]) * float(r[p]) for r in unique if start <= r[d] < end)


def d10() -> dict[str, Any]:
    unique, _, header = retail_unique_rows()
    last = max(r[header.index("InvoiceDate")] for r in unique)[:10]  # YYYY-MM-DD
    day_after = f"{last[:8]}{int(last[8:10]) + 1:02d}"
    r2010 = revenue_between(unique, header, "2010-01-01", "2010" + day_after[4:])
    r2011 = revenue_between(unique, header, "2011-01-01", day_after)
    return {"growth_same_span_pct": (r2011 / r2010 - 1) * 100, "last_day": int(last[8:10])}


def d11() -> dict[str, Any]:
    unique, total, header = retail_unique_rows()
    return {"net_revenue_2010": revenue_between(unique, header, "2010-01-01", "2011-01-01"),
            "duplicates_removed": total - len(unique)}


# ---------------------------------------------------------------- holdout: HR

def hr() -> list[dict[str, str]]:
    return list(rows("WA_Fn-UseC_-HR-Employee-Attrition.csv"))


def h01() -> dict[str, Any]:
    income = defaultdict(list)
    for r in hr():
        income[r["Attrition"]].append(int(r["MonthlyIncome"]))
    return {"median_income_left": statistics.median(income["Yes"]), "median_income_stayed": statistics.median(income["No"]),
            "n_left": len(income["Yes"]), "n_stayed": len(income["No"])}


def attrition_rates(key: str) -> dict[str, float]:
    hits, totals = defaultdict(int), defaultdict(int)
    for r in hr():
        totals[r[key]] += 1
        hits[r[key]] += r["Attrition"] == "Yes"
    return {k: rate(hits, totals, k) for k in totals}


def h02() -> dict[str, Any]:
    rates = attrition_rates("Department")
    ranking = sorted(rates, key=rates.get, reverse=True)
    return {"department_ranking": ranking, "top_department_rate": rates[ranking[0]]}


def h03() -> dict[str, Any]:
    rates = attrition_rates("OverTime")
    return {"rate_overtime": rates["Yes"], "rate_no_overtime": rates["No"]}


# ----------------------------------------------------------- holdout: Climate

def h04() -> dict[str, Any]:
    temps = defaultdict(list)
    for r in rows("DailyDelhiClimateTrain.csv"):
        temps[r["date"][:4]].append(float(r["meantemp"]))
    full = {year: statistics.fmean(v) for year, v in temps.items() if len(v) >= 365}
    best = max(full, key=full.get)
    return {"warmest_year": best, "warmest_year_mean": full[best]}


def h05() -> dict[str, Any]:
    values = [float(r["meanpressure"]) for r in rows("DailyDelhiClimateTrain.csv")]
    kept = [v for v in values if 900 <= v <= 1100]
    return {"mean_pressure_clean": statistics.fmean(kept), "implausible_values": len(values) - len(kept)}


# --------------------------------------------------------------- final: Heart

def f01() -> dict[str, Any]:
    chol = defaultdict(list)
    zeros = 0
    for r in rows("heart.csv"):
        if int(r["Cholesterol"]) == 0:
            zeros += 1
        else:
            chol[r["HeartDisease"]].append(int(r["Cholesterol"]))
    return {"median_chol_disease": statistics.median(chol["1"]), "median_chol_no_disease": statistics.median(chol["0"]),
            "zero_cholesterol_rows": zeros}


def f02() -> dict[str, Any]:
    hits, totals = defaultdict(int), defaultdict(int)
    for r in rows("heart.csv"):
        totals[r["Sex"]] += 1
        hits[r["Sex"]] += int(r["HeartDisease"])
    return {"rate_male": rate(hits, totals, "M"), "rate_female": rate(hits, totals, "F")}


# -------------------------------------------------------------- final: Houses

def f03() -> dict[str, Any]:
    dates = [r["date"][:8] for r in rows("kc_house_data.csv")]  # YYYYMMDD
    last_day = int(max(dates)[6:8])
    may_2015 = sum(d.startswith("201505") for d in dates)
    may_2014 = sum(d.startswith("201405") and int(d[6:8]) <= last_day for d in dates)
    return {"sales_may_2014_same_days": may_2014, "sales_may_2015": may_2015, "last_day_may_2015": last_day}


def f04() -> dict[str, Any]:
    prices = [float(r["price"]) for r in rows("kc_house_data.csv")]
    return {"median_price": statistics.median(prices), "mean_price": statistics.fmean(prices)}


def f05() -> dict[str, Any]:
    per_sqft = defaultdict(list)
    for r in rows("kc_house_data.csv"):
        per_sqft[r["zipcode"]].append(float(r["price"]) / float(r["sqft_living"]))
    medians = {z: statistics.median(v) for z, v in per_sqft.items() if len(v) >= 100}
    top = sorted(medians, key=medians.get, reverse=True)[:3]
    return {"top_zipcodes": top, "top_zip_median_per_sqft": medians[top[0]]}


CROSSCHECKS: dict[str, Callable[[], dict[str, Any]]] = {
    "d01": d01, "d02": d02, "d03": d03, "d04": d04, "d05": d05, "d06": d06, "d07": d07, "d08": d08,
    "d09": d09, "d10": d10, "d11": d11, "h01": h01, "h02": h02, "h03": h03, "h04": h04, "h05": h05,
    "f01": f01, "f02": f02, "f03": f03, "f04": f04, "f05": f05,
}


def same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str):
        return expected == str(actual)
    if isinstance(expected, list):
        return [str(x) for x in expected] == [str(x) for x in actual][: len(expected)]
    # Only rounding noise is allowed: expected values are stored with 6 decimals.
    return abs(float(expected) - float(actual)) <= max(1e-6, 1e-9 * abs(float(expected)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-final", action="store_true", help="Also print the values of the final split")
    args = parser.parse_args()

    questions = json.loads((EVALS_DIR / "questions.json").read_text(encoding="utf-8"))
    mismatches = 0
    print(f"{'id':4} {'key':26} {'reference.py':>34} {'crosscheck.py':>34}  result")
    for question in questions:
        second = CROSSCHECKS[question["id"]]()
        hide = question["split"] == "final" and not args.show_final
        for check in question["checks"]:
            key = check.get("key")
            if key is None:
                continue
            expected = check["items"] if check["type"] == "ranking" else check["value"]
            ok = same(expected, second[key])
            mismatches += not ok
            left = "(hidden)" if hide else str(expected)
            right = "(hidden)" if hide else str(second[key][: len(expected)] if check["type"] == "ranking" else second[key])
            print(f"{question['id']:4} {key:26} {left[:34]:>34} {right[:34]:>34}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n{'All values match.' if not mismatches else f'{mismatches} MISMATCHES'}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
