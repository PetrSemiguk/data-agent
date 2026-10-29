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
from functools import cache
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

@cache
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


# ------------------------------------------------------ dev_hard: Superstore

def x01() -> dict[str, Any]:
    sales = defaultdict(float)
    for r in superstore():
        sales[r["Sub-Category"]] += float(r["Sales"])
    top = sorted(sales, key=sales.get, reverse=True)
    return {"top_subcategories": top[:5], "first_sales": sales[top[0]], "second_sales": sales[top[1]]}


def x02() -> dict[str, Any]:
    sales = defaultdict(float)
    for r in superstore():
        sales[(r["Order Date"].split("/")[2], r["Sub-Category"])] += float(r["Sales"])
    growth = {sub: (sales[("2017", sub)] / sales[("2016", sub)] - 1) * 100
              for (year, sub) in sales if year == "2016" and sales[("2016", sub)] >= 10_000}
    best = max(growth, key=growth.get)
    return {"fastest_subcategory": best, "sales_2016": sales[("2016", best)], "sales_2017": sales[("2017", best)],
            "growth_pct": growth[best]}


def x03() -> dict[str, Any]:
    data = superstore()
    discounts = [float(r["Discount"]) for r in data]
    sales = [float(r["Sales"]) for r in data]
    weighted = sum(d * s for d, s in zip(discounts, sales)) / sum(sales)
    return {"mean_discount": statistics.fmean(discounts) * 100, "weighted_discount": weighted * 100}


def x04() -> dict[str, Any]:
    lines = defaultdict(list)
    for r in superstore():
        lines[r["Order ID"]].append((float(r["Profit"]), float(r["Sales"])))
    loss_orders = [rows for rows in lines.values() if any(p < 0 for p, _ in rows)]
    loss_sales = sum(s for rows in loss_orders for p, s in rows if p < 0)
    all_sales = sum(s for rows in loss_orders for _, s in rows)
    return {"orders_with_loss": len(loss_orders), "share_orders_pct": len(loss_orders) / len(lines) * 100,
            "loss_lines_sales_pct": loss_sales / all_sales * 100}


# --------------------------------------------------------- dev_hard: Titanic

def x05() -> dict[str, Any]:
    names = {"1": "first", "2": "second", "3": "third"}
    hits, totals = defaultdict(int), defaultdict(int)
    for r in rows("Titanic-Dataset.csv"):
        key = f"{names[r['Pclass']]}_{r['Sex']}"
        totals[key] += 1
        hits[key] += int(r["Survived"])
    return {key: rate(hits, totals, key) for key in totals}


def x06() -> dict[str, Any]:
    data = list(rows("Titanic-Dataset.csv"))
    southampton = [r for r in data if r["Embarked"] == "S"]
    third = [r for r in southampton if r["Pclass"] == "3"]
    survivors = sum(int(r["Survived"]) for r in third)
    return {"third_share_of_s": len(third) / len(southampton) * 100, "survival_third_s": survivors / len(third) * 100,
            "third_s_survivors_of_all": survivors / len(data) * 100}


def x07() -> dict[str, Any]:
    third = [r for r in rows("Titanic-Dataset.csv") if r["Pclass"] == "3"]
    known = [r for r in third if r["Age"].strip()]
    children = [int(r["Survived"]) for r in known if float(r["Age"]) < 18]
    adults = [int(r["Survived"]) for r in known if float(r["Age"]) >= 18]
    return {"child_rate": sum(children) / len(children) * 100, "adult_rate": sum(adults) / len(adults) * 100,
            "missing_age": len(third) - len(known)}


# -------------------------------------------------- dev_hard: Online Retail II

def x08() -> dict[str, Any]:
    unique, _, header = retail_unique_rows()
    q, p, d = header.index("Quantity"), header.index("Price"), header.index("InvoiceDate")
    revenue, days = defaultdict(float), defaultdict(set)
    for r in unique:
        if r[d].startswith("2010-"):
            month = int(r[d][5:7])
            revenue[month] += float(r[q]) * float(r[p])
            days[month].add(r[d][:10])
    per_day = {m: revenue[m] / len(days[m]) for m in revenue}
    lowest = min(per_day, key=per_day.get)
    return {"lowest_month": MONTHS[lowest - 1], "lowest_per_day": per_day[lowest]}


def x09() -> dict[str, Any]:
    unique, _, header = retail_unique_rows()
    q, p, d, inv = (header.index(c) for c in ("Quantity", "Price", "InvoiceDate", "Invoice"))
    gross = lost = 0.0
    for r in unique:
        if r[d].startswith("2011-"):
            value = float(r[q]) * float(r[p])
            if r[inv].startswith("C"):
                lost += value
            else:
                gross += value
    return {"gross_2011": gross, "cancelled_2011": abs(lost), "cancelled_share_pct": abs(lost) / gross * 100}


def x10() -> dict[str, Any]:
    unique, _, header = retail_unique_rows()
    q, p, cid, country = (header.index(c) for c in ("Quantity", "Price", "Customer ID", "Country"))
    revenue, customers = defaultdict(float), defaultdict(set)
    for r in unique:
        if r[cid].strip():
            revenue[r[country]] += float(r[q]) * float(r[p])
            customers[r[country]].add(r[cid])
    average = {c: revenue[c] / len(customers[c]) for c in revenue if len(customers[c]) >= 20}
    top = sorted(average, key=average.get, reverse=True)[:3]
    return {"top_countries": top, "top_country_avg": average[top[0]]}


# --------------------------------------------- dev_open: questions without hints

def o01() -> dict[str, Any]:
    unique, _, header = retail_unique_rows()
    q, p, cid, country = (header.index(c) for c in ("Quantity", "Price", "Customer ID", "Country"))
    revenue, customers = defaultdict(float), defaultdict(set)
    for r in unique:
        if r[cid].strip():
            revenue[r[country]] += float(r[q]) * float(r[p])
            customers[r[country]].add(r[cid])
    average = {c: revenue[c] / len(customers[c]) for c in revenue}
    top_any = max(average, key=average.get)
    top_10 = max((c for c in average if len(customers[c]) >= 10), key=average.get)
    return {"top_country": list(dict.fromkeys([top_any, top_10]))}


def o02() -> dict[str, Any]:
    days = [int(r["InvoiceDate"][8:10]) for r in rows("online_retail_II.csv") if r["InvoiceDate"].startswith("2011-12")]
    return {"last_day": max(days)}


def o03() -> dict[str, Any]:
    invoices = {r["Invoice"] for r in rows("online_retail_II.csv")
                if r["InvoiceDate"].startswith("2011-") and not r["Invoice"].startswith("C")}
    return {"orders_2011": len(invoices)}


def o04() -> dict[str, Any]:
    totals = defaultdict(float)
    for r in superstore():
        totals[r["Order ID"]] += float(r["Sales"])
    return {"median_order_value": statistics.median(totals.values())}


def o05() -> dict[str, Any]:
    data = list(rows("Titanic-Dataset.csv"))
    known = [(float(r["Age"]), int(r["Survived"])) for r in data if r["Age"].strip()]
    children = [s for age, s in known if age < 18]
    adults = [s for age, s in known if age >= 18]
    higher = sum(children) / len(children) > sum(adults) / len(adults)
    return {"children_higher": "yes" if higher else "no", "missing_age": len(data) - len(known)}


def o06() -> dict[str, Any]:
    unique, _, header = retail_unique_rows()
    q, p, code, desc = (header.index(c) for c in ("Quantity", "Price", "StockCode", "Description"))
    revenue, names = defaultdict(float), defaultdict(lambda: defaultdict(int))
    for r in unique:
        if r[code][:1].isdigit():
            revenue[r[code]] += float(r[q]) * float(r[p])
            if r[desc].strip() and float(r[p]) > 0:  # zero-price lines are stock notes, not names
                names[r[code]][r[desc].strip()] += 1
    top = sorted(revenue, key=revenue.get, reverse=True)[:3]
    return {"top_products": [sorted(names[c], key=lambda n: (-names[c][n], n)) + [c] for c in top]}


def o07() -> dict[str, Any]:
    profit, sales, orders = defaultdict(float), defaultdict(float), defaultdict(set)
    for r in superstore():
        profit[r["State"]] += float(r["Profit"])
        sales[r["State"]] += float(r["Sales"])
        orders[r["State"]].add(r["Order ID"])
    margin = {s: profit[s] / sales[s] for s in sales}
    accepted = [max(margin, key=margin.get)]
    accepted += [max((s for s in margin if len(orders[s]) >= minimum), key=margin.get) for minimum in (20, 50, 100)]
    return {"top_state": list(dict.fromkeys(accepted))}


def o08() -> dict[str, Any]:
    alone = [int(r["Survived"]) for r in rows("Titanic-Dataset.csv") if int(r["SibSp"]) + int(r["Parch"]) == 0]
    return {"alone_rate": sum(alone) / len(alone) * 100}


CROSSCHECKS: dict[str, Callable[[], dict[str, Any]]] = {
    "o01": o01, "o02": o02, "o03": o03, "o04": o04, "o05": o05, "o06": o06, "o07": o07, "o08": o08,
    "x01": x01, "x02": x02, "x03": x03, "x04": x04, "x05": x05,
    "x06": x06, "x07": x07, "x08": x08, "x09": x09, "x10": x10,
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
