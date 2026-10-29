"""Generate the synthetic demo sales dataset (demo/demo_sales.csv).

The data is invented (no real dataset is used), so it can be shipped with the
app. The same seed always gives the same file. It has the traps a careful
analyst should notice:
- exact duplicate rows (the same order exported twice);
- an incomplete last month (data ends on 2025-06-11);
- service rows that are not products (shipping fees, discount adjustments);
- a tiny region (Islands, a handful of orders) with large orders, so it tops
  any ranking by average;
- skewed amounts: a few bulk business orders hold a large share of revenue.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_FILE = PROJECT_ROOT / "demo" / "demo_sales.csv"
SEED = 2025

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-06-11")  # the last month is cut off on purpose
ORDERS = 2600
DUPLICATES = 35
SMALL_REGION_ORDERS = 5
BULK_ORDERS = 25

REGIONS = {"North": 0.32, "South": 0.27, "East": 0.23, "West": 0.18}
SMALL_REGION = "Islands"
PRODUCTS = {
    # product: (category, base unit price)
    "Desk Lamp": ("Furniture", 34.0),
    "Office Chair": ("Furniture", 189.0),
    "Bookshelf": ("Furniture", 129.0),
    "Standing Desk": ("Furniture", 449.0),
    "Notebook A5": ("Office Supplies", 4.5),
    "Gel Pens (10 pack)": ("Office Supplies", 8.9),
    "Stapler": ("Office Supplies", 12.5),
    "Paper Ream": ("Office Supplies", 6.2),
    "Wireless Mouse": ("Electronics", 24.9),
    "Keyboard": ("Electronics", 49.0),
    "Monitor 27in": ("Electronics", 279.0),
    "USB-C Hub": ("Electronics", 39.0),
}
SERVICE_ROWS = {"Shipping Fee": 0.08, "Discount Adjustment": 0.03}


def make_demo_data(seed: int = SEED) -> pd.DataFrame:
    """Build the dataset; the same seed gives the same rows."""
    rng = np.random.default_rng(seed)
    days = (END - START).days + 1
    names = list(PRODUCTS)

    def order_rows(count: int, region_choice: list[str], probabilities: list[float] | None) -> pd.DataFrame:
        products = rng.choice(names, size=count)
        base = np.array([PRODUCTS[p][1] for p in products])
        return pd.DataFrame({
            "order_date": START + pd.to_timedelta(rng.integers(0, days, size=count), unit="D"),
            "region": rng.choice(region_choice, size=count, p=probabilities),
            "customer_type": "Consumer",
            "category": [PRODUCTS[p][0] for p in products],
            "product": products,
            "quantity": rng.geometric(0.55, size=count),
            "unit_price": np.round(base * rng.uniform(0.9, 1.1, size=count), 2),
        })

    orders = order_rows(ORDERS, list(REGIONS), list(REGIONS.values()))

    # A few bulk business orders: the skew in amounts.
    bulk = order_rows(BULK_ORDERS, list(REGIONS), list(REGIONS.values()))
    bulk["customer_type"] = "Business"
    bulk["quantity"] = rng.integers(40, 150, size=BULK_ORDERS)

    # The tiny region: few orders, but all of them large.
    small = order_rows(SMALL_REGION_ORDERS, [SMALL_REGION], None)
    small["product"] = rng.choice(["Standing Desk", "Monitor 27in"], size=SMALL_REGION_ORDERS)
    small["category"] = [PRODUCTS[p][0] for p in small["product"]]
    small["unit_price"] = [PRODUCTS[p][1] for p in small["product"]]
    small["quantity"] = rng.integers(3, 6, size=SMALL_REGION_ORDERS)

    data = pd.concat([orders, bulk, small], ignore_index=True)
    data["amount"] = np.round(data["quantity"] * data["unit_price"], 2)

    # Service rows: not products, but they sit in the same table.
    services = []
    for name, share in SERVICE_ROWS.items():
        picked = data.sample(n=int(len(data) * share), random_state=rng.integers(1 << 31)).copy()
        picked["category"] = "Service"
        picked["product"] = name
        picked["quantity"] = 1
        if name == "Shipping Fee":
            picked["unit_price"] = rng.choice([4.99, 9.99, 14.99], size=len(picked))
        else:
            picked["unit_price"] = -np.round(picked["amount"] * 0.1, 2)
        picked["amount"] = picked["unit_price"]
        services.append(picked)
    data = pd.concat([data, *services], ignore_index=True)

    data = data.sort_values(["order_date", "region", "product"], kind="stable", ignore_index=True)
    data.insert(0, "order_id", [f"SO-{10000 + i}" for i in range(len(data))])

    # Exact duplicates: some orders exported twice.
    copies = data.sample(n=DUPLICATES, random_state=rng.integers(1 << 31))
    data = pd.concat([data, copies]).sort_index(kind="stable").reset_index(drop=True)

    data["order_date"] = data["order_date"].dt.strftime("%Y-%m-%d")
    return data


def write_demo_data(path: Path = DEMO_FILE, seed: int = SEED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed line endings and float format: the same bytes on every system.
    make_demo_data(seed).to_csv(path, index=False, lineterminator="\n", float_format="%.2f")
    return path


def main() -> int:
    path = write_demo_data()
    data = pd.read_csv(path)
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}: {len(data):,} rows, {path.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
