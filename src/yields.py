#!/usr/bin/env python3
"""
Treasury yields and the fed funds rate -> data/yields.csv

  date, y3m, y2, y10   par yields of the 3-month bill, the 2-year and the 10-year note (treasury.gov)
  r10                  10-year real yield, from TIPS (treasury.gov)
  effr, target         effective fed funds rate and the target, midpoint when it is a range (New York Fed)

  python src/yields.py

Closed years are kept in data/raw, only this year is fetched again.
"""

import time

import pandas as pd
import requests

from config import DATA_DIR, EFFR_URL, FIRST_YEAR, RAW_DIR, TREASURY_URL

UA = {"User-Agent": "Mozilla/5.0"}


def treasury(kind: str) -> pd.DataFrame:
    """kind: nominal or real. One file per year in data/raw/treasury"""
    d = RAW_DIR / "treasury"
    d.mkdir(parents=True, exist_ok=True)
    typ = "daily_treasury_yield_curve" if kind == "nominal" else "daily_treasury_real_yield_curve"
    this_year = pd.Timestamp.today().year
    parts = []
    for year in range(FIRST_YEAR - 1, this_year + 1):          # one year more, for the first weekly change
        f = d / f"{kind}_{year}.csv"
        if not f.exists() or year == this_year:
            r = requests.get(TREASURY_URL.format(year=year, type=typ), headers=UA, timeout=60)
            r.raise_for_status()
            if "Date" not in r.text[:50]:
                raise SystemExit(f"treasury.gov did not send a csv for {kind} {year}")
            f.write_bytes(r.content)
            time.sleep(0.3)
        parts.append(pd.read_csv(f))
    t = pd.concat(parts)
    t["date"] = pd.to_datetime(t["Date"])
    return t.drop_duplicates("date").set_index("date").sort_index()


def fed_funds() -> pd.DataFrame:
    rows = []
    end = pd.Timestamp.today().normalize()
    # the api cuts long answers, so a few years at a time
    for a in range(FIRST_YEAR - 1, end.year + 1, 6):
        b = min(pd.Timestamp(a + 5, 12, 31), end)
        r = requests.get(EFFR_URL.format(start=f"{a}-01-01", end=b.date()), headers=UA, timeout=60)
        r.raise_for_status()
        rows += r.json()["refRates"]
        time.sleep(0.3)
    e = pd.DataFrame(rows)
    e["date"] = pd.to_datetime(e["effectiveDate"])
    e = e.drop_duplicates("date").set_index("date").sort_index()
    # one target before december 2008, a range after: the midpoint
    lo = e["targetRateFrom"]
    hi = e["targetRateTo"] if "targetRateTo" in e else lo
    return pd.DataFrame({"effr": e["percentRate"], "target": (lo + hi.fillna(lo)) / 2})


def main():
    nom, real = treasury("nominal"), treasury("real")
    y = pd.DataFrame({"y3m": nom["3 Mo"], "y2": nom["2 Yr"], "y10": nom["10 Yr"], "r10": real["10 YR"]})
    y = y.join(fed_funds(), how="outer").sort_index()
    y.index.name = "date"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    y.round(4).to_csv(DATA_DIR / "yields.csv")
    print(f"yields.csv: {len(y):,} days, {y.index[0].date()} to {y.index[-1].date()}")
    print(y.notna().sum().to_string())
    print(y.dropna().iloc[[0, -1]].to_string())


if __name__ == "__main__":
    main()
