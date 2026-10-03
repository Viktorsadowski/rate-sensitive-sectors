"""
Paths and settings shared by all scripts.

The prices are not built here. They come from the insider project (github.com/Viktorsadowski/insider-trade-worth):
its data folder has prices.parquet, sp500_members.csv, universe.csv and price_map.csv. PRICE_DIR in .env says
where that folder is, by default the insider repo next to this one.
The SEC contact line is read from .env too (not in git), or from normal env vars. See .env.example.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env():
    # tiny .env reader so there is no extra dependency. KEY=value per line, real env vars win
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

DATA_DIR = Path(os.environ.get("RATES_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PRICE_DIR = Path(os.environ.get("PRICE_DIR", ROOT.parent / "insider-trade-worth" / "data"))
MANUAL_DIR = ROOT / "data" / "manual"      # small hand-made tables, these are in git
FIG_DIR = ROOT / "figures"

# SEC wants a name and an email in the User-Agent, and max 10 requests a second
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
SEC_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# daily par yield curve, one csv per year. type is daily_treasury_yield_curve or daily_treasury_real_yield_curve
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
                "{year}/all?type={type}&field_tdr_date_value={year}&page&_format=csv")
# effective fed funds rate and the target, from the New York Fed
EFFR_URL = "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate={start}&endDate={end}&type=rate"
# today's S&P 500 members with GICS sector and sub-industry
MEMBERS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

FIRST_YEAR = 2006
BENCHMARK = "SPY"
MAIN_RATE = "y10"            # the 10-year yield: the rate used when a chart or a sentence can only show one
SPLIT = 2017                 # second half starts here, same split as the insider project
MIN_WEEKS = 150              # a bucket needs this many weeks before it gets a beta

# the eleven GICS sectors
SECTORS = ["Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials",
           "Health Care", "Industrials", "Information Technology", "Materials", "Real Estate", "Utilities"]

# chart colours, same as nba-moneyball and insider-trade-worth
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, LIGHT_GREY = "#2a78d6", "#eb6834", "#c9c8c3"
