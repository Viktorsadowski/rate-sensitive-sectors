#!/usr/bin/env python3
"""
A sector for every company with a price series -> data/classes.csv

  ticker, cik, first_year, last_year   one row per price series and company (a ticker can be two companies over time)
  sector, sub                          GICS sector and sub-industry
  source                               list: today's S&P 500 list. sic: the SEC's SIC code, sector only.
                                       hand: data/manual/sector_by_hand.csv

Today's members get sector and sub-industry from the public list. The companies that left the index are not on any
free list, so they get a sector from their SIC code (data/manual/sic_sectors.csv), a few by hand. Without them
the early years would be survivors only.

  python src/classes.py

The SIC codes are kept in data/raw/sic.csv, only missing ones are fetched (about 800 the first time, 5 to 10
minutes). A run that stops halfway keeps what it has.
"""

import io
import sys
import time

import pandas as pd
import requests

from config import (DATA_DIR, MANUAL_DIR, MEMBERS_URL, PRICE_DIR, RAW_DIR, SEC_COMPANY_URL, SEC_USER_AGENT, SECTORS)

SIC_FILE = RAW_DIR / "sic.csv"


def todays_members() -> pd.DataFrame:
    r = requests.get(MEMBERS_URL, timeout=60)
    r.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "sp500_today.csv").write_bytes(r.content)
    m = pd.read_csv(io.BytesIO(r.content))
    return m.rename(columns={"Symbol": "ticker", "GICS Sector": "sector", "GICS Sub-Industry": "sub"}) \
        .set_index("ticker")[["sector", "sub"]]


def ask_sec(cik: int) -> dict:
    """one company from the SEC. A few tries, the connection drops now and then and the SEC sends 429 when it
    is busy"""
    for wait in (0, 5, 20, 60):
        time.sleep(wait)
        try:
            r = requests.get(SEC_COMPANY_URL.format(cik=cik), headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
        except requests.RequestException:
            continue
        if r.status_code == 404:                     # the SEC does not know the number
            return {"cik": cik, "sic": None, "sic_description": "", "name": ""}
        if r.status_code == 200:
            j = r.json()
            return {"cik": cik, "sic": j.get("sic") or None, "sic_description": j.get("sicDescription", ""),
                    "name": j.get("name", "")}
        if r.status_code == 403:
            sys.exit("the SEC refused the request (403). Check SEC_USER_AGENT in .env, it has to hold an email.")
    sys.exit(f"no answer from the SEC for cik {cik} after four tries. Run it again, what is fetched so far is kept.")


def sic_codes(ciks) -> pd.DataFrame:
    """cik -> sic, sic_description, name. Asked from the SEC once per company and kept in data/raw/sic.csv.
    Saved every 50 companies, so a run that stops halfway goes on from where it was"""
    cols = ["cik", "sic", "sic_description", "name"]
    have = pd.read_csv(SIC_FILE) if SIC_FILE.exists() else pd.DataFrame(columns=cols)
    todo = sorted(set(int(k) for k in ciks) - set(have.cik))
    if todo and not SEC_USER_AGENT:
        sys.exit("SEC_USER_AGENT is not set. Copy .env.example to .env and put your name and email in it, "
                 "the SEC blocks requests without it.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = have.to_dict("records")
    try:
        for i, cik in enumerate(todo):
            rows.append(ask_sec(cik))
            time.sleep(0.15)                 # the SEC allows 10 a second, stay well under
            if (i + 1) % 50 == 0:
                pd.DataFrame(rows, columns=cols).to_csv(SIC_FILE, index=False)
                print(f"  sic codes: {i + 1} of {len(todo)}")
    finally:
        if todo:
            pd.DataFrame(rows, columns=cols).to_csv(SIC_FILE, index=False)
    return pd.DataFrame(rows, columns=cols).set_index("cik")


def sic_to_sector(sic, table) -> str:
    if pd.isna(sic):
        return ""
    for lo, hi, sector in table:
        if lo <= int(sic) <= hi:
            return sector
    return ""


def main():
    # which price series is which company in which years: the insider project's own check
    pm = pd.read_csv(PRICE_DIR / "price_map.csv")
    pm = pm[pm.valid]
    c = pm.groupby(["ticker", "cik"]).year.agg(first_year="min", last_year="max").reset_index()
    c["cik"] = c.cik.astype(int)

    table = pd.read_csv(MANUAL_DIR / "sic_sectors.csv", comment="#")
    bad = set(table.sector) - set(SECTORS)
    if bad:
        sys.exit(f"sic_sectors.csv has sectors that are not GICS sectors: {bad}")
    table = list(table.itertuples(index=False, name=None))

    sic = sic_codes(c.cik)
    c["sic"] = c.cik.map(sic.sic)
    c["sic_description"] = c.cik.map(sic.sic_description)
    c["sic_sector"] = [sic_to_sector(s, table) for s in c.sic]

    # today's list is about the company that has the ticker now: the last one in the price data, plus its own
    # earlier SEC numbers (Alphabet was Google Inc., Disney got a new one in 2019), which universe.csv marks.
    # A different company that had the ticker before gets its SIC sector
    today = todays_members()
    u = pd.read_csv(PRICE_DIR / "universe.csv").dropna(subset=["cik"])
    same = u[u.how.isin(["same ticker, earlier cik", "manual"])]
    same = set(zip(same.ticker, same.cik.astype(int)))
    latest = c.sort_values(["last_year", "first_year"]).groupby("ticker").cik.last()
    on_list = pd.Series([k == latest[tk] or (tk, k) in same for tk, k in zip(c.ticker, c.cik)], index=c.index)
    on_list &= c.ticker.isin(today.index)
    c["sector"] = c.ticker.map(today.sector).where(on_list, c.sic_sector)
    c["sub"] = c.ticker.map(today["sub"]).where(on_list, "")
    c["source"] = ["list" if a else ("sic" if s else "") for a, s in zip(on_list, c.sector)]
    # the few where the SIC code is missing or clearly wrong
    hand = pd.read_csv(MANUAL_DIR / "sector_by_hand.csv", comment="#")
    for h in hand.itertuples():
        row = (c.ticker == h.ticker) & (c.cik == h.cik) & ~on_list
        if not row.any():
            print(f"  sector_by_hand.csv: {h.ticker} {h.cik} is not a company that left, line not used")
        c.loc[row, ["sector", "source"]] = [h.sector, "hand"]

    # how good is the SIC table? Today's members have both
    chk = c[on_list & (c.sic_sector != "")]
    hit = (chk.sic_sector == chk.sector).mean()
    print(f"SIC table against today's list: {hit:.0%} of {len(chk)} companies in the right sector")
    by = chk.assign(hit=chk.sic_sector == chk.sector).groupby("sector").hit.agg(["mean", "size"])
    print((by.assign(mean=by["mean"].round(2))).to_string())
    miss = chk[chk.sic_sector != chk.sector][["ticker", "sic", "sic_description", "sic_sector", "sector"]]
    miss.to_csv(DATA_DIR / "sic_misses.csv", index=False)

    c["sic"] = pd.to_numeric(c.sic, errors="coerce").astype("Int64")
    out = c[["ticker", "cik", "first_year", "last_year", "sector", "sub", "source", "sic", "sic_description"]]
    out.to_csv(DATA_DIR / "classes.csv", index=False)
    print(f"classes.csv: {len(out)} price series, {int((out.source == 'list').sum())} from today's list, "
          f"{int((out.source == 'sic').sum())} from the SIC code, {int((out.source == 'hand').sum())} by hand, "
          f"{int((out.source == '').sum())} without a sector")
    left = out[out.source == ""]
    if len(left):
        print("  no sector: " + " ".join(left.ticker))
    print(out[out.source != "list"].groupby("sector").size().to_string())


if __name__ == "__main__":
    main()
