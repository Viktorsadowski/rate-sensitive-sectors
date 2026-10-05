#!/usr/bin/env python3
"""
Which sectors are rate-sensitive: the numbers and the charts.

  data/results.csv     every rate beta: level (sector or sub), bucket, rate, period, spec, beta, se, t,
                       beta to the market, weeks (days for the daily spec), stocks
  data/order.csv       the sub-industries in order: sector, sub, beta, se, the beta pulled toward its sector,
                       and its place in the nested and in the flat order
  data/rolling.csv     sector betas over five-year windows
  data/stability.csv   does an order made on 2006-2016 hold in 2017-2026
  data/windows.csv     which window is the best guess of the next years' betas: 3, 5, 10 years or everything
  figures/*.png

The rate beta of a bucket: its weekly return regressed on the weekly return of the average S&P 500 stock and the
weekly change in a yield. The coefficient on the yield is the beta, in percent per 1 point rise in the yield at an
unchanged market. Weeks end on Wednesday. Standard errors are Newey-West with 4 lags.

  python src/analysis.py
"""

import textwrap
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from config import (BENCHMARK, BLUE, DATA_DIR, FIG_DIR, FIRST_YEAR, GRID, INK, LIGHT_GREY, MAIN_RATE, MIN_WEEKS,
                    MUTED, ORANGE, PRICE_DIR, SPLIT, SURFACE)

LIGHT_BLUE = "#9dbfeb"       # BLUE at 45%
ALL = f"{FIRST_YEAR}-2026"
LONG_WEEKS = (4, 13)         # the longer return horizons that are checked, in weeks
AHEAD = (3, 5)               # the window test: how many years ahead the beta is compared with
RATES = {"y10": "10-year yield", "y2": "2-year yield", "y3m": "3-month bill", "r10": "10-year real yield",
         "be10": "10-year breakeven inflation"}


# ── the weekly panel ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_prices():
    """days, total return index per ticker, who was a member on which day, and the yields on those days"""
    p = pd.read_parquet(PRICE_DIR / "prices.parquet")
    p["date"] = pd.to_datetime(p["date"])
    days = pd.DatetimeIndex(sorted(p.loc[p.ticker == BENCHMARK, "date"].unique()))
    adj = p[p.date.isin(days)].pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
    adj = adj.reindex(days)

    # members, and only the years where the price series is the company's own (the insider project's rule)
    sp = pd.read_csv(PRICE_DIR / "sp500_members.csv", parse_dates=["start", "end"])
    pm = pd.read_csv(PRICE_DIR / "price_map.csv")
    member = pd.DataFrame(False, index=days, columns=adj.columns)
    for s in sp.itertuples():
        if s.ticker in member.columns:
            end = s.end if pd.notna(s.end) else days[-1] + pd.Timedelta(days=1)
            member.loc[(days >= s.start) & (days < end), s.ticker] = True
    member[[tk for tk in member.columns if tk not in set(pm.ticker[pm.valid])]] = False
    for (tk, y), v in pm.groupby(["ticker", "year"]).valid.max().items():
        if not v and tk in member.columns:
            member.loc[days.year == y, tk] = False

    y = pd.read_csv(DATA_DIR / "yields.csv", parse_dates=["date"]).set_index("date")
    y["slope"] = y.y10 - y.y2
    y["be10"] = y.y10 - y.r10
    return days, adj, member, y.reindex(days).ffill()


def weekly_panel():
    """
    ret: weekly return per stock (weeks x tickers), only for stocks that were S&P 500 members when the week started.
    dy: weekly change in every rate. spy: weekly return of SPY. All in percent or percentage points.
    """
    days, adj, member, y = load_prices()
    # weeks end on wednesday, or on the last trading day before it. A week that has not ended yet is dropped
    last_day = adj.index.to_series().resample("W-WED").last()
    done = last_day.notna() & (last_day.index <= days[-1])
    wk = adj.resample("W-WED").last()[done.values]
    mem = member.resample("W-WED").last()[done.values]
    wk.index = mem.index = last_day[done].values
    ret = wk.pct_change(fill_method=None).where(mem.shift(1, fill_value=False)) * 100
    spy = wk[BENCHMARK].pct_change() * 100
    keep = ret.index.year >= FIRST_YEAR
    ret, spy = ret[keep].drop(columns=BENCHMARK), spy[keep]
    yw = y.reindex(wk.index)
    dy = yw.diff()[keep]
    return ret, dy, spy, yw[keep]


def daily_panel():
    """the same with one row per trading day, for the check with daily returns: ret and dy"""
    days, adj, member, y = load_prices()
    ret = adj.pct_change(fill_method=None).where(member.shift(1, fill_value=False)) * 100
    keep = ret.index.year >= FIRST_YEAR
    return ret[keep].drop(columns=BENCHMARK), y.diff()[keep]


def class_of(ret: pd.DataFrame, col: str) -> pd.DataFrame:
    """sector (or sub-industry) of every stock in every week. A ticker that was two companies gets the class of
    the one whose years are closest"""
    c = pd.read_csv(DATA_DIR / "classes.csv", keep_default_na=False)
    c = c[c[col] != ""]
    years = ret.index.year.values
    out = pd.DataFrame("", index=ret.index, columns=ret.columns)
    for tk, g in c.groupby("ticker"):
        if tk not in out.columns:
            continue
        if len(g) == 1:
            out[tk] = g[col].iloc[0]
            continue
        # distance from the year to each company's first..last year, 0 inside
        dist = np.array([np.maximum(0, np.maximum(r.first_year - years, years - r.last_year)) for r in g.itertuples()])
        out[tk] = g[col].values[dist.argmin(axis=0)]
    return out


def buckets(ret: pd.DataFrame, cls: pd.DataFrame):
    """weekly return of the average stock in every bucket, and how many stocks it had that week"""
    names = sorted(set(np.unique(cls.values)) - {""})
    R = pd.DataFrame({b: ret.where(cls == b).mean(axis=1) for b in names})
    N = pd.DataFrame({b: ret.where(cls == b).notna().sum(axis=1) for b in names})
    return R, N


# ── statistics ─────────────────────────────────────────────────────────────────

def ols_nw(y, X, lags=4):
    """ols with newey-west standard errors. Returns coefficients, standard errors, number of weeks"""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    k = X.shape[1]
    # too few weeks, or a rate that did not move in them
    if len(y) < MIN_WEEKS or np.linalg.matrix_rank(X) < k:
        return np.full(k, np.nan), np.full(k, np.nan), len(y)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for lag in range(1, lags + 1):
        G = Xe[lag:].T @ Xe[:-lag]
        S += (1 - lag / (lags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(XtXi @ S @ XtXi)), len(y)


def rate_betas(R, mkt, dy, rate=MAIN_RATE, sel=None, market=True, lags=4) -> pd.DataFrame:
    """one row per bucket: beta to the rate, its standard error and t, beta to the market, weeks"""
    sel = np.ones(len(R), bool) if sel is None else np.asarray(sel)
    rows = []
    for b in R.columns:
        cols = [np.ones(sel.sum())] + ([mkt.values[sel]] if market else []) + [dy[rate].values[sel]]
        coef, se, n = ols_nw(R[b].values[sel], np.column_stack(cols), lags)
        rows.append({"bucket": b, "beta": coef[-1], "se": se[-1], "t": coef[-1] / se[-1],
                     "mkt_beta": coef[1] if market else np.nan, "weeks": n})
    return pd.DataFrame(rows).set_index("bucket")


def two_rates(R, mkt, dy, a, b) -> pd.DataFrame:
    """two rates in the same regression, e.g. the 2-year and the slope: which one is it"""
    rows = []
    for name in R.columns:
        X = np.column_stack([np.ones(len(R)), mkt.values, dy[a].values, dy[b].values])
        coef, se, n = ols_nw(R[name].values, X)
        rows.append({"bucket": name, f"b_{a}": coef[2], f"t_{a}": coef[2] / se[2], f"b_{b}": coef[3],
                     f"t_{b}": coef[3] / se[3], "weeks": n})
    return pd.DataFrame(rows).set_index("bucket")


def rank_corr(a, b) -> float:
    """spearman: the correlation of the ranks"""
    return float(pd.Series(np.asarray(a, float)).rank().corr(pd.Series(np.asarray(b, float)).rank()))


def over(x: pd.Series, k: int, how: str) -> pd.Series:
    """a weekly series over the last k weeks, one value per week so they overlap: the compounded return ("prod")
    or the summed change ("sum")"""
    return x.rolling(k).sum() if how == "sum" else (np.exp(np.log1p(x / 100).rolling(k).sum()) - 1) * 100


def pull_to_sector(sub: pd.DataFrame) -> pd.Series:
    """
    A sub-industry with two stocks has a noisy beta. Pull it toward its sector: the less sure the beta (big se)
    the more of the sector it gets. tau2 is how much the true betas spread inside a sector, what is left of the
    spread after the noise is taken out. One tau2 for all sectors.
    """
    dev = sub.beta - sub.sector_beta
    tau2 = max(0.0, np.nanvar(dev) - np.nanmean(sub.se ** 2))
    w = tau2 / (tau2 + sub.se ** 2)
    return sub.sector_beta + w * dev


def results():
    ret, dy, spy, yw = weekly_panel()
    mkt = ret.mean(axis=1)                       # the average member that week, with or without a sector
    first, second = ret.index.year < SPLIT, ret.index.year >= SPLIT
    periods = {ALL: None, f"{FIRST_YEAR}-{SPLIT - 1}": first, f"{SPLIT}-2026": second}
    S, NS = buckets(ret, class_of(ret, "sector"))
    U, NU = buckets(ret, class_of(ret, "sub"))
    print(f"{len(ret)} weeks, {ret.index[0].date()} to {ret.index[-1].date()}. Stocks per week: "
          f"{int(ret.notna().sum(axis=1).median())}, with a sector {int(NS.sum(axis=1).median())}, "
          f"with a sub-industry {int(NU.sum(axis=1).median())}")

    rows = []

    def add(t, level, rate, period, spec, N):
        t = t.reset_index().assign(level=level, rate=rate, period=period, spec=spec)
        t["stocks"] = t.bucket.map(N.replace(0, np.nan).median()).values
        rows.append(t)

    # the base: every rate, every period, sectors and sub-industries
    for level, R, N in (("sector", S, NS), ("sub", U, NU)):
        for rate in RATES:
            for name, sel in periods.items():
                add(rate_betas(R, mkt, dy, rate, sel), level, rate, name, "base", N)
    # other ways to measure it, sectors and the main rate only
    add(rate_betas(S, spy, dy), "sector", MAIN_RATE, ALL, "SPY as the market", NS)
    add(rate_betas(S, mkt, dy, market=False), "sector", MAIN_RATE, ALL, "market not held fixed", NS)
    calm = ~ret.index.year.isin([2008, 2009, 2020])
    add(rate_betas(S, mkt, dy, sel=calm), "sector", MAIN_RATE, ALL, "without 2008, 2009 and 2020", NS)
    # other return horizons. Daily, and 4 and 13 weeks: those overlap, so the standard errors get more lags
    dret, ddy = daily_panel()
    SD, _ = buckets(dret, class_of(dret, "sector"))
    add(rate_betas(SD, dret.mean(axis=1), ddy, lags=5), "sector", MAIN_RATE, ALL, "daily returns", NS)
    for k in LONG_WEEKS:
        Sk = pd.DataFrame({c: over(S[c], k, "prod") for c in S})
        add(rate_betas(Sk, over(mkt, k, "prod"), pd.DataFrame({MAIN_RATE: over(dy[MAIN_RATE], k, "sum")}),
                       lags=2 * k), "sector", MAIN_RATE, ALL, f"{k}-week returns", NS)
    # only today's members: what the sectors look like without the companies that left
    c = pd.read_csv(DATA_DIR / "classes.csv", keep_default_na=False)
    stay = sorted(set(c.ticker[c.source == "list"]) & set(ret.columns))
    Sl, NSl = buckets(ret[stay], class_of(ret, "sector")[stay])
    for name, sel in periods.items():
        add(rate_betas(Sl, mkt, dy, sel=sel), "sector", MAIN_RATE, name, "today's members only", NSl)
    # the market itself on the yield
    for name, sel in periods.items():
        add(rate_betas(pd.DataFrame({"the average stock": mkt}), mkt, dy, sel=sel, market=False), "market",
            MAIN_RATE, name, "market not held fixed", pd.DataFrame({"the average stock": ret.notna().sum(axis=1)}))

    res = pd.concat(rows, ignore_index=True)
    res = res[["level", "bucket", "rate", "period", "spec", "beta", "se", "t", "mkt_beta", "weeks", "stocks"]]
    res.to_csv(DATA_DIR / "results.csv", index=False)

    # two rates at once
    pair = two_rates(S, mkt, dy, "y2", "slope").join(two_rates(S, mkt, dy, "r10", "be10").drop(columns="weeks"))
    pair.to_csv(DATA_DIR / "two_rates.csv")

    # sector betas over five-year windows, one step a year
    roll = {}
    for y in range(FIRST_YEAR + 4, ret.index.year.max() + 1):
        sel = (ret.index.year > y - 5) & (ret.index.year <= y)
        roll[y] = rate_betas(S, mkt, dy, sel=sel).beta
    roll["market"] = None
    roll = pd.DataFrame({k: v for k, v in roll.items() if v is not None})
    roll.loc["the average stock"] = [rate_betas(pd.DataFrame({"m": mkt}), mkt, dy, market=False,
                                                sel=(ret.index.year > y - 5) & (ret.index.year <= y)).beta.iloc[0]
                                     for y in roll.columns]
    roll.to_csv(DATA_DIR / "rolling.csv")

    # the order of the sub-industries
    pick = lambda level, period: res[(res.level == level) & (res.rate == MAIN_RATE) & (res.period == period) &
                                     (res.spec == "base")].set_index("bucket")
    sec, sub = pick("sector", ALL), pick("sub", ALL)
    sector_of = c[c.source == "list"].drop_duplicates("sub").set_index("sub").sector
    o = sub[["beta", "se", "t", "weeks", "stocks"]].copy()
    o.insert(0, "sector", sector_of.reindex(o.index))
    # tickers, not price series: a company that changed its SEC number has two series and is one stock
    o["stocks_today"] = c[c.source == "list"].drop_duplicates("ticker").groupby("sub").size().reindex(o.index)
    o["sector_beta"] = o.sector.map(sec.beta)
    o = o.dropna(subset=["beta"])
    o["pulled"] = pull_to_sector(o)
    for p in periods:
        if p != ALL:
            o[f"beta_{p}"] = pick("sub", p).beta.reindex(o.index)
    # nested: sectors by their beta, sub-industries by their pulled beta inside the sector. flat: one axis
    o = o.sort_values(["sector_beta", "pulled"])
    o["nested_place"] = np.arange(1, len(o) + 1)
    o["flat_place"] = o.pulled.rank(method="first").astype(int)
    o.index.name = "sub"
    o.to_csv(DATA_DIR / "order.csv")

    stability(res, o, periods)
    win = window_test({"sector": S, "sub-industry": U}, mkt, dy)
    return res, o, roll, pair, win, (ret, dy, mkt, yw)


def stability(res, o, periods):
    """does an order made on the first half hold in the second half? Rank correlations, signed and by size"""
    p1, p2 = [p for p in periods if p != ALL]
    pick = lambda level, period: res[(res.level == level) & (res.rate == MAIN_RATE) & (res.period == period) &
                                     (res.spec == "base")].set_index("bucket")
    rows = []
    a, b = pick("sector", p1), pick("sector", p2)
    rows.append({"what": "sectors, signed", "n": len(a), "rank_corr": rank_corr(a.beta, b.beta)})
    rows.append({"what": "sectors, size", "n": len(a), "rank_corr": rank_corr(a.beta.abs(), b.beta.abs())})
    u1, u2 = pick("sub", p1), pick("sub", p2)
    both = u1.index[u1.beta.notna() & u2.beta.reindex(u1.index).notna()]
    u1, u2 = u1.loc[both], u2.loc[both]
    sec = o.sector.reindex(both)
    rows.append({"what": "sub-industries, signed", "n": len(both), "rank_corr": rank_corr(u1.beta, u2.beta)})
    rows.append({"what": "sub-industries, size", "n": len(both), "rank_corr": rank_corr(u1.beta.abs(), u2.beta.abs())})
    # three orders made on the first half, each against the second-half betas of the sub-industries
    s1 = sec.map(a.beta)
    u1 = u1.assign(sector=sec, sector_beta=s1)
    u1["pulled"] = pull_to_sector(u1)
    nested = u1.sort_values(["sector_beta", "pulled"]).assign(place=lambda d: np.arange(len(d))).place.reindex(both)
    for name, x in (("sector order only", s1), ("nested: sector, then sub-industry inside", nested),
                    ("flat: sub-industries on one axis", u1.pulled)):
        rows.append({"what": f"made on {p1}, against {p2}: {name}", "n": len(both),
                     "rank_corr": rank_corr(x, u2.beta)})
    st = pd.DataFrame(rows)
    st.to_csv(DATA_DIR / "stability.csv", index=False)
    print(st.round(2).to_string(index=False))


def window_test(levels: dict, mkt, dy) -> pd.DataFrame:
    """
    Which window is the best guess of the beta that comes next? At the end of every year the betas are estimated
    on the last 3, 5 and 10 years and on everything since the start, and each estimate is compared with the betas
    measured over the following years: rank correlation, error (root mean square) and share with the right sign.
    Only the years where all four windows exist, so they are compared on the same ground.
    """
    years = mkt.index.year
    everything = f"all since {FIRST_YEAR}"
    rows = []
    for level, R in levels.items():
        beta = lambda y0, y1: rate_betas(R, mkt, dy, sel=(years >= y0) & (years <= y1)).beta
        for ahead in AHEAD:
            for end in range(FIRST_YEAR + 9, years.max()):
                # the years after have to be there: the last year in the data is not a full one
                if ((years > end) & (years <= end + ahead)).sum() < max(MIN_WEEKS, 0.9 * 52 * ahead):
                    continue
                after = beta(end + 1, end + ahead)
                est = {"3 years": beta(end - 2, end), "5 years": beta(end - 4, end), "10 years": beta(end - 9, end),
                       everything: beta(FIRST_YEAR, end)}
                ok = after.notna() & pd.concat(est, axis=1).notna().all(axis=1)
                for name, e in est.items():
                    rows.append({"level": level, "ahead": ahead, "end": end, "window": name, "buckets": int(ok.sum()),
                                 "rank_corr": rank_corr(e[ok], after[ok]),
                                 "error": float(np.sqrt(((e[ok] - after[ok]) ** 2).mean())),
                                 "right_sign": float((np.sign(e[ok]) == np.sign(after[ok])).mean())})
    win = pd.DataFrame(rows)
    win.to_csv(DATA_DIR / "windows.csv", index=False)
    t = win.groupby(["level", "ahead", "window"], sort=False)[["rank_corr", "error", "right_sign"]].mean()
    print(t.round(2).to_string())
    return win


# ── charts ─────────────────────────────────────────────────────────────────────

# Calibri on windows, its metric clone Carlito on linux: the same widths, so a chart looks the same on both
_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Calibri", "Carlito") if f in _fonts), "DejaVu Sans")
FS, FS_NUM, FS_PANEL, FS_HEAD, FS_SUB = 9.5, 9, 11, 13.5, 10      # text sizes: axes, numbers, panel title, headline, the lines under it
UNIT = "return in % for a 1 point rise in the yield, at an unchanged market"


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=FS, length=3)


def new(w=8.4, h=4.0, **kw):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def title(ax, t):
    """the title of one panel"""
    ax.set_title(t, loc="left", fontsize=FS_PANEL, color=INK, pad=7)


def head(fig, t, sub, panels=True):
    """
    The headline of a chart and, under it, what is plotted in a sentence or two: the sample, the unit and how to
    read it. With these a chart can be read on its own, without the report around it. panels: the axes have
    titles of their own, leave room for them.
    """
    fig.canvas.draw()
    # flush with the left edge of everything that is drawn (axis names, tick labels), not with the axes
    boxes = [a.get_tightbbox().transformed(fig.transFigure.inverted()) for a in fig.axes]
    left, right = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    top = max(a.get_position().y1 for a in fig.axes)
    w, h = fig.get_size_inches()
    sub, t = sub.replace("$", r"\$"), t.replace("$", r"\$")

    def too_wide(width):
        probe = fig.text(0, 0, textwrap.fill(sub, width), fontsize=FS_SUB)
        wide = probe.get_window_extent().width > (right - left) * fig.bbox.width
        probe.remove()
        return wide

    # as many letters on a line as fit over the chart, measured
    width = 220
    while too_wide(width):
        width -= 2
    # then as narrow as it gets with the same number of lines: no last line of one word
    n = len(textwrap.wrap(sub, width))
    while len(textwrap.wrap(sub, width - 1)) == n:
        width -= 1
    sub = textwrap.fill(sub, width)
    y = top + (0.36 if panels else 0.12) / h
    fig.text(left, y, sub, fontsize=FS_SUB, color=MUTED, ha="left", va="bottom", linespacing=1.3)
    fig.text(left, y + (0.19 * (sub.count("\n") + 1) + 0.07) / h, t, fontsize=FS_HEAD, color=INK, ha="left",
             va="bottom")


def xlab(ax, t):
    ax.set_xlabel(t, fontsize=FS, color=MUTED)


def ylab(ax, t):
    ax.set_ylabel(t, fontsize=FS, color=MUTED)


def legend_below(fig, ax, **kw):
    """the names in one row under the chart"""
    fig.canvas.draw()
    low = min(a.get_tightbbox().transformed(fig.transFigure.inverted()).y0 for a in fig.axes)
    kw = {"handletextpad": 0.3, "columnspacing": 2.2, **kw}
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, low - 0.01), ncol=4,
               frameon=False, fontsize=FS, labelcolor=MUTED, **kw)


def ends(ax, left="hurt by rising yields", right="helped by rising yields"):
    """what the two ends of the axis mean. Bottom left and top right: the corners that are empty when the
    rows are sorted from most hurt to most helped"""
    ax.annotate(left, (0, 0), xycoords="axes fraction", xytext=(4, 3), textcoords="offset points", fontsize=FS_NUM,
                color=MUTED, va="bottom")
    ax.annotate(right, (1, 1), xycoords="axes fraction", xytext=(-4, -3), textcoords="offset points",
                fontsize=FS_NUM, color=MUTED, va="top", ha="right")


def dots(ax, x, y, t, col, label, size=40):
    """filled when the t-value is beyond 2, hollow when it is not"""
    sig = (t.abs() > 2).values
    # one name in the legend: on the filled ones, or on the hollow ones when none is filled
    ax.scatter(x[sig], y[sig], s=size, color=col, zorder=4, label=label if sig.any() else None)
    ax.scatter(x[~sig], y[~sig], s=size, facecolor=SURFACE, edgecolor=col, linewidth=1.5, zorder=4,
               label=None if sig.any() else label)


def sector_table(res, rate=MAIN_RATE, period=ALL, spec="base"):
    return res[(res.level == "sector") & (res.rate == rate) & (res.period == period) & (res.spec == spec)] \
        .set_index("bucket")


def fig_yields(yw, roll):
    """the setting: the rates themselves, and what the stock market as a whole did when the 10-year moved"""
    fig, axes = new(10.4, 3.2, ncols=2, gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    style(ax)
    ax.plot(yw.index, yw.y2, color=LIGHT_BLUE, lw=1.6)
    ax.plot(yw.index, yw.y10, color=BLUE, lw=1.8)
    last = yw.dropna(subset=["y10", "y2"]).iloc[-1]
    for v, lab, col in ((last.y10, "10-year", BLUE), (last.y2, "2-year", "#5c93dd")):
        ax.annotate(lab, (yw.index[-1], v), xytext=(5, 0), textcoords="offset points", fontsize=FS_NUM, color=col,
                    va="center", annotation_clip=False)
    ax.set_xlim(yw.index[0], yw.index[-1])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(0, None)
    title(ax, "Treasury yields")

    ax = axes[1]
    style(ax)
    m = roll.loc["the average stock"]
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.plot(m.index.astype(int), m.values, color=INK, lw=1.8, marker="o", ms=3.5, zorder=3)
    ax.set_xticks([y for y in m.index.astype(int) if y % 4 == 2])
    xlab(ax, "five years ending in")
    ylab(ax, "% per 1 point rise in the 10-year")
    title(ax, "The average stock when the 10-year rises")
    fig.subplots_adjust(wspace=0.32)
    head(fig, "Rates since 2006, and how the stock market took them",
         "Left: weekly values. Right: the return of the average S&P 500 stock regressed on the weekly change in the "
         "10-year yield, over five-year windows. Stocks used to rise in the weeks when yields rose, in the last "
         "three windows they fell. This is why every sector is measured against the market.")
    save(fig, "yields.png")


def fig_sectors(res):
    """the main chart: the sectors in order, the whole sample with its interval and the two halves"""
    a = sector_table(res).sort_values("beta")
    p1, p2 = f"{FIRST_YEAR}-{SPLIT - 1}", f"{SPLIT}-2026"
    h1, h2 = sector_table(res, period=p1).loc[a.index], sector_table(res, period=p2).loc[a.index]
    fig, ax = new(8.4, 3.7)
    style(ax, "x")
    y = np.arange(len(a))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.errorbar(a.beta, y, xerr=1.96 * a.se, fmt="none", ecolor=MUTED, elinewidth=1.2, zorder=3)
    ax.scatter(a.beta, y, s=46, color=INK, zorder=5, label=f"{FIRST_YEAR} to 2026, with 95% interval")
    ax.scatter(h1.beta, y, s=40, facecolor=SURFACE, edgecolor=BLUE, linewidth=1.5, zorder=4, label=p1.replace("-", " to "))
    ax.scatter(h2.beta, y, s=40, color=BLUE, zorder=4, label=p2.replace("-", " to "))
    ax.set_yticks(y)
    ax.set_yticklabels(a.index, fontsize=FS, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(a) - 0.3)
    xlab(ax, UNIT.replace("the yield", "the 10-year yield"))
    ends(ax)
    legend_below(fig, ax)
    head(fig, "Rate sensitivity by sector: two ends and a wide middle",
         "Weekly return of the average stock in each S&P 500 sector, regressed on the return of the average S&P 500 "
         "stock and the change in the 10-year Treasury yield. Companies that were in the index, for the weeks they "
         "were in it.", panels=False)
    save(fig, "sectors.png")


def fig_rolling(roll, res):
    """one small panel per sector: its beta over five-year windows. Same scale in all"""
    order = list(sector_table(res).sort_values("beta").index)
    fig, axes = new(10.4, 4.3, nrows=3, ncols=4, sharex=True, sharey=True)
    years = roll.columns.astype(int)
    for ax, name in zip(axes.ravel(), order + [None]):
        if name is None:
            ax.axis("off")
            continue
        style(ax)
        v = roll.loc[name].values.astype(float)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.fill_between(years, 0, v, where=v >= 0, color=BLUE, alpha=0.18, lw=0, interpolate=True)
        ax.fill_between(years, 0, v, where=v <= 0, color=ORANGE, alpha=0.18, lw=0, interpolate=True)
        ax.plot(years, v, color=INK, lw=1.6, zorder=3)
        ax.set_title(name, loc="left", fontsize=FS + 0.5, color=INK, pad=4)
        ax.set_xticks([2010, 2014, 2018, 2022, 2026])
        ax.set_xticklabels(["2010", "", "2018", "", "2026"])
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax.tick_params(axis="x", length=2)
    # the last row has three panels, the one above the empty corner shows its years too
    axes[1, 3].tick_params(labelbottom=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.42)
    ylab(axes[1, 0], "% per 1 point rise in the 10-year yield")
    head(fig, "The same measure over five-year windows",
         "Rate beta of each sector over the five years ending in the year on the axis, market held fixed. Orange: hurt "
         "by rising yields, blue: helped. Utilities and Real Estate never change side, Financials do in 2014, and "
         "Energy jumps in 2022.")
    save(fig, "rolling.png")


def fig_horizons(res):
    """the same betas with returns over a day, a week, four weeks and a quarter. One row per sector, the grey
    line runs from the smallest to the largest of the four"""
    a = sector_table(res).sort_values("beta")
    specs = [("daily returns", "1 day"), ("base", "1 week")] + [(f"{k}-week returns", f"{k} weeks") for k in LONG_WEEKS]
    shades = ("#bcd4f3", "#6fa3e3", BLUE, "#143a6b")          # light to dark: short to long
    b = pd.DataFrame({lab: sector_table(res, spec=spec).beta for spec, lab in specs}).loc[a.index]
    fig, ax = new(8.4, 3.7)
    style(ax, "x")
    y = np.arange(len(a))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.hlines(y, b.min(axis=1), b.max(axis=1), color=LIGHT_GREY, lw=1.4, zorder=2)
    for lab, col in zip(b.columns, shades):
        ax.scatter(b[lab], y, s=44, color=col, edgecolor=SURFACE, linewidth=0.7, zorder=4, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(a.index, fontsize=FS, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(a) - 0.3)
    xlab(ax, UNIT.replace("the yield", "the 10-year yield"))
    ends(ax)
    legend_below(fig, ax)
    corr = ", ".join(f"{rank_corr(b['1 week'], b[c]):.2f}" for c in b.columns if c != "1 week")
    head(fig, "Daily, weekly or quarterly returns: the same order, larger numbers",
         f"The same regression with returns over one day, one week, {LONG_WEEKS[0]} weeks and {LONG_WEEKS[1]} weeks "
         f"(the longer ones overlap), {FIRST_YEAR} to 2026. Rank correlation with the weekly order: {corr}.",
         panels=False)
    save(fig, "horizons.png")


def fig_windows(win):
    """the window test in two panels: how far off each window is, and how well it ranks, for the betas of the
    following five years. Sectors and sub-industries"""
    ahead = max(AHEAD)
    w = win[win.ahead == ahead]
    t = w.groupby(["level", "window"], sort=False)[["error", "rank_corr"]].mean()
    names = list(w.window.unique())
    x = np.arange(len(names))
    fig, axes = new(9.6, 3.3, ncols=2)
    for ax, col, ttl, fmt in zip(axes, ("error", "rank_corr"),
                                 (f"How far off the beta of the next {ahead} years", "How well it ranks them"),
                                 ("{:.1f}", "{:.2f}")):
        style(ax)
        for level, c, lab in (("sector", BLUE, "sectors"), ("sub-industry", ORANGE, "sub-industries")):
            v = t.loc[level].loc[names, col].values
            ax.plot(x, v, color=c, lw=2, marker="o", ms=5.5, zorder=3, label=lab)
            for xi, vi in zip(x, v):
                ax.annotate(fmt.format(vi), (xi, vi), xytext=(0, 7), textcoords="offset points", ha="center",
                            fontsize=FS_NUM, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlim(-0.35, len(names) - 0.65)
        ax.set_ylim(0, 1 if col == "rank_corr" else t[col].max() * 1.22)
        xlab(ax, "window the beta is estimated on")
        title(ax, ttl)
    ylab(axes[0], "error, in points of beta")
    ylab(axes[1], "rank correlation")
    fig.subplots_adjust(wspace=0.24)
    legend_below(fig, axes[0])
    ends_ = sorted(w.end.unique())
    head(fig, "The longest window misses least and ranks about as well",
         f"At the end of every year from {ends_[0]} to {ends_[-1]} the rate betas are estimated on four windows and "
         f"compared with the betas measured over the following {ahead} years. Averages over those years. Left: root "
         "mean square error. Right: rank correlation across the buckets.")
    save(fig, "windows.png")


def fig_parts(pair, res):
    """two rates at once. Left: the 2-year and the slope of the curve. Right: the real yield and expected inflation"""
    order = sector_table(res).sort_values("beta").index
    fig, axes = new(10.4, 3.4, ncols=2, sharey=True)
    y = np.arange(len(order))[::-1]
    sets = ((("y2", BLUE, "the 2-year yield"), ("slope", ORANGE, "the slope, 10-year minus 2-year")),
            (("r10", BLUE, "the 10-year real yield"), ("be10", ORANGE, "expected inflation, 10-year breakeven")))
    for ax, pairs, ttl in zip(axes, sets, ("Short end or long end", "Real yield or inflation")):
        style(ax, "x")
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        for k, (r, col, lab) in enumerate(pairs):
            dots(ax, pair.loc[order, f"b_{r}"], y + (0.5 - k) * 0.26, pair.loc[order, f"t_{r}"], col, lab)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.tick_params(axis="y", length=0)
        ax.legend(frameon=False, fontsize=FS, labelcolor=MUTED, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                  ncol=1, handletextpad=0.3)
        xlab(ax, "return in % for a 1 point rise, at an unchanged market")
        title(ax, ttl)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=FS, color=INK)
    fig.subplots_adjust(wspace=0.06)
    head(fig, "What in the 10-year yield is it?",
         "Two parts of the 10-year yield in the same regression, so each is measured with the other held fixed. "
         f"Filled: t-value beyond 2. S&P 500 sectors, {FIRST_YEAR} to 2026.")
    save(fig, "parts.png")


NAMED = {"Gold": "Gold", "Homebuilding": "Homebuilding", "Data Center REITs": "Data center REITs",
         "Hotel & Resort REITs": "Hotel REITs", "Investment Banking & Brokerage": "Investment banks",
         "Oil & Gas Equipment & Services": "Oil services", "Agricultural Products & Services": "Agricultural products",
         "Oil & Gas Storage & Transportation": "Pipelines", "Health Care Facilities": "Hospitals",
         "Diversified Banks": "Banks", "Water Utilities": "Water utilities", "Semiconductors": "Semiconductors",
         "Commodity Chemicals": "Commodity chemicals"}


def fig_subs(o, res):
    """every sub-industry as a dot in its sector's row: the spread inside a sector"""
    a = sector_table(res).sort_values("beta")
    fig, ax = new(8.8, 4.9)
    style(ax, "x")
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    pos = dict(zip(a.index, np.arange(len(a))[::-1]))
    for sec, g in o.groupby("sector"):
        one = g.stocks_today == 1
        ax.scatter(g.beta[~one], [pos[sec]] * (~one).sum(), s=14 + 9 * g.stocks_today[~one], color=BLUE, alpha=0.75,
                   linewidth=0, zorder=3)
        ax.scatter(g.beta[one], [pos[sec]] * one.sum(), s=22, facecolor=SURFACE, edgecolor=BLUE, linewidth=1.0, zorder=3)
        ax.plot([a.beta[sec]] * 2, [pos[sec] - 0.32, pos[sec] + 0.32], color=INK, lw=1.6, zorder=4)
    for name, lab in NAMED.items():
        if name in o.index:
            ax.annotate(lab, (o.beta[name], pos[o.sector[name]]), xytext=(0, 10), textcoords="offset points",
                        fontsize=FS_NUM - 0.5, color=INK, ha="center", va="center")
    ax.set_yticks(list(pos.values()))
    ax.set_yticklabels(list(pos.keys()), fontsize=FS, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, len(a) - 0.2)
    xlab(ax, UNIT.replace("the yield", "the 10-year yield"))
    head(fig, "Inside the sectors: sub-industries spread as wide as the sectors do",
         f"One dot per GICS sub-industry of today's S&P 500 with enough weeks for a beta, {len(o)} of them, in the row "
         f"of its sector. Bigger dot: more stocks, hollow: one stock. Black line: the sector. {FIRST_YEAR} to 2026.",
         panels=False)
    save(fig, "subs.png")


def fig_stability(o):
    """left: every sub-industry, its beta in the first half against the second. Right: three orders made on the
    first half, tested on the second"""
    p1, p2 = f"{FIRST_YEAR}-{SPLIT - 1}", f"{SPLIT}-2026"
    st = pd.read_csv(DATA_DIR / "stability.csv").set_index("what").rank_corr
    d = o.dropna(subset=[f"beta_{p1}", f"beta_{p2}"])
    fig, axes = new(10.4, 3.9, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    style(ax, "both")
    lim = float(np.ceil(max(d[f"beta_{p1}"].abs().max(), d[f"beta_{p2}"].abs().max())))
    ax.plot([-lim, lim], [-lim, lim], color=LIGHT_GREY, lw=1.2, zorder=1)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.scatter(d[f"beta_{p1}"], d[f"beta_{p2}"], s=10 + 7 * d.stocks_today, color=BLUE, alpha=0.6, linewidth=0, zorder=3)
    # a few names, the text on the side where there is room
    for name, (dx, dy_, ha) in {"Gold": (7, 0, "left"), "Diversified Banks": (-7, 5, "right"),
                                "Investment Banking & Brokerage": (7, -3, "left"),
                                "Oil & Gas Equipment & Services": (7, 0, "left")}.items():
        if name in d.index:
            ax.annotate(NAMED[name], (d[f"beta_{p1}"][name], d[f"beta_{p2}"][name]), xytext=(dx, dy_), ha=ha,
                        textcoords="offset points", fontsize=FS_NUM - 0.5, color=INK, va="center")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    xlab(ax, f"rate beta {p1.replace('-', ' to ')}")
    ylab(ax, f"rate beta {p2.replace('-', ' to ')}")
    title(ax, f"Each sub-industry in the two halves (rank correlation {st['sub-industries, signed']:.2f})")

    ax = axes[1]
    style(ax, "x")
    rows = [("sectors, signed", "Sectors,\nby sign and size"), ("sectors, size", "Sectors,\nby size only"),
            ("sub-industries, signed", "Sub-industries,\nby sign and size"),
            ("sub-industries, size", "Sub-industries,\nby size only")]
    yy = np.arange(len(rows))[::-1]
    v = [st[k] for k, _ in rows]
    ax.barh(yy, v, 0.56, color=[BLUE, LIGHT_BLUE, BLUE, LIGHT_BLUE], zorder=3)
    for yi, vi in zip(yy, v):
        ax.annotate(f"{vi:.2f}", (vi, yi), xytext=(4, 0), textcoords="offset points", fontsize=FS_NUM, color=INK, va="center")
    ax.set_yticks(yy)
    ax.set_yticklabels([lab for _, lab in rows], fontsize=FS, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    xlab(ax, "rank correlation between the two halves")
    title(ax, "How well the order holds")
    fig.subplots_adjust(wspace=0.42)
    head(fig, "Does the order hold? By sign yes, by size no",
         f"Rate betas measured on {p1.replace('-', ' to ')} and on {p2.replace('-', ' to ')} separately. Left: one dot "
         "per sub-industry, bigger with more stocks, on the grey line when it is the same in both. Right: the rank "
         "correlation between the two halves, for an order by sign and size (most hurt to most helped) and for an "
         "order by size alone.")
    save(fig, "stability.png")


def main():
    res, o, roll, pair, win, panel = results()
    pd.set_option("display.width", 200)
    base = res[(res.level == "sector") & (res.spec == "base") & (res.rate == MAIN_RATE)]
    print(base.pivot(index="bucket", columns="period", values="beta").sort_values(ALL).round(2).to_string())
    print(f"results.csv: {len(res):,} rows. order.csv: {len(o)} sub-industries")
    ret, dy, mkt, yw = panel
    fig_yields(yw, roll)
    fig_sectors(res)
    fig_horizons(res)
    fig_rolling(roll, res)
    fig_windows(win)
    fig_parts(pair, res)
    fig_subs(o, res)
    fig_stability(o)


if __name__ == "__main__":
    main()
