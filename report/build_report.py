#!/usr/bin/env python3
"""
Builds report/rate_sensitive_sectors.pdf from the text below + the figures in figures/.

Numbers in the text are from the October 2026 run (prices and yields to 30 September 2026), written by hand.
The table and the figures are read from data/ and figures/, so they follow the data if it changes.

  python src/analysis.py
  python report/build_report.py
"""

import os
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
DATA = Path(os.environ.get("RATES_DATA_DIR", ROOT / "data"))
OUT = HERE / "rate_sensitive_sectors.pdf"
TITLE = "Which sectors are rate-sensitive?"

# windows: Times New Roman + Calibri, linux: their metric clones Liberation Serif + Carlito (same as nba-moneyball)
F = "/usr/share/fonts/truetype"
FONTS = {"Serif": [r"C:\Windows\Fonts\times.ttf", f"{F}/liberation/LiberationSerif-Regular.ttf"],
         "Serif-Bold": [r"C:\Windows\Fonts\timesbd.ttf", f"{F}/liberation/LiberationSerif-Bold.ttf"],
         "Sans": [r"C:\Windows\Fonts\calibri.ttf", f"{F}/crosextra/Carlito-Regular.ttf"],
         "Sans-Bold": [r"C:\Windows\Fonts\calibrib.ttf", f"{F}/crosextra/Carlito-Bold.ttf"]}
for name, paths in FONTS.items():
    path = next((p for p in paths if Path(p).exists()), None)
    if path is None:
        raise SystemExit(f"no font found for {name}, tried {paths}")
    pdfmetrics.registerFont(TTFont(name, path))

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
GRID = colors.HexColor("#e6e5e0")
PANEL = colors.HexColor("#f4f3ef")

W, H = A4
MARGIN = 2.1 * cm
TEXT_W = W - 2 * MARGIN

body = ParagraphStyle("body", fontName="Serif", fontSize=10.3, leading=14.2, textColor=INK, alignment=TA_JUSTIFY,
                      spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.2, leading=14, textColor=INK, spaceBefore=9,
                    spaceAfter=4)
title = ParagraphStyle("title", fontName="Sans-Bold", fontSize=24, leading=28, textColor=INK, alignment=TA_LEFT)
subtitle = ParagraphStyle("subtitle", fontName="Sans", fontSize=13, leading=17, textColor=MUTED, spaceBefore=4)
byline = ParagraphStyle("byline", fontName="Sans", fontSize=10, leading=13, textColor=MUTED, spaceBefore=10)
caption = ParagraphStyle("caption", fontName="Sans", fontSize=8.6, leading=11, textColor=MUTED, spaceBefore=3,
                         spaceAfter=10)
abstract = ParagraphStyle("abstract", parent=body, fontSize=10, leading=13.8)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=10.4, textColor=INK)
cell_head = ParagraphStyle("cell_head", parent=cell, fontName="Sans-Bold", textColor=MUTED)


def P(t, s=body):
    return Paragraph(t.replace("S&P", "S&amp;P"), s)


def bullets(items, lead=None):
    b = [Paragraph(t.replace("S&P", "S&amp;P"), bullet, bulletText="•") for t in items]
    # the line that introduces a list stays on the page of its first point
    return [KeepTogether([P(lead), b[0]])] + b[1:] if lead else b


def figure(path, cap, width=TEXT_W):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, P(cap, caption)])


def table(rows, widths, cap=None):
    data = [[P(str(c), cell_head) for c in rows[0]]] + [[P(str(c), cell) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, GRID),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f6f3")))
    t.setStyle(TableStyle(st))
    parts = [t]
    if cap:
        parts.append(P(cap, caption))
    return KeepTogether(parts)


def box(flow):
    t = Table([[flow]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def on_page(c, doc):
    c.saveState()
    c.setFont("Sans", 8)
    c.setFillColor(MUTED)
    if doc.page > 1:
        c.drawString(MARGIN, H - 1.25 * cm, TITLE)
        c.drawRightString(W - MARGIN, H - 1.25 * cm, "Viktor Sadowski, October 2026")
    c.drawCentredString(W / 2, 1.2 * cm, str(doc.page))
    c.restoreState()


# ── the table from the data ───────────────────────────────────────────────────

MIN_STOCKS = 3       # a sub-industry needs this many stocks today to be in the table


def ends_table(n=8):
    """the n most hurt and the n most helped sub-industries, side by side"""
    o = pd.read_csv(DATA / "order.csv", index_col=0)
    o = o[o.stocks_today >= MIN_STOCKS].sort_values("beta")
    short = {"Communication Services": "Communication", "Consumer Discretionary": "Cons. Discr.",
             "Information Technology": "Technology", "Consumer Staples": "Cons. Staples"}
    line = lambda r: [r.Index, short.get(r.sector, r.sector), f"{r.beta:+.1f} ({r.t:.1f})"]
    rows = [["Most hurt", "Sector", "Beta (t)", "Most helped", "Sector", "Beta (t)"]]
    for a, b in zip(o.head(n).itertuples(), o.tail(n).iloc[::-1].itertuples()):
        rows.append(line(a) + line(b))
    return rows


# ── content ───────────────────────────────────────────────────────────────────

s = []
s += [Spacer(1, 1.2 * cm),
      P(TITLE, title),
      P("S&P 500 sectors and sub-industries against Treasury yields, 2006 to 2026: who is hurt when rates rise, who "
        "is helped, and whether the order holds", subtitle),
      P("Viktor Sadowski · October 2026 · github.com/Viktorsadowski/rate-sensitive-sectors", byline),
      Spacer(1, 0.6 * cm)]

s.append(box([
    P("Summary", ParagraphStyle("abs_h", parent=h2, spaceBefore=0)),
    P("Everyone has a list of rate-sensitive sectors. This project measures one: the weekly return of every S&P 500 "
      "sector and sub-industry from 2006 to 2026 against the change in the 10-year Treasury yield, with the market "
      "held fixed. The number is the return in percent for a 1 point rise in the yield.", abstract),
    P("There are two ends and a wide middle. Utilities (-3.5) and Real Estate (-3.2) are hurt by rising yields, "
      "Energy (+4.2) and Financials (+2.1) are helped, and the seven sectors between them lie from -1.3 to +1.0, "
      "most of them too close to tell apart. Utilities and Real Estate are on the same side in every five-year "
      "window. Financials are helped from the 2010-2014 window on, Energy mostly since 2022, Technology only "
      "before 2017.",
      abstract),
    P("What counts is the yields the market sets. The 2-year yield gives the same order as the 10-year (rank "
      "correlation 0.96), the fed funds target gives none (-0.07). Energy reacts to expected inflation and not to "
      "real yields. Inside the sectors the sub-industries spread as wide as the "
      "sectors do: gold and homebuilders behave like utilities, hotel REITs do not behave like REITs. An order by "
      "sign and size, from most hurt to most helped, holds from one decade to the next (0.78 for sectors). An "
      "order by size alone does not (0.34).", abstract),
]))

# 1
s.append(P("1. The question", h1))
s.append(P("Utilities and REITs are bond proxies, banks like higher rates, growth stocks are long duration. These "
           "are things everyone says and few have measured. There are three questions here. Which sectors are hurt "
           "and which are helped when rates rise, and by how much? Is the order the same in 2010 and in 2025? And "
           "which rate is it: the long yield, the short yield, or the rate the Fed sets? The use is practical. A "
           "model that has the sector of a stock as a number needs an order for the sectors, and how they react to "
           "rates is one candidate for it."))

# 2
s.append(P("2. Data and method", h1))
s.append(P("The stocks are the S&P 500 members from 2006 to September 2026, each for the weeks it was in the index "
           "(point-in-time members from fja05680/sp500). The prices are the ones built for the insider project, daily "
           "closes with dividends from Yahoo Finance, and from Tiingo for companies that are no longer listed: 792 "
           "price series, about 455 stocks in a week. Yields are the daily par yields from treasury.gov, the fed "
           "funds rate is from the New York Fed."))
s.append(P("Today's members have a GICS sector and sub-industry on a public list. The companies that left the index "
           "are on no free list, so the 277 price series of those get a sector from their SIC code at the SEC, through "
           "a hand-made table of code ranges, and 17 of them by hand. On today's members, where both are known, "
           "the table gives the right sector for 89%. Sub-industries are for today's members only."))
s.append(P("The measure is one regression per sector: its weekly return, Wednesday to Wednesday, on the return of "
           "the average S&P 500 stock and on the change in the yield over the same week. The coefficient on the yield "
           "is the rate beta. A beta of -3.5 means that in a week where the 10-year yield rises 0.1 point and the "
           "average stock is flat, the sector loses 0.35%. Every sector is the equal-weighted average of its stocks, "
           "standard errors are Newey-West with four lags, and there are 1,083 weeks."))
s.append(P("The market has to be held fixed because its own reaction to yields changed sign (Figure 1). From 2006 "
           "to 2016 the average stock gained 6.3% for a 1 point rise in the 10-year (t = 4.1): yields mostly rose on "
           "good news about growth. From 2017 to 2026 it is -0.7%, and -4.2% over the last five years, when they "
           "rose on inflation. Without the market in the regression every sector's number would mostly be this."))
s.append(figure(FIG / "yields.png", "Figure 1. The rates used here, and the reason every sector is measured against "
                "the market.", width=TEXT_W))

# 3
s.append(P("3. Results", h1))
s.append(P("3.1 The order of the sectors", h2))
s.append(P("Figure 2 has the eleven sectors from most hurt to most helped. Utilities lose 3.5% for a 1 point rise in "
           "the 10-year at an unchanged market (t = -6.0) and Real Estate 3.2% (t = -5.0). At the other end Energy "
           "gains 4.2% (t = 3.8) and Financials 2.1% (t = 2.4). Between them, Consumer Staples (-1.3) and Health "
           "Care (-1.2) are on the hurt side with t-values of -3.1 and -2.7, and Industrials (+0.6) and Technology "
           "(+1.0) on the helped side with 2.4 and 2.3. Consumer Discretionary, Materials and Communication Services "
           "cannot be told from zero."))
s.append(figure(FIG / "sectors.png", "Figure 2. Black: the whole period with its 95% interval. Blue: the two halves "
                "on their own.", width=TEXT_W * 0.9))
s.append(P("The order does not depend on how it is measured. With SPY as the market, with four-week returns, without "
           "2008, 2009 and 2020, or with today's members only, the sector order has a rank correlation of 0.91 to "
           "0.95 with the one in Figure 2. The companies that left do move the numbers, by up to 1.0: with today's "
           "members only Energy is +3.1 and not +4.2, Financials +1.5 and not +2.1, Materials +0.1 and not +0.6, "
           "and Consumer Discretionary -1.5 and not -1.0."))

s.append(P("3.2 Over time", h2))
s.append(P("The blue points in Figure 2 show that the two halves do not agree everywhere, and Figure 3 shows when "
           "it changed. Utilities and Real Estate are hurt in every five-year window since 2006, most in the years "
           "of low yields before 2020, when a 1 point rise cost them 8 to 10%, and about 4% now. Consumer Staples "
           "are also on the hurt side throughout."))
s.append(figure(FIG / "rolling.png", "Figure 3. Five-year windows, moved one year at a time. The same scale in every "
                "panel.", width=TEXT_W))
s.append(P("The helped end is younger. Financials were on the hurt side in the four windows that end in 2010 to "
           "2013 and have been helped in every window since, +3.4 over 2017 to 2026 (t = 6.6). Energy was between 0 "
           "and +3 in every window that ends in 2021 or before, and has been at +7 to +9 since. Technology went the "
           "other way: +2.2 from 2006 to 2016 (t = 4.0), negative in the windows that end in 2022 to 2025, and "
           "-0.3 over 2017 to 2026 (t = -0.5). Health Care joined the hurt side after 2020. A list of rate-sensitive sectors made in "
           "2013 would have had Technology among the helped and Financials among the hurt."))

s.append(P("3.3 Which rate", h2))
s.append(P("The 2-year yield tells the same story as the 10-year (Figure 4): the two sector orders have a rank "
           "correlation of 0.96. The numbers are a little smaller for Utilities and Real Estate (-3.0 and -2.3) and "
           "larger for Financials, +2.7 with a t-value of 3.9. The fed funds rate tells none. Against the target the "
           "sector order has a rank correlation of -0.07 with the order in Figure 2, and no sector has a t-value "
           "beyond 2. Against the effective rate it is -0.05. The target moved in 46 of the 1,083 weeks, and most moves "
           "were expected before they came, so the stocks had already reacted. Stocks follow the rates the market expects, and the 2-year yield is the free "
           "measure of that."))
s.append(P("Figure 5 splits the 10-year yield in two ways. First into the 2-year yield and the slope of the curve. "
           "Utilities, Real Estate and Energy react to both by about the same amount, so for them the 10-year is the "
           "right rate. Consumer Staples, Health Care, Technology and Financials react to the 2-year and not to the "
           "slope: Financials gain 2.9% for a 1 point rise in the 2-year (t = 3.4) and nothing measurable for a "
           "steeper curve, which is not what the textbook on bank margins says. Industrials, Materials and "
           "Communication Services react to the slope only."))
s.append(figure(FIG / "rates.png", "Figure 4. One regression per rate, each with the market held fixed.",
                width=TEXT_W * 0.9))
s.append(figure(FIG / "parts.png", "Figure 5. Left: the 2-year yield and the slope together. Right: the real yield "
                "and expected inflation together. The breakeven is the 10-year yield minus the 10-year real yield.",
                width=TEXT_W))
s.append(P("The second split is into the real yield and expected inflation. Energy gains 14.8% for a 1 point rise "
           "in expected inflation (t = 5.1) and nothing for a rise in the real yield. What looks like rate sensitivity "
           "in Energy is inflation: the oil price moves both. Consumer Staples are the mirror: -3.2% for inflation "
           "(t = -4.9) and nothing for the real yield. Financials, Industrials and Technology gain on the real "
           "yield and Health Care loses on it. Utilities and Real Estate lose on both."))

s.append(P("3.4 Inside the sectors", h2))
s.append(P("Today's S&P 500 has 126 sub-industries, and 121 have enough weeks for a beta (Figure 6). They spread as "
           "wide inside a sector as the sectors do between them: only 42% of the spread between sub-industries is "
           "between sectors. Gold (-8.8) sits in Materials next to commodity chemicals (+7.3). Homebuilders (-5.8) "
           "sit in Consumer Discretionary and react like a utility. In Real Estate the data center, storage and "
           "tower REITs are at -5 to -8 while hotel REITs are at +0.7. In Financials the banks and investment banks "
           "(+3.0 to +5.4) carry the sector, and insurance brokers, payment companies and exchanges are near zero."))
s.append(figure(FIG / "subs.png", "Figure 6. Sub-industries of today's members. 23 have one stock, 36 have five or "
                "more.", width=TEXT_W * 0.92))
s.append(table(ends_table(), [4.55 * cm, 2.2 * cm, 1.65 * cm, 4.55 * cm, 2.2 * cm, 1.65 * cm],
               "Table 1. The eight most hurt and the eight most helped sub-industries with at least three stocks today. "
               "Return in % for a 1 point rise in the 10-year yield at an unchanged market, t-value in brackets."))
s.append(P("These numbers are noisier than the sector ones. 50 of the 121 have a t-value beyond 2, and a sub-industry "
           "with one or two stocks is close to a single company. For an order, each beta is therefore pulled toward "
           "the beta of its sector, more the less precisely it is measured (data/order.csv has both)."))
s.append(P("Figure 7 asks whether the order holds. Sorted from most hurt to most helped, the sector order of 2006 to "
           "2016 has a rank correlation of 0.78 with the order of 2017 to 2026, and the sub-industry order 0.47. "
           "Sorted by the size of the reaction alone the same numbers are 0.34 and 0.25. The sign carries the "
           "order: banks and utilities both react strongly, in opposite directions, and how strongly changes more "
           "from decade to decade than which way."))
s.append(P("The last test is the practical one. Make an order on 2006 to 2016 and see how well it sorts the "
           "sub-industries by their betas of 2017 to 2026. The sector order alone, every sub-industry at the place "
           "of its sector, gets a rank correlation of 0.44. Sorting the sub-industries inside their sector adds "
           "little, 0.46. One axis for all sub-industries, whatever their sector, gets 0.53. So most of what holds "
           "is in the sector, and the fine order inside a sector is mostly noise from one decade to the next."))
s.append(figure(FIG / "stability.png", "Figure 7. Left: the 105 sub-industries with a beta in both halves. Right: "
                "the same comparison as one number, for sectors and for sub-industries.", width=TEXT_W))

# 4
s.append(P("4. What it says", h1))
s.append(P("Sorted by how they react to rates, the sectors are: Utilities, Real Estate, Consumer Staples, Health "
           "Care, Consumer Discretionary, Industrials, Materials, Communication Services, Technology, Financials, "
           "Energy. Only the ends of that list are measured well. Utilities and Real Estate are hurt in every "
           "period. Financials are helped in every five-year window that ends in 2014 or later, Energy strongly "
           "only since 2022. The seven in the middle should not be read as more than three groups: Staples and "
           "Health Care a little hurt, Industrials a little helped, the rest neutral. Technology is among the "
           "neutral ones, since its sign did not hold."))
s.append(P("For a model the order should be by sign and size, not by size, and it should not be treated as fixed. "
           "The rate to use is the 2-year or the 10-year yield, the fed funds rate gives nothing. And a sector is a rough "
           "bucket for this: gold, homebuilders and hotel REITs are in the wrong one."))
s += bullets(lead="The limits:", items=[
    "The companies that left the index have a sector from their SIC code, and about one in nine of those is wrong. "
    "Companies without a free price history are missing, among them the banks that failed in 2008, so Financials "
    "before 2010 are measured on the ones that lived.",
    "Sub-industries are today's members only, in today's GICS classes, and many are thin. A company that changed "
    "its business is in its present class for all years.",
    "One straight line per bucket: a rise and a fall in yields count the same, and so do a calm week and a week "
    "in a crisis. Equal weights, so a sector is its average company and not its largest ones.",
    "The beta is against the average stock. A sector with a beta of zero still falls when the whole market falls "
    "on a rise in yields, as it has over the last five years.",
])
s.append(P("Next steps: what in a company explains its beta (dividend yield, debt, how far away its profits are), "
           "rises and falls in yields on their own, and the same test on single stocks."))

# where the code and the data are
s.append(Spacer(1, 4))
s.append(P("Code: github.com/Viktorsadowski/rate-sensitive-sectors, three scripts on top of the price data of "
           "insider-trade-worth (run order in the README). Data: S&P 500 history from github.com/fja05680/sp500, "
           "today's members from github.com/datasets/s-and-p-500-companies, prices from Yahoo Finance and Tiingo, "
           "yields from treasury.gov, the fed funds rate from the New York Fed, SIC codes from the SEC. Nothing "
           "licensed. data/results.csv has every beta, also the ones not shown here.", caption))


# headings stick to whatever comes right after them
def flat(*fs):
    out = []
    for f in fs:
        out += f._content if isinstance(f, KeepTogether) else [f]
    return out


story, i = [], 0
while i < len(s):
    f = s[i]
    if isinstance(f, Paragraph) and f.style.name in ("h1", "h2") and i + 1 < len(s):
        nxt = s[i + 1]
        if isinstance(nxt, Paragraph) and nxt.style.name == "h2" and i + 2 < len(s):
            story.append(KeepTogether(flat(f, nxt, s[i + 2])))
            i += 3
            continue
        story.append(KeepTogether(flat(f, nxt)))
        i += 2
        continue
    story.append(f)
    i += 1

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.0 * cm,
                        bottomMargin=2.0 * cm, title=TITLE, author="Viktor Sadowski")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"report -> {OUT}")
