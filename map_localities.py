# Map of the 2026 Knesset election by locality: first-place list and bloc majority.
# Data: dados/expc.csv (results by locality, CEC), dados/bycode2021.xlsx (CBS locality coordinates, Israel TM grid),
# dados/levant_outlines.json (Natural Earth 10m country outlines, lon/lat) projected to ITM with a Transverse Mercator.
# usage: python3 map_localities.py            (writes map_localities_2026.png)
#        BARE=1 python3 map_localities.py     (no header or footer text, the version embedded in the article)
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

BG, PANEL, INK, MUTED, LAND, NEIGH = "#0d1117", "#141b24", "#e8eaed", "#9aa4b2", "#232c38", "#1a212b"
PINK = "#e8478f"

# ---- Israel TM projection (GRS80) ----------------------------------------------------------------
a, f = 6378137.0, 1 / 298.257222101
e2 = 2 * f - f * f; ep2 = e2 / (1 - e2)
lat0, lon0 = np.radians(31 + 44 / 60 + 3.817 / 3600), np.radians(35 + 12 / 60 + 16.261 / 3600)
k0, FE, FN = 1.0000067, 219529.584, 626907.390


def meridian(phi):
    return a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
                - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
                + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi)
                - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))


def to_itm(lon, lat):
    phi, lam = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2); T = np.tan(phi) ** 2; C = ep2 * np.cos(phi) ** 2
    A = (lam - lon0) * np.cos(phi)
    x = FE + k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120)
    y = FN + k0 * (meridian(phi) - meridian(lat0) + N * np.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                                                        + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


def rings(geom):
    if geom["type"] == "Polygon":
        return [np.array(geom["coordinates"][0])]
    return [np.array(poly[0]) for poly in geom["coordinates"]]


# ---- data ------------------------------------------------------------------------------------------
D = "data" if os.path.isdir("data") else "dados"
c = pd.read_csv(f"{D}/expc.csv", encoding="utf-8-sig"); c = c.loc[:, ~c.columns.str.startswith("Unnamed")]
META = ['סמל ועדה', 'שם ישוב', 'סמל ישוב', 'בזב', 'מצביעים', 'פסולים', 'כשרים']
P = [p for p in c.columns if p not in META]
c = c[c['סמל ישוב'] != 9999].copy()
c["winner"] = c[P].idxmax(axis=1)
x = pd.read_excel(f"{D}/bycode2021.xlsx")[['סמל יישוב', 'קואורדינטות', 'שם יישוב באנגלית']].rename(columns={'סמל יישוב': 'סמל ישוב'})
m = c.merge(x, on='סמל ישוב', how="inner")
co = m['קואורדינטות'].astype(int).astype(str).str.zfill(10)
m["X"] = co.str[:5].astype(int) * 10.0; m["Y"] = co.str[5:].astype(int) * 10.0
BLOC = ["מחל", "ב", "ט", "שס", "ג"]; ARAB = ["ודם", "עם"]
m["bloc"] = m[BLOC].sum(axis=1) / m['כשרים'].clip(lower=1)
m["arab"] = m[ARAB].sum(axis=1) / m['כשרים'].clip(lower=1)
m["cat"] = np.where(m.bloc >= .5, "coal", np.where(m.arab >= .5, "arab", "opp"))

NAMES = {"מחל": "Likud", "דרך": "Yashar!", "ג": "UTJ", "ודם": "Joint List", "עם": "Ra'am", "ט": "Religious Zionism",
         "שס": "Shas", "אמת": "Democrats", "ל": "Yisrael Beiteinu", "ב": "Otzma Yehudit", "רק": "Beyachad", "ך": "Amcha Yisrael"}
COL = {"מחל": "#4a9ae8", "דרך": "#e8478f", "ג": "#a78bfa", "ודם": "#34d399", "עם": "#14b8a6", "ט": "#f59e0b",
       "שס": "#facc15", "אמת": "#f87171", "ל": "#93c5fd", "ב": "#c2410c", "רק": "#fb923c", "ך": "#9ca3af"}
BCOL = {"coal": "#4a9ae8", "opp": "#e8478f", "arab": "#34d399"}
BNAME = {"coal": "Netanyahu bloc 50% or more", "opp": "Bloc under 50%, opposition ahead", "arab": "Arab parties 50% or more"}

geo = json.load(open(f"{D}/levant_outlines.json"))
BARE = os.environ.get("BARE") == "1"   # no header/footer text, for embedding in the article
fig = plt.figure(figsize=(14, 13.4 if BARE else 15), facecolor=BG)
if not BARE: fig.text(0.04, 0.975, "WHO CAME FIRST IN EACH LOCALITY · 26TH KNESSET, 27 OCTOBER 2026", color=PINK, fontsize=13,
         family="DejaVu Sans Mono", weight="bold", va="top")
if not BARE: fig.text(0.04, 0.957, "Each circle is a locality; its area grows with the number of valid votes. Results by locality (CEC file of 2 November), "
         "coordinates from the CBS locality list.", color=INK, fontsize=11.5, va="top")
axes = ([fig.add_axes([0.02, 0.01, 0.47, 0.955]), fig.add_axes([0.51, 0.01, 0.47, 0.955])] if BARE else
        [fig.add_axes([0.04, 0.075, 0.45, 0.845]), fig.add_axes([0.52, 0.075, 0.45, 0.845])])
XL, YL = (120000, 300000), (370000, 800000)
for ax in axes:
    ax.set_facecolor(PANEL)
    for iso, col in [("EGY", NEIGH), ("JOR", NEIGH), ("SYR", NEIGH), ("LBN", NEIGH), ("ISR", LAND), ("PSE", LAND)]:
        for ring in rings(geo[iso]):
            px, py = to_itm(ring[:, 0], ring[:, 1])
            ax.add_patch(MplPolygon(np.c_[px, py], closed=True, facecolor=col, edgecolor="#3a4654", linewidth=0.6))
    for ring in rings(geo["PSE"]):
        px, py = to_itm(ring[:, 0], ring[:, 1])
        ax.plot(px, py, color="#6b7685", linewidth=0.7, linestyle=(0, (3, 3)))
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#2b3440")

size = lambda v: 6 + 0.055 * np.sqrt(v)   # marker area in points^2, sqrt scale so area ~ votes
order = m.sort_values('כשרים', ascending=False)
axes[0].scatter(order.X, order.Y, s=size(order['כשרים']) ** 2 / 4, c=[COL.get(w, "#9ca3af") for w in order.winner], alpha=.85,
                edgecolors=BG, linewidths=.4)
axes[1].scatter(order.X, order.Y, s=size(order['כשרים']) ** 2 / 4, c=[BCOL[k] for k in order.cat], alpha=.85, edgecolors=BG, linewidths=.4)
axes[0].set_title("First-place list", color=INK, fontsize=12.5, loc="left", pad=8)
axes[1].set_title("Bloc majority", color=INK, fontsize=12.5, loc="left", pad=8)

# city labels (English names from the CBS list)
LABELS = {"Jerusalem": (12000, -3000), "Tel Aviv - Yafo": (-9000, 3000), "Haifa": (-8000, 4000), "Be'er Sheva": (9000, -2000),
          "Nazareth": (9000, 2000), "Eilat": (9000, 0), "Umm al-Fahm": (9000, 1500), "Rahat": (9000, -3000), "Bene Beraq": (10000, -5000),
          "Ashdod": (-9000, -4000), "Netanya": (-9000, 1000), "Modi'in Illit": (9000, 2500), "Bet Shemesh": (-6000, -9000), "Ari'el": (9000, 2000)}
for i, r in m.iterrows():
    n = r['שם יישוב באנגלית']
    if n in LABELS:
        dx, dy = LABELS[n]
        for ax in axes:
            ax.annotate(n.replace(" - Yafo", "").replace("Bene Beraq", "Bnei Brak").replace("Be'er Sheva", "Beersheba"), (r.X, r.Y),
                        xytext=(r.X + dx, r.Y + dy), color=INK, fontsize=8.5, ha="left" if dx > 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color="#6b7685", lw=.5))

# inset: the centre of the country, where the circles overlap
IX, IY = (150000, 240000), (612000, 706000)
IL = {"Jerusalem": (7000, 0), "Tel Aviv - Yafo": (-6000, 0), "Bene Beraq": (6500, -2500), "Netanya": (-5500, 0), "Ashdod": (-5500, 0),
      "Modi'in Illit": (7000, 0), "Bet Shemesh": (0, -4000), "Rishon LeZiyyon": (-6500, -800), "Petah Tiqwa": (6500, 1800), "Holon": (-5000, -1500),
      "Rehovot": (-5500, -1500)}
for k, (ax, colors) in enumerate([(axes[0], [COL.get(w, "#9ca3af") for w in order.winner]), (axes[1], [BCOL[c_] for c_ in order.cat])]):
    ax.add_patch(MplPolygon([[IX[0], IY[0]], [IX[1], IY[0]], [IX[1], IY[1]], [IX[0], IY[1]]], closed=True, fill=False, edgecolor="#9aa4b2", linewidth=.7))
    ins = fig.add_axes([0.035 + 0.49 * k, 0.02, 0.2, 0.22] if BARE else [0.055 + 0.48 * k, 0.085, 0.2, 0.22])
    ins.set_facecolor(PANEL)
    for iso, col in [("ISR", LAND), ("PSE", LAND)]:
        for ring in rings(geo[iso]):
            px, py = to_itm(ring[:, 0], ring[:, 1])
            ins.add_patch(MplPolygon(np.c_[px, py], closed=True, facecolor=col, edgecolor="#3a4654", linewidth=0.6))
    for ring in rings(geo["PSE"]):
        px, py = to_itm(ring[:, 0], ring[:, 1])
        ins.plot(px, py, color="#6b7685", linewidth=0.7, linestyle=(0, (3, 3)))
    ins.scatter(order.X, order.Y, s=size(order['כשרים']) ** 2 / 2.2, c=colors, alpha=.85, edgecolors=BG, linewidths=.4)
    for i, r in m.iterrows():
        n = r['שם יישוב באנגלית']
        if n in IL:
            dx, dy = IL[n]
            ins.text(r.X + dx, r.Y + dy, n.replace(" - Yafo", "").replace("Bene Beraq", "Bnei Brak").replace("Rishon LeZiyyon", "Rishon LeZion").replace("Petah Tiqwa", "Petah Tikva"),
                     color=INK, fontsize=7, ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center")
    ins.set_xlim(*IX); ins.set_ylim(*IY); ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor("#9aa4b2")
    ins.set_title("centre of the country", color=MUTED, fontsize=8, loc="left", pad=3)

# legends with counts
cnt = m.groupby("winner").agg(n=("winner", "size"), v=('כשרים', "sum")).sort_values("v", ascending=False)
handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=COL.get(k, "#9ca3af"), markeredgecolor=BG,
                  label=f"{NAMES.get(k, k)}: {int(r.n)} loc., {r.v / 1e6:.2f} M votes" if r.v >= 1e6 else f"{NAMES.get(k, k)}: {int(r.n)} loc., {r.v / 1e3:.0f} k votes")
           for k, r in cnt.iterrows() if r.n >= 2]
leg = axes[0].legend(handles=handles, loc="lower right", fontsize=7.6, frameon=True, facecolor=BG, edgecolor="#2b3440", labelcolor=INK, borderpad=.8)
cb = m.groupby("cat").agg(n=("cat", "size"), v=('כשרים', "sum"))
handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=BCOL[k], markeredgecolor=BG,
                  label=f"{BNAME[k]}: {int(cb.loc[k].n)} loc., {cb.loc[k].v / 1e6:.2f} M votes") for k in ["coal", "opp", "arab"]]
axes[1].legend(handles=handles, loc="upper left", fontsize=7.6, frameon=True, facecolor=BG, edgecolor="#2b3440", labelcolor=INK, borderpad=.8)

if not BARE: fig.text(0.04, 0.045, "Bloc: Likud, Otzma Yehudit, Religious Zionism, Shas and UTJ. Arab parties: Joint List and Ra'am. 'Jewish opposition parties ahead' means "
         "the bloc is under 50% and the Arab parties under 50%. Dashed line: 1949 armistice line. Seven localities without CBS coordinates (8,560 votes) are not shown. "
         "Double-envelope votes (soldiers, diplomats, hospitals) have no locality.", color=MUTED, fontsize=9, va="top", wrap=True)
if not BARE: fig.text(0.04, 0.012, "Source: Central Elections Committee, results by locality (expc.csv, 2 November 2026); CBS localities file 2021; Natural Earth. "
         "Analysis: lilianezukerman.com", color=MUTED, fontsize=9, va="top")
fig.savefig(os.environ.get("OUT", "map_localities_2026.png"), dpi=150, facecolor=BG)
print("saved; localities drawn:", len(m))
print(cnt.to_string()); print(cb.to_string())
