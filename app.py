"""
Evolution app for Big Data

    conda activate bigdata
    pip install streamlit pandas pyarrow
    streamlit run app.py
"""

import os
from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(os.environ.get("BIGDATA_DIR", r"C:\bigdata")) / "data"
RANKS_CSV = DATA_DIR.parent / "lookup" / "ranks.csv"
MAX_CHILDREN_SHOWN = 8
ENTITY = "http://www.wikidata.org/entity/"
RANK_NAMES = {"Q7432": "species", "Q68947": "subspecies", "Q34740": "genus",
              "Q35409": "family", "Q36602": "order", "Q37517": "class"}

st.set_page_config(page_title="Animal Lineage Explorer", page_icon="🌳", layout="wide")


@st.cache_data
def load_raw():
    frames = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        d = pd.read_csv(f, dtype=str)
        d["dataset"] = f.stem
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    for c in ["taxon", "parent", "rank"]:
        if c in raw:
            raw[c] = raw[c].str.replace(ENTITY, "", regex=False)
    return raw


@st.cache_data
def load_taxa(raw):
    names = dict(RANK_NAMES)
    if RANKS_CSV.exists():
        r = pd.read_csv(RANKS_CSV, dtype=str)
        names.update(dict(zip(r["rank"].str.replace(ENTITY, "", regex=False), r["rankLabel"])))

    df = (raw[raw["parent"].isna() | (raw["taxon"] != raw["parent"])]
          .sort_values(["taxon", "parent", "commonName"], na_position="last")
          .groupby("taxon", as_index=False).first())
    df["rank"] = df["rank"].map(lambda x: names.get(x, x) if isinstance(x, str) else "unranked")
    df["commonName"] = df["commonName"].fillna("")

    parent = dict(zip(df["taxon"], df["parent"]))
    known = set(parent)
    paths = {}

    def lineage(t):
        if t in paths:
            return paths[t]
        chain, seen = [], set()
        cur = t
        while cur in known and cur not in seen:
            if cur in paths:
                chain.extend(paths[cur])
                break
            seen.add(cur)
            chain.append(cur)
            cur = parent.get(cur)
        paths[t] = chain
        return chain

    df["path"] = df["taxon"].map(lineage)
    df["depth"] = df["path"].str.len() - 1

    rank_of = dict(zip(df["taxon"], df["rank"]))
    name_of = dict(zip(df["taxon"], df["name"]))

    def first_with_rank(path, rank):
        return next((name_of[t] for t in path if rank_of.get(t) == rank), None)

    df["order"] = df["path"].map(lambda p: first_with_rank(p, "order"))
    df["family"] = df["path"].map(lambda p: first_with_rank(p, "family"))
    df["_name_lc"] = df["name"].str.lower()
    df["_common_lc"] = df["commonName"].str.lower()
    return df


@st.cache_data
def build_index(_df):
    info = _df.set_index("taxon")[["name", "commonName", "rank", "parent", "depth"]].to_dict("index")
    children = _df.dropna(subset=["parent"]).groupby("parent")["taxon"].apply(list).to_dict()
    return info, children


def quality_report(raw):
    per = raw.groupby("taxon").agg(rows=("taxon", "size"), parents=("parent", "nunique"),
                                   names=("commonName", "nunique"), rank=("rank", "count"),
                                   parent_any=("parent", "count"), common=("commonName", "count"))
    return pd.DataFrame([
        ("Raw rows", len(raw)),
        ("Distinct taxa", len(per)),
        ("Duplicate rows removed", len(raw) - len(per)),
        ("Taxa with 2+ parents (conflicting classification)", int((per["parents"] > 1).sum())),
        ("Taxa with 2+ English common names", int((per["names"] > 1).sum())),
        ("Taxa missing rank", int((per["rank"] == 0).sum())),
        ("Taxa missing parent", int((per["parent_any"] == 0).sum())),
        ("Taxa missing English common name", int((per["common"] == 0).sum())),
        ("Taxa listed as their own parent", int((raw["taxon"] == raw["parent"]).groupby(raw["taxon"]).any().sum())),
    ], columns=["check", "count"])


def search(df, text, limit=50):
    t = text.strip().lower()
    if not t:
        return df.iloc[0:0]
    hit = df[df["_common_lc"].str.contains(t, regex=False) | df["_name_lc"].str.contains(t, regex=False)].copy()
    hit["_score"] = 3
    hit.loc[hit["_common_lc"].str.startswith(t) | hit["_name_lc"].str.startswith(t), "_score"] = 1
    hit.loc[(hit["_common_lc"] == t) | (hit["_name_lc"] == t), "_score"] = 0
    hit["_has_common"] = hit["commonName"] == ""
    return hit.sort_values(["_score", "_has_common", "depth", "name"]).head(limit)


def esc(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def node_label(info, tid):
    r = info.get(tid)
    if r is None:
        return tid
    lines = [r["name"]]
    if r["commonName"]:
        lines.append(f"({r['commonName']})")
    lines.append(r["rank"])
    return "\\n".join(esc(x) for x in lines)


def lineage_graph(selected, path, levels_up, info, children):
    chain = set(path)
    anc_idx = min(levels_up, len(path) - 1)
    branch_root = path[anc_idx]

    nodes, edges, more = [], [], []
    seen = set()


    q = deque([(branch_root, 0)])
    while q:
        tid, d = q.popleft()
        if tid in seen:
            continue
        seen.add(tid)
        nodes.append(tid)
        if d >= anc_idx:
            continue
        kids = children.get(tid, [])
        kids = sorted(kids, key=lambda k: (k not in chain,
                                           info.get(k, {}).get("commonName", "") == "",
                                           info.get(k, {}).get("name", "")))
        for k in kids[:MAX_CHILDREN_SHOWN]:
            edges.append((tid, k))
            q.append((k, d + 1))
        if len(kids) > MAX_CHILDREN_SHOWN:
            more.append((tid, len(kids) - MAX_CHILDREN_SHOWN))


    for i in range(anc_idx, len(path) - 1):
        child, parent = path[i], path[i + 1]
        nodes.append(parent)
        edges.append((parent, child))

    lines = [
        "digraph G {",
        '  rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,'
        ' color="#9aa5b1", fillcolor="#f5f7fa", fontcolor="#1f2933"];',
        '  edge [color="#9aa5b1", arrowhead=none];',
    ]
    for tid in dict.fromkeys(nodes):
        style = ""
        if tid == selected:
            style = ', fillcolor="#2f855a", fontcolor="white", color="#22543d", penwidth=2'
        elif tid in chain:
            style = ', fillcolor="#e6f4ea", color="#2f855a", penwidth=1.5'
        lines.append(f'  "{tid}" [label="{node_label(info, tid)}"{style}];')
    for a, b in dict.fromkeys(edges):
        bold = ', color="#2f855a", penwidth=2' if a in chain and b in chain else ""
        lines.append(f'  "{a}" -> "{b}" [{bold.lstrip(", ")}];')
    for tid, n in more:
        mid = f"more_{tid}"
        lines.append(f'  "{mid}" [label="+{n} more", shape=plaintext, style="", fontcolor="#7b8794"];')
        lines.append(f'  "{tid}" -> "{mid}" [style=dashed];')
    lines.append("}")
    return "\n".join(lines), len(dict.fromkeys(nodes))


if not list(DATA_DIR.glob("*.csv")):
    st.error(f"No CSV files in {DATA_DIR}. Download them from query.wikidata.org first.")
    st.stop()

raw = load_raw()
taxa = load_taxa(raw)
info, children = build_index(taxa)

st.title("🌳 Animal Lineage Explorer")
st.caption(f"{len(taxa):,} taxa from Wikidata, read straight from the downloaded CSVs")

tab_tree, tab_value, tab_quality = st.tabs(["Lineage explorer", "Analysis", "Data quality"])


with tab_tree:
    left, right = st.columns([1, 3], gap="large")

    with left:
        query = st.text_input("Search an animal", value="lion", placeholder="lion, Panthera leo, python…")
        matches = search(taxa, query)
        if matches.empty:
            st.info("No match. Try the Latin name, or a different spelling.")
            st.stop()

        def describe(tid):
            r = info[tid]
            return f"{r['commonName'] or r['name']}  ·  {r['name']} ({r['rank']})" if r["commonName"] \
                else f"{r['name']} ({r['rank']})"

        selected = st.selectbox(f"{len(matches)} match(es)", matches["taxon"].tolist(), format_func=describe)
        row = taxa.loc[taxa["taxon"] == selected].iloc[0]
        path = row["path"]

        levels = st.slider("How far out to show relatives", 1, min(4, max(1, len(path) - 1)), 2,
                           help="1 = siblings (same parent), 2 = cousins (same grandparent), and so on")
        show_all = st.toggle("Show the whole lineage up to the top", value=False)

        st.markdown(f"**{row['name']}**" + (f" — {row['commonName']}" if row["commonName"] else ""))
        st.markdown(f"Rank: `{row['rank']}`  \nOrder: `{row.get('order') or '–'}`  \n"
                    f"Family: `{row.get('family') or '–'}`  \nLineage depth: `{row['depth']}` steps")
        st.markdown(f"[Open on Wikidata ↗](https://www.wikidata.org/wiki/{selected})")

    with right:
        shown_path = path if show_all else path[: max(levels + 6, 1)]
        dot, n_nodes = lineage_graph(selected, shown_path, levels, info, children)
        st.graphviz_chart(dot, width="stretch")
        if not show_all and len(path) > len(shown_path):
            st.caption(f"Showing the nearest {len(shown_path) - 1} of {len(path) - 1} ancestors. "
                       "Turn on 'Show the whole lineage' to see all of them.")

        with st.expander("Lineage as a list"):
            st.dataframe(
                pd.DataFrame([{"step": i, "rank": info[t]["rank"], "name": info[t]["name"],
                               "common name": info[t]["commonName"], "wikidata": t}
                              for i, t in enumerate(path) if t in info]),
                hide_index=True, width="stretch",
            )


with tab_value:
    species = taxa[taxa["rank"] == "species"]
    per_order = species.groupby(["dataset", "order"]).size().reset_index(name="count")
    per_family = species.groupby("family").size().reset_index(name="count")
    by_rank = taxa.groupby("rank").size().reset_index(name="count")
    coverage = species.groupby("dataset").agg(
        species=("taxon", "size"),
        pct_with_common_name=("commonName", lambda s: round(100 * (s != "").mean(), 1)),
        avg_lineage_depth=("depth", lambda s: round(s.mean(), 1)),
        max_lineage_depth=("depth", "max"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        if len(per_order):
            st.subheader("Species per order (top 15)")
            top = per_order.dropna(subset=["order"]).nlargest(15, "count")
            st.bar_chart(top, x="order", y="count", color="dataset", horizontal=True)
    with c2:
        if len(per_family):
            st.subheader("Biggest families (top 15)")
            st.bar_chart(per_family.dropna(subset=["family"]).nlargest(15, "count"),
                         x="family", y="count", horizontal=True)

    c3, c4 = st.columns(2)
    with c3:
        if len(by_rank):
            st.subheader("Taxa by rank")
            st.bar_chart(by_rank.dropna(subset=["rank"]).nlargest(12, "count"), x="rank", y="count")
    with c4:
        if len(coverage):
            st.subheader("Per dataset")
            st.dataframe(coverage.rename(columns={
                "pct_with_common_name": "% with English name",
                "avg_lineage_depth": "avg lineage depth",
                "max_lineage_depth": "max lineage depth"}), hide_index=True, width="stretch")


with tab_quality:
    report = quality_report(raw)
    st.subheader("Problems found in the raw data (Veracity)")
    n_raw = int(report.loc[report["check"] == "Raw rows", "count"].iloc[0])
    n_distinct = int(report.loc[report["check"] == "Distinct taxa", "count"].iloc[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw rows downloaded", f"{n_raw:,}")
    m2.metric("Distinct taxa", f"{n_distinct:,}")
    m3.metric("Duplicate rows", f"{n_raw - n_distinct:,}",
              f"{100 * (n_raw - n_distinct) / n_raw:.1f}% of raw", delta_color="off")
    issues = report[~report["check"].isin(["Raw rows", "Distinct taxa", "Duplicate rows removed"])]
    st.bar_chart(issues, x="check", y="count", horizontal=True)
    st.dataframe(report, hide_index=True, width="stretch")