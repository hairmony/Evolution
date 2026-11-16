"""
Evolution app for Big Data

    conda activate bigdata
    pip install streamlit pandas pyarrow
    streamlit run app.py
"""

import glob
import os
import socket
import sys
import time
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


def import_pyspark():
    try:
        import pyspark
    except ImportError:
        home = os.environ["SPARK_HOME"]
        sys.path.insert(0, os.path.join(home, "python"))
        sys.path.insert(0, glob.glob(os.path.join(home, "python", "lib", "py4j-*.zip"))[0])
    from pyspark.sql import SparkSession, functions as F
    return SparkSession, F


def clean_master_url(url):
    url = url.strip().strip('"').rstrip("/")
    if not url.startswith("spark://"):
        return url
    hostport = url[len("spark://"):]
    if ":" in hostport:
        host, port = hostport.split(":", 1)
    else:
        parts = hostport.split(".")
        if len(parts) == 5:
            host, port = ".".join(parts[:4]), parts[4]
        else:
            host, port = hostport, "7077"
    return f"spark://{host.strip()}:{port.strip()}"


SPARK_MASTER = clean_master_url(os.environ.get("SPARK_MASTER", "spark://172.20.10.3:7077"))


def own_ip_towards(master_url):
    host = master_url.split("://", 1)[-1].split(":")[0]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 7077))
        return s.getsockname()[0]
    finally:
        s.close()


@st.cache_resource
def get_spark():
    SparkSession, _ = import_pyspark()
    builder = (SparkSession.builder.appName("AnimalLineageApp")
               .master(SPARK_MASTER)
               .config("spark.sql.shuffle.partitions", "8"))
    if SPARK_MASTER.startswith("spark://"):
        driver_host = os.environ.get("DRIVER_HOST") or own_ip_towards(SPARK_MASTER)
        builder = (builder.config("spark.driver.host", driver_host)
                   .config("spark.driver.bindAddress", "0.0.0.0"))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def rank_names():
    names = dict(RANK_NAMES)
    if RANKS_CSV.exists():
        r = pd.read_csv(RANKS_CSV, dtype=str)
        names.update(dict(zip(r["rank"].str.replace(ENTITY, "", regex=False), r["rankLabel"])))
    return names


@st.cache_data(show_spinner=False)
def compute_with_spark():
    t0 = time.time()
    spark = get_spark()
    _, F = import_pyspark()

    raw = (spark.read.csv(DATA_DIR.as_uri() + "/*.csv", header=True)
           .withColumn("dataset", F.regexp_extract(F.input_file_name(), r"([^/\\]+)\.csv$", 1)))
    for c in ["taxon", "parent", "rank"]:
        raw = raw.withColumn(c, F.regexp_replace(c, ENTITY, ""))
    raw = raw.repartition(8).cache()
    hosts = raw.rdd.mapPartitions(lambda _: [socket.gethostname()]).distinct().collect()

    n_raw = raw.count()
    per = raw.groupBy("taxon").agg(
        F.countDistinct("parent").alias("parents"),
        F.countDistinct("commonName").alias("names"),
        F.count("rank").alias("rank"),
        F.count("parent").alias("parent_any"),
        F.count("commonName").alias("common"),
        F.max((F.col("taxon") == F.col("parent")).cast("int")).alias("self_parent"),
    ).cache()
    n_distinct = per.count()
    report = pd.DataFrame([
        ("Raw rows", n_raw),
        ("Distinct taxa", n_distinct),
        ("Duplicate rows removed", n_raw - n_distinct),
        ("Taxa with 2+ parents (conflicting classification)", per.filter("parents > 1").count()),
        ("Taxa with 2+ English common names", per.filter("names > 1").count()),
        ("Taxa missing rank", per.filter("rank = 0").count()),
        ("Taxa missing parent", per.filter("parent_any = 0").count()),
        ("Taxa missing English common name", per.filter("common = 0").count()),
        ("Taxa listed as their own parent", per.filter("self_parent = 1").count()),
    ], columns=["check", "count"])

    rmap = F.create_map([F.lit(x) for pair in rank_names().items() for x in pair])
    clean = (raw.filter(F.col("parent").isNull() | (F.col("taxon") != F.col("parent")))
             .groupBy("taxon").agg(
                 F.first("name", ignorenulls=True).alias("name"),
                 F.first("rank", ignorenulls=True).alias("rank_id"),
                 F.min("parent").alias("parent"),
                 F.min("commonName").alias("commonName"),
                 F.first("dataset").alias("dataset"))
             .withColumn("rank", F.coalesce(rmap[F.col("rank_id")], F.col("rank_id"), F.lit("unranked")))
             .drop("rank_id").cache())

    nodes = clean.select(F.col("taxon").alias("n_taxon"), F.col("parent").alias("n_parent"))
    lin = clean.select("taxon", F.array("taxon").alias("path"), F.col("parent").alias("cur"))
    rounds = 0
    for _ in range(60):
        rounds += 1
        j = lin.join(nodes, lin.cur == nodes.n_taxon, "left")
        ok = F.col("n_taxon").isNotNull() & ~F.array_contains("path", F.col("cur"))
        lin = j.select(
            "taxon",
            F.when(ok, F.concat("path", F.array("cur"))).otherwise(F.col("path")).alias("path"),
            F.when(ok, F.col("n_parent")).alias("cur"),
        ).localCheckpoint()
        if lin.filter(F.col("cur").isNotNull()).count() == 0:
            break

    lineage = lin.select("taxon", "path", (F.size("path") - 1).alias("depth")).join(clean, "taxon")
    ranked = (lineage.select("taxon", F.explode("path").alias("ancestor"))
              .join(clean.select(F.col("taxon").alias("ancestor"), F.col("rank").alias("a_rank"),
                                 F.col("name").alias("a_name")), "ancestor"))
    groups = ranked.groupBy("taxon").agg(
        F.first(F.when(F.col("a_rank") == "order", F.col("a_name")), ignorenulls=True).alias("order"),
        F.first(F.when(F.col("a_rank") == "family", F.col("a_name")), ignorenulls=True).alias("family"),
    )
    full = lineage.join(groups, "taxon", "left").cache()
    species = full.filter(F.col("rank") == "species")
    analysis = {
        "per_order": species.groupBy("dataset", "order").count().toPandas(),
        "per_family": species.groupBy("family").count().toPandas(),
        "by_rank": full.groupBy("rank").count().toPandas(),
        "coverage": species.groupBy("dataset").agg(
            F.count("*").alias("species"),
            F.round(100 * F.avg((F.col("commonName").isNotNull()).cast("int")), 1).alias("pct_with_common_name"),
            F.round(F.avg("depth"), 1).alias("avg_lineage_depth"),
            F.max("depth").alias("max_lineage_depth"),
        ).toPandas(),
    }
    taxa = full.toPandas()

    taxa["path"] = taxa["path"].map(list)
    taxa["commonName"] = taxa["commonName"].fillna("")
    taxa["_name_lc"] = taxa["name"].str.lower()
    taxa["_common_lc"] = taxa["commonName"].str.lower()
    stats = {"hosts": sorted(hosts), "seconds": round(time.time() - t0, 1), "rounds": rounds,
             "master": SPARK_MASTER, "driver": spark.conf.get("spark.driver.host")}
    return taxa, report, analysis, stats


@st.cache_data
def build_index(_df):
    info = _df.set_index("taxon")[["name", "commonName", "rank", "parent", "depth"]].to_dict("index")
    children = _df.dropna(subset=["parent"]).groupby("parent")["taxon"].apply(list).to_dict()
    return info, children


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

with st.spinner(f"Spark is processing the data on {SPARK_MASTER} ..."):
    taxa, report, analysis, stats = compute_with_spark()
info, children = build_index(taxa)

with st.sidebar:
    st.subheader("Spark cluster")
    st.markdown(f"Master: `{stats['master']}`  \nApp (driver) IP: `{stats['driver']}`")
    st.markdown("Machines that processed data:")
    for h in stats["hosts"]:
        st.markdown(f"- `{h}`")
    st.markdown(f"Lineage rounds: `{stats['rounds']}`  \nProcessing time: `{stats['seconds']} s`")
    if st.button("Recompute on cluster"):
        compute_with_spark.clear()
        st.rerun()

st.title("🌳 Animal Lineage Explorer")
st.caption(f"{len(taxa):,} taxa from Wikidata, processed on a PySpark cluster")

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
    per_order = analysis["per_order"]
    per_family = analysis["per_family"]
    by_rank = analysis["by_rank"]
    coverage = analysis["coverage"]

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