"""
Evolution app for Big Data

    conda activate bigdata
    pip install streamlit pandas pyarrow
    streamlit run app.py
"""

import findspark
findspark.init()

import altair as alt
import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MY_IP = "10.110.15.223"
MASTER = "local[2]" # local[2] 
FOLDER = "C:/bigdata"

STATUS_NAMES = {
    "Q211005": "Least Concern",
    "Q719675": "Near Threatened",
    "Q278113": "Vulnerable",
    "Q96377276": "Endangered",
    "Q219127": "Critically Endangered",
    "Q239509": "Extinct in the Wild",
    "Q237350": "Extinct",
    "Q3245245": "Data Deficient",
}
ACTIVITY_NAMES = {
    "Q101029366": "nocturnal",
    "Q4284186": "diurnal",
    "Q3255068": "crepuscular",
    "Q5052479": "cathemeral",
}
RANK_NAMES = {
    "Q7432": "species",
    "Q68947": "subspecies",
    "Q34740": "genus",
    "Q35409": "family",
    "Q36602": "order",
    "Q37517": "class",
    "Q164280": "subfamily",
    "Q2136103": "superfamily",
    "Q5867959": "suborder",
    "Q2889003": "infraorder",
    "Q227936": "tribe",
    "Q713623": "clade",
    "Q1306176": "hybrid",
    "Q3238261": "subgenus",
    "Q5867051": "subclass",
    "Q5868144": "superorder",
    "Q2007442": "infraclass",
    "Q6054237": "magnorder",
}


@st.cache_resource
def get_spark():
    builder = SparkSession.builder.appName("AnimalLineage").master(MASTER).config("spark.sql.shuffle.partitions", "8")
    if MASTER.startswith("spark://"):
        builder = builder.config("spark.driver.host", MY_IP).config("spark.driver.bindAddress", "0.0.0.0")
    return builder.getOrCreate()


def short_id(column):
    return F.regexp_replace(column, "http://www.wikidata.org/(entity|prop/direct)/", "")


def make_map(names):
    pairs = []
    for code, name in names.items():
        pairs += [F.lit(code), F.lit(name)]
    return F.create_map(pairs)


@st.cache_data
def load_data():
    spark = get_spark()

    animals = spark.read.csv(f"file:///{FOLDER}/data/*.csv", header=True)
    animals = (animals.withColumn("taxon", short_id("taxon"))
                      .withColumn("parent", short_id("parent"))
                      .withColumn("rank", short_id("rank")))

    features = spark.read.csv(f"file:///{FOLDER}/features/*.csv", header=True)
    features = (features.withColumn("taxon", short_id("taxon"))
                        .withColumn("prop", short_id("prop"))
                        .withColumn("value", short_id("value")))

    computers = animals.select(F.expr("reflect('java.net.InetAddress', 'getLocalHost')")).distinct()
    computers = [row[0] for row in computers.collect()]

    clean = animals.groupBy("taxon").agg(
        F.first("name").alias("name"),
        F.initcap(F.min_by("commonName", F.length("commonName"))).alias("common"),
        F.first("rank").alias("rank"),
        F.min("parent").alias("parent"),
    )
    clean = clean.withColumn("rank", F.coalesce(make_map(RANK_NAMES)[F.col("rank")], F.lit("other")))
    clean = clean.cache()

    parents = clean.select(F.col("taxon").alias("p_taxon"), F.col("parent").alias("p_parent"))
    lineage = clean.select("taxon", F.array("taxon").alias("path"), F.col("parent").alias("next"))
    for step in range(40):
        joined = lineage.join(parents, lineage.next == parents.p_taxon, "left")
        can_climb = F.col("p_taxon").isNotNull() & ~F.array_contains("path", F.col("next"))
        lineage = joined.select(
            "taxon",
            F.when(can_climb, F.concat("path", F.array("next"))).otherwise(F.col("path")).alias("path"),
            F.when(can_climb, F.col("p_parent")).alias("next"),
        ).localCheckpoint()
        if lineage.filter(F.col("next").isNotNull()).count() == 0:
            break

    orders = clean.filter(F.col("rank") == "order").select(F.col("taxon").alias("order_id"),
                                                           F.col("name").alias("order"))
    animal_order = (lineage.select("taxon", F.explode("path").alias("order_id"))
                           .join(orders, "order_id")
                           .select("taxon", "order")
                           .dropDuplicates(["taxon"]))
    taxa = clean.join(lineage.select("taxon", "path"), "taxon").join(animal_order, "taxon", "left")

    counts = features.groupBy("taxon").agg(
        F.countDistinct("prop").alias("features"),
        F.count(F.when(F.col("prop") == "P1843", True)).alias("languages"),
    )
    images = (features.filter(F.col("prop") == "P18")
                      .groupBy("taxon").agg(F.min("value").alias("image")))
    card_props = ["P141", "P9566", "P4214", "P3063"]
    card = (features.filter(F.col("prop").isin(card_props))
                    .groupBy("taxon").pivot("prop", card_props).agg(F.min("value"))
                    .withColumnRenamed("P141", "status")
                    .withColumnRenamed("P9566", "activity")
                    .withColumnRenamed("P4214", "lifespan")
                    .withColumnRenamed("P3063", "gestation")
                    .join(counts, "taxon", "right"))

    return {
        "taxa": taxa.toPandas(),
        "computers": computers,
        "images": dict(images.toPandas().values),
        "cards": card.toPandas().set_index("taxon"),
    }


def label(row):
    if isinstance(row["common"], str) and row["common"]:
        return f"{row['common']} ({row['name']})"
    return row["name"]


def open_lineage(taxon, name):
    st.session_state["clicked_id"] = taxon
    st.session_state["search"] = name.lower()


def show_card(taxon, animal_id):
    row = taxa.set_index("taxon").loc[taxon]
    info = data["cards"].loc[taxon] if taxon in data["cards"].index else None

    with st.container(border=True):
        picture = thumbnail(data["images"].get(taxon))
        if picture:
            st.image(picture.replace("?width=240", "?width=500"), width="stretch")
        st.subheader(short_label(row))
        st.caption(f"{row['name']} · {row['rank']}")

        by_id = taxa.set_index("taxon")
        for rank in ["class", "order", "family", "genus", "species"]:
            match = [t for t in row["path"] if by_id.loc[t, "rank"] == rank]
            st.write(f"**{rank.title()}:** {label(by_id.loc[match[0]]) if match else 'N/A'}")
        st.divider()

        facts = {
            "Conservation status": STATUS_NAMES.get(info["status"], info["status"]) if info is not None else None,
            "Active": ACTIVITY_NAMES.get(info["activity"], info["activity"]) if info is not None else None,
            "Longest recorded lifespan": info["lifespan"] if info is not None else None,
            "Gestation period": info["gestation"] if info is not None else None,
        }
        for key, value in facts.items():
            st.write(f"**{key}:** {value if pd.notna(value) and value != '' else 'N/A'}")
        if info is not None and (pd.notna(info["lifespan"]) or pd.notna(info["gestation"])):
            st.caption("Lifespan and gestation come without units in the download.")

        st.link_button("Open on Wikidata", f"https://www.wikidata.org/wiki/{taxon}")
        if taxon != animal_id:
            st.button("Open this lineage", type="primary", on_click=open_lineage, args=(taxon, row["name"]))


def short_label(row):
    if isinstance(row["common"], str) and row["common"]:
        return row["common"]
    return row["name"]


def thumbnail(url):
    if isinstance(url, str) and url.startswith("http"):
        return url.replace("http://", "https://") + "?width=240"
    return ""


def pick_relatives(group, images, how_many):
    group = group.copy()
    group["no_name"] = group["common"].isna()
    group["no_image"] = ~group["taxon"].isin(list(images))
    return group.sort_values(["no_name", "no_image"])["taxon"].head(how_many).tolist()


def tree_chart(animal_id, taxa, images, whole_tree):
    by_id = taxa.set_index("taxon")
    path = list(by_id.loc[animal_id, "path"])
    per_row = 6
    if whole_tree:
        ancestors = path[1:][::-1]
        child_limit, grandchild_limit = None, 120
    else:
        ancestors = path[1:3][::-1]
        child_limit, grandchild_limit = per_row, per_row

    siblings = pick_relatives(taxa[(taxa["parent"] == by_id.loc[animal_id, "parent"]) & (taxa["taxon"] != animal_id)],
                              images, 4)
    child_rows = taxa[taxa["parent"] == animal_id]
    children = pick_relatives(child_rows, images, child_limit or len(child_rows))
    grandchild_rows = taxa[taxa["parent"].isin(children)].copy()
    grandchild_rows["no_name"] = grandchild_rows["common"].isna()
    grandchild_rows["no_image"] = ~grandchild_rows["taxon"].isin(list(images))
    grandchild_rows = grandchild_rows.sort_values(["no_name", "no_image"])
    grandchild_rows["turn"] = grandchild_rows.groupby("parent").cumcount()
    grandchild_rows["parent_order"] = grandchild_rows["parent"].map(children.index)
    taken = grandchild_rows.sort_values(["turn", "parent_order"]).head(grandchild_limit)
    grandchildren = taken.sort_values(["parent_order", "turn"])["taxon"].tolist()
    hidden = (len(child_rows) - len(children)) + (len(grandchild_rows) - len(grandchildren))

    rows = []
    for y, taxon in enumerate(ancestors):
        rows.append({"taxon": taxon, "x": 0, "y": y, "kind": "ancestor"})
    middle = len(ancestors)
    rows.append({"taxon": animal_id, "x": 0, "y": middle, "kind": "selected"})
    for i, taxon in enumerate(siblings):
        side = -1 if i % 2 == 0 else 1
        rows.append({"taxon": taxon, "x": side * (i // 2 + 1), "y": middle, "kind": "sibling"})

    def add_level(level, first_row, kind):
        for i, taxon in enumerate(level):
            row, column = i // per_row, i % per_row
            in_this_row = min(per_row, len(level) - row * per_row)
            rows.append({"taxon": taxon, "x": column - (in_this_row - 1) / 2, "y": first_row + row, "kind": kind})
        return first_row + (len(level) + per_row - 1) // per_row

    after_children = add_level(children, middle + 1, "child")
    add_level(grandchildren, after_children, "grandchild")
    nodes = pd.DataFrame(rows)
    nodes["name"] = [short_label(by_id.loc[t]) for t in nodes["taxon"]]
    nodes["rank"] = [by_id.loc[t, "rank"] for t in nodes["taxon"]]
    nodes["image"] = [thumbnail(images.get(t)) for t in nodes["taxon"]]
    position = nodes.set_index("taxon")

    below_label = 0.45
    top_of_square = 0.28
    branch_gap = 0.36
    lines = []
    for row in range(len(ancestors) - 1):
        lines.append({"x": 0, "y": row + below_label, "x2": 0, "y2": row + 1 - top_of_square})
    if ancestors:
        for _, node in nodes[nodes["y"] == middle].iterrows():
            lines.append({"x": 0, "y": middle - 1 + below_label, "x2": node["x"], "y2": middle - top_of_square})
    kids = nodes[nodes["kind"] == "child"]
    if len(kids):
        lines.append({"x": 0, "y": middle + below_label, "x2": 0, "y2": kids["y"].max() - branch_gap})
    for row_y, row_kids in kids.groupby("y"):
        left, right = min(row_kids["x"].min(), 0), max(row_kids["x"].max(), 0)
        lines.append({"x": left, "y": row_y - branch_gap, "x2": right, "y2": row_y - branch_gap})
        for x_position in row_kids["x"]:
            lines.append({"x": x_position, "y": row_y - branch_gap, "x2": x_position, "y2": row_y - top_of_square})
    for _, node in nodes[nodes["kind"] == "grandchild"].iterrows():
        parent = position.loc[by_id.loc[node["taxon"], "parent"]]
        lines.append({"x": parent["x"], "y": parent["y"] + below_label, "x2": node["x"], "y2": node["y"] - top_of_square})
    lines = pd.DataFrame(lines, columns=["x", "y", "x2", "y2"])
    last_row = nodes["y"].max()

    half = max(per_row, 2 * (len(siblings) // 2 + len(siblings) % 2) + 1) / 2 + 0.2
    node = "min(width / 8.4 * 0.85, 120)"
    x = alt.X("x:Q", scale=alt.Scale(domain=[-half, half]), axis=None)
    y = alt.Y("y:Q", scale=alt.Scale(domain=[last_row + 0.7, -0.6]), axis=None)
    colors = alt.Scale(domain=["selected", "ancestor", "sibling", "child", "grandchild"],
                       range=["#16a34a", "#166534", "#334155", "#1e3a5f", "#312e5a"])
    click_box = alt.selection_point(name="click_box", fields=["taxon"], on="click")
    click_picture = alt.selection_point(name="click_picture", fields=["taxon"], on="click")

    edges = alt.Chart(lines).mark_rule(color="#94a3b8").encode(x=x, y=y, x2="x2:Q", y2="y2:Q")
    frames = alt.Chart(nodes).mark_square(size=alt.expr(f"pow({node} + 10, 2)"), opacity=1, cursor="pointer").encode(
        x=x, y=y, color=alt.Color("kind:N", scale=colors, legend=None)).add_params(click_box)
    pictures = alt.Chart(nodes[nodes["image"] != ""]).mark_image(
        width=alt.expr(node), height=alt.expr(node), cursor="pointer").encode(
        x=x, y=y, url="image:N").add_params(click_picture)
    names = alt.Chart(nodes).mark_text(dy=alt.expr(f"{node} / 2 + 18"), fontSize=13, color="#ffffff").encode(
        x=x, y=y, text="name:N")
    ranks = alt.Chart(nodes).mark_text(dy=alt.expr(f"{node} / 2 + 34"), fontSize=11, color="#cbd5e1").encode(
        x=x, y=y, text="rank:N")
    chart = alt.layer(edges, frames, pictures, names, ranks).properties(height=250 * (last_row + 1))
    return chart, hidden


st.set_page_config(page_title="Tree of Life", layout="wide")
st.title("Tree of Life")

with st.spinner("Spark is processing the data..."):
    data = load_data()
taxa = data["taxa"]

st.sidebar.header("PySpark Cluster")
st.sidebar.write(f"Master: {MASTER}")
st.sidebar.write("Workers:")
for computer in data["computers"]:
    st.sidebar.write(f"- {computer}")

search = st.text_input("Search", "snow leopard", key="search").lower()
found = taxa[taxa["common"].str.lower().str.contains(search, na=False)
             | taxa["name"].str.lower().str.contains(search, na=False)].copy()
found["exact"] = (found["common"].str.lower() == search) | (found["name"].str.lower() == search)
found["is_species"] = found["rank"] == "species"
found = found.sort_values(["exact", "is_species"], ascending=False).head(30)

if found.empty:
    st.write("No animal found.")
else:
    choices = dict(zip(found.apply(label, axis=1), found["taxon"]))
    ids = list(choices.values())
    start = ids.index(st.session_state["clicked_id"]) if st.session_state.get("clicked_id") in ids else 0
    picked = st.selectbox("Select a taxon", list(choices), index=start, key=f"pick_{search}")
    animal_id = choices[picked]

    st.caption("Click on a taxon in the tree to see info.")
    whole_tree_text = "Show expanded tree"
    whole_tree = st.toggle(whole_tree_text)
    chart, hidden = tree_chart(animal_id, taxa, data["images"], whole_tree)
    tree_column, card_column = st.columns([3, 1])
    with tree_column:
        event = st.altair_chart(chart, width="stretch", on_select="rerun", key=f"tree_{animal_id}")
        if hidden > 0:
            st.caption(f"{hidden} more groups below are not shown. "
                       + (f"Turn on '{whole_tree_text}' to see more." if not whole_tree
                          else "Click on a node above to explore it."))
    clicked = (event.selection.get("click_box") or []) + (event.selection.get("click_picture") or [])
    with card_column:
        show_card(clicked[0]["taxon"] if clicked else animal_id, animal_id)

    by_id = taxa.set_index("taxon")
    steps = [by_id.loc[t] for t in by_id.loc[animal_id, "path"]]
    st.write("Full lineage:")
    st.dataframe(pd.DataFrame({"rank": [s["rank"] for s in steps],
                               "name": [label(s) for s in steps]}), hide_index=True)