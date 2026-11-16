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
import requests
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MASTER = "local[2]" # "spark://MASTER_IP" OR "local[2]""
MY_IP = "10.157.240.163" # <--- CHANGE THIS!
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
    builder = SparkSession.builder.appName("TreeOfLife").master(MASTER).config("spark.sql.shuffle.partitions", "8")
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

# Data loading/cleaning

@st.cache_data
def load_data():
    spark = get_spark()

    animals = spark.read.csv(f"file:///{FOLDER}/data/*.csv", header=True, escape='"')
    animals = (animals.withColumn("taxon", short_id("taxon"))
                      .withColumn("parent", short_id("parent"))
                      .withColumn("rank", short_id("rank")))

    features = spark.read.csv(f"file:///{FOLDER}/features/*.csv", header=True, escape='"')
    features = (features.withColumn("taxon", short_id("taxon"))
                        .withColumn("prop", short_id("prop"))
                        .withColumn("value", short_id("value")))

    last_download = animals.select(F.max("_metadata.file_modification_time")).first()[0]

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
    classes = clean.filter(F.col("rank") == "class").select(F.col("taxon").alias("class_id"),
                                                            F.col("name").alias("class"))
    animal_class = (lineage.select("taxon", F.posexplode("path").alias("pos", "class_id"))
                           .join(classes, "class_id")
                           .groupBy("taxon").agg(F.min_by("class", "pos").alias("class")))
    taxa = (clean.join(lineage.select("taxon", "path"), "taxon")
                 .join(animal_order, "taxon", "left")
                 .join(animal_class, "taxon", "left"))

    file_taxa = (animals.select("taxon", F.input_file_name().alias("file")).distinct()
                        .filter(~F.col("file").endswith("spine.csv"))
                        .join(lineage.select("taxon", "path"), "taxon"))
    in_file = (file_taxa.select("file", "taxon", F.explode("path").alias("ancestor"))
                        .join(file_taxa.select("file", F.col("taxon").alias("ancestor")), ["file", "ancestor"]))
    below = in_file.groupBy("file", "ancestor").count().withColumnRenamed("ancestor", "taxon")
    above = in_file.groupBy("file", "taxon").agg((F.count("*") - 1).alias("above"))
    total = file_taxa.groupBy("file").agg(F.count("*").alias("total"))
    roots = (file_taxa.join(below, ["file", "taxon"]).join(above, ["file", "taxon"]).join(total, "file")
                      .filter((F.col("count") + F.col("above")) >= 0.99 * F.col("total")))
    downloaded = (roots.groupBy("file").agg(F.max_by("taxon", F.size("path")).alias("root_id"))
                       .join(clean.select(F.col("taxon").alias("root_id"), F.col("name").alias("class")), "root_id"))

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
    species = taxa.filter(F.col("rank") == "species")

    volume = species.filter(F.col("order").isNotNull()).groupBy("class", "order").count()

    selected = ["P141", "P9566", "P4214", "P3063", "P18", "P181", "P2067", "P443", "P10", "P51", "P4896"]
    value_kind = (F.when(F.col("value").rlike(r"^Q[0-9]+$"), "Link to another item")
                   .when(F.lower("value").rlike(r"\.(jpg|jpeg|png|gif|svg|tif|tiff|webp)$"), "Image")
                   .when(F.lower("value").rlike(r"\.(ogg|oga|wav|mp3|flac|opus)$"), "Audio")
                   .when(F.lower("value").rlike(r"\.(webm|ogv|mp4|mpg|mpeg)$"), "Video")
                   .when(F.lower("value").rlike(r"\.stl$"), "3D model")
                   .when(F.col("value").rlike(r"^-?[0-9]+(\.[0-9]+)?$"), "Number")
                   .otherwise("Other"))
    variety = (features.filter(F.col("prop").isin(selected))
                       .join(taxa.select("taxon", "class"), "taxon")
                       .filter(F.col("class").isNotNull())
                       .withColumn("kind", value_kind)
                       .groupBy("class", "kind").count())

    feature_names = {"P2067": "Body mass", "P18": "Photo", "P141": "Conservation status",
                     "P9566": "Activity pattern", "P3063": "Gestation", "P4214": "Lifespan"}
    values_per_feature = (features.filter(F.col("prop").isin(list(feature_names)))
                                  .groupBy("prop", "taxon").agg(F.countDistinct("value").alias("values"))
                                  .join(taxa.select("taxon", "class"), "taxon")
                                  .filter(F.col("class").isNotNull()))
    conflicts = (values_per_feature.groupBy("class", "prop").agg(
                     F.count("*").alias("animals"),
                     F.sum((F.col("values") > 1).cast("int")).alias("several_values"))
                 .withColumn("feature", make_map(feature_names)[F.col("prop")]))

    return {
        "volume": volume.toPandas(),
        "variety": variety.toPandas(),
        "conflicts": conflicts.toPandas(),
        "taxa": taxa.toPandas(),
        "last_download": last_download,
        "downloads": downloaded.toPandas(),
        "computers": computers,
        "images": dict(images.toPandas().values),
        "cards": card.toPandas().set_index("taxon"),
    }


def refresh_classes(class_ids, files):
    values = " ".join(f"wd:{class_id}" for class_id in class_ids)
    query = f"""SELECT ?taxon ?name WHERE {{ VALUES ?class {{ {values} }} ?taxon wdt:P171+ ?class ; wdt:P225 ?name . }}"""
    response = requests.get("https://query.wikidata.org/sparql", params={"query": query, "format": "json"},
                            headers={"User-Agent": "TreeOfLifeStudentProject/1.0"}, timeout=60)

    spark = get_spark()
    answer = F.from_json(F.lit(response.text),
                         "struct<results:struct<bindings:array<struct<taxon:struct<value:string>,name:struct<value:string>>>>>")
    new = (spark.range(1).select(F.explode(answer["results"]["bindings"]).alias("row"))
                .select(short_id("row.taxon.value").alias("taxon"), F.col("row.name.value").alias("name"))
                .groupBy("taxon").agg(F.first("name").alias("new_name")))
    old = (spark.read.csv(files, header=True, escape='"')
                .select(short_id("taxon").alias("taxon"), "name")
                .filter(~F.col("taxon").isin(class_ids))
                .groupBy("taxon").agg(F.first("name").alias("name")))

    both = old.join(new, "taxon", "full")
    changes = both.select("taxon", "name", "new_name", F.when(F.col("name").isNull(), "New")
                                                        .when(F.col("new_name").isNull(), "Removed")
                                                        .when(F.col("name") != F.col("new_name"), "Renamed")
                                                        .alias("change")).filter(F.col("change").isNotNull())
    counts = dict(changes.groupBy("change").count().collect())
    return old.count(), counts, changes.toPandas()


def label(row):
    if isinstance(row["common"], str) and row["common"]:
        return f"{row['common']} ({row['name']})"
    return row["name"]

def open_lineage(taxon, name):
    st.session_state["search"] = name.lower()
    st.session_state[f"pick_{name.lower()}"] = taxon

# Card display

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

        features = {
            "Conservation status": STATUS_NAMES.get(info["status"], info["status"]) if info is not None else None,
            "Active": ACTIVITY_NAMES.get(info["activity"], info["activity"]) if info is not None else None,
            "Longest recorded lifespan": info["lifespan"] if info is not None else None,
            "Gestation period": info["gestation"] if info is not None else None,
        }
        for key, value in features.items():
            st.write(f"**{key}:** {value if pd.notna(value) and value != '' else 'N/A'}")
        if info is not None and (pd.notna(info["lifespan"]) or pd.notna(info["gestation"])):
            st.caption("Lifespan and gestation come without units in the download.")

        st.link_button("Open on Wikidata", f"https://www.wikidata.org/wiki/{taxon}")
        if taxon != animal_id:
            st.button("Open this lineage", type="primary", on_click=open_lineage, args=(taxon, short_label(row)))

# Node label

def short_label(row):
    if isinstance(row["common"], str) and row["common"]:
        return row["common"]
    return row["name"]

# Node image

def thumbnail(url):
    if isinstance(url, str) and url.startswith("http"):
        return url.replace("http://", "https://") + "?width=240"
    return ""

# Tree generation

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

    siblings = taxa[(taxa["parent"] == by_id.loc[animal_id, "parent"]) & (taxa["taxon"] != animal_id)]["taxon"].tolist()[:4]
    child_rows = taxa[taxa["parent"] == animal_id]
    children = child_rows["taxon"].tolist()[:child_limit]
    grandchild_rows = taxa[taxa["parent"].isin(children)].copy()
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
    tags = alt.Chart(nodes).mark_rect(width=alt.expr(f"{node} + 10"), height=40,
                                      yOffset=alt.expr(f"{node} / 2 + 25")).encode(
        x=x, y=y, color=alt.Color("kind:N", scale=colors, legend=None))
    names = alt.Chart(nodes).mark_text(dy=alt.expr(f"{node} / 2 + 18"), fontSize=13, fontWeight="bold",
                                       color="white", limit=alt.expr(node)).encode(
        x=x, y=y, text="name:N")
    ranks = alt.Chart(nodes).mark_text(dy=alt.expr(f"{node} / 2 + 34"), fontSize=11, color="white").encode(
        x=x, y=y, text="rank:N")
    chart = alt.layer(edges, frames, tags, pictures, names, ranks).properties(height=250 * (last_row + 1))
    return chart, hidden


st.set_page_config(page_title="Tree of Life", layout="wide")
st.title("TREE OF LIFE")

with st.spinner("Spark is processing the data..."):
    data = load_data()
taxa = data["taxa"]

# PySpark sidebar shows workers and master connected

st.sidebar.header("PySpark Cluster")
st.sidebar.write(f"Master: {MASTER}")
st.sidebar.write("Workers:")
for computer in data["computers"]:
    st.sidebar.write(f"- {computer}")


tab1, tab2 = st.tabs(["Tree of Life", "Data Visualization"])

####################
# INTERACTIVE TREE #
####################

with tab1:
    labels = dict(zip(taxa["taxon"], taxa["common"].fillna(taxa["name"])))
    search_column, pick_column = st.columns(2)
    search = search_column.text_input("Search", "snow leopard", key="search").lower()
    found = taxa[taxa["common"].str.lower().str.contains(search, na=False)
                 | taxa["name"].str.lower().str.contains(search, na=False)]
    exact = (found["common"].str.lower() == search) | (found["name"].str.lower() == search)
    found = pd.concat([found[exact], found[~exact]]).head(30)

    if found.empty:
        pick_column.write("No animal found.")
    else:
        animal_id = pick_column.selectbox("Select a taxon", found["taxon"].tolist(), format_func=labels.get,
                                          key=f"pick_{search}")

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
        
######################
# DATA VISUALIZATION #
######################

with tab2:
    st.subheader("VOLUME: Species per Order")
    
    volume = data["volume"]
    
    chosen = st.multiselect("Classes", sorted(volume["class"].dropna().unique()), placeholder="All classes")
    
    if chosen:
        volume = volume[volume["class"].isin(chosen)]
    volume = volume.groupby("order", as_index=False)["count"].sum().sort_values("count", ascending=False)
    
    st.altair_chart(alt.Chart(volume.head(15)).mark_bar().encode(
        x=alt.X("count:Q", title="Number of species"), y=alt.Y("order:N", sort="-x", title="Order")), width="stretch")
    total = volume["count"].sum()
    top = volume.iloc[0]
    
    st.write(f"{top['order']} contains {top['count']:,} species out of {total:,} (~{top['count'] / total:.0%}) across {len(volume)} orders in the dataset.")

    st.subheader("VARIETY: Types of values in the database")

    st.write("A chart of the different types of data used in the dataset.")
    
    variety = data["variety"]
    
    kind_classes = st.multiselect("Classes", sorted(variety["class"].unique()), placeholder="All classes",
                                  key="variety_classes")
    
    if kind_classes:
        variety = variety[variety["class"].isin(kind_classes)]
    variety = variety.groupby("kind", as_index=False)["count"].sum()
    variety["share"] = variety["count"] / variety["count"].sum()
    
    st.altair_chart(alt.Chart(variety).mark_arc(innerRadius=60).encode(
        theta="count:Q",
        color=alt.Color("kind:N", title="Type of data"),
        tooltip=[alt.Tooltip("kind:N", title="Format"), alt.Tooltip("count:Q", title="Count"),
                 alt.Tooltip("share:Q", title="Share", format=".0%")]), width="stretch")
    

    st.subheader("VERACITY: Features with conflicting values")

    st.write("Some are unique terms for newborn vs adult, male vs female.")
    
    conflicts = data["conflicts"]
    
    conflict_classes = st.multiselect("Classes", sorted(conflicts["class"].unique()), placeholder="All classes",
                                      key="veracity_classes")
    
    if conflict_classes:
        conflicts = conflicts[conflicts["class"].isin(conflict_classes)]
    conflicts = conflicts.groupby("feature", as_index=False)[["animals", "several_values"]].sum()
    conflicts["share"] = conflicts["several_values"] / conflicts["animals"]
    
    st.altair_chart(alt.Chart(conflicts).mark_bar().encode(
        x=alt.X("share:Q", title="Species %", axis=alt.Axis(format="%")),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("animals:Q", title="Species with Feature"),
                 alt.Tooltip("several_values:Q", title="Species with >1 Value"),
                 alt.Tooltip("share:Q", title="Share", format=".1%")]), width="stretch")
    

    st.subheader("VELOCITY: Update the dataset")
    
    st.write(f"Last download: {data['last_download']:%Y-%m-%d %H:%M}")
    
    downloads = data["downloads"]
    class_names = sorted(downloads["class"].unique())
    chosen_classes = st.multiselect("Classes", class_names, default=class_names[:1], key="velocity_classes")
    
    if st.button("Refresh from Wikidata", disabled=not chosen_classes):
        picked = downloads[downloads["class"].isin(chosen_classes)]
        with st.spinner("Asking Wikidata..."):
            st.session_state["velocity"] = (chosen_classes, *refresh_classes(sorted(set(picked["root_id"])),
                                                                             picked["file"].tolist()))
    
    if "velocity" in st.session_state:
        checked_classes, local_count, counts, changes = st.session_state["velocity"]
        st.write(f"{', '.join(checked_classes)}: {local_count:,} groups in our download.")
        new_col, removed_col, renamed_col = st.columns(3)
        new_col.metric("New", counts.get("New", 0))
        removed_col.metric("Removed", counts.get("Removed", 0))
        renamed_col.metric("Renamed", counts.get("Renamed", 0))
        st.dataframe(changes[["change", "taxon", "name", "new_name"]], hide_index=True)