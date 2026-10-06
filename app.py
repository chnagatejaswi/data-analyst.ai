import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import ollama
except ImportError:
    ollama = None

st.set_page_config(page_title="DataPilot AI", layout="wide")

# ---------- Color system (60-30-10, split-complementary) ----------
BG, SURFACE, PRIMARY = "#0E1020", "#1A1D3A", "#6C63FF"
AMBER, TEAL, TEXT, MUTED = "#FFB84D", "#2EC4B6", "#EDEEFF", "#9A9CC4"
SEQ = [PRIMARY, AMBER, TEAL, "#FF6B8A", "#A29BFE", "#7BDFF2"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: {BG}; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {PRIMARY}33; }}
.hero {{
  background: linear-gradient(135deg, {PRIMARY} 0%, #3B2FC9 60%, {BG} 100%);
  padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.5rem;
  box-shadow: 0 10px 40px {PRIMARY}44;
}}
.hero h1 {{ margin: 0; font-size: 2.2rem; color: #fff; }}
.hero p {{ margin: .4rem 0 0; color: #DCDDFF; }}
.badge {{ display:inline-block; background:{AMBER}; color:{BG}; font-weight:700;
  padding:2px 12px; border-radius:99px; font-size:.75rem; margin-bottom:.6rem; }}
.card {{ background:{SURFACE}; border:1px solid {PRIMARY}44; border-radius:16px;
  padding:1.1rem 1.3rem; text-align:center; }}
.card .num {{ font-size:1.9rem; font-weight:700; color:{AMBER}; }}
.card .lbl {{ font-size:.8rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.08em; }}
.stButton>button {{ background:{AMBER}; color:{BG}; font-weight:600; border:none;
  border-radius:10px; padding:.5rem 1.2rem; }}
.stButton>button:hover {{ background:{TEAL}; color:{BG}; }}
.stTabs [data-baseweb="tab"] {{ color:{MUTED}; font-weight:600; }}
.stTabs [aria-selected="true"] {{ color:{TEAL}; }}
h2, h3 {{ color:{TEXT}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <span class="badge">LOCAL AI - PRIVATE</span>
  <h1>DataPilot AI</h1>
  <p>Upload a dataset, explore it visually, and ask a local Ollama model for insights.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Data ----------
@st.cache_data
def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "category": rng.choice(["Electronics", "Fashion", "Grocery", "Home"], n),
            "units": rng.integers(5, 120, n),
            "price": rng.uniform(10, 500, n).round(2),
            "discount": rng.choice([0, 5, 10, 20], n),
        }
    )
    df["revenue"] = (df["units"] * df["price"] * (1 - df["discount"] / 100)).round(2)
    return df


def themed(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT, colorway=SEQ, margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor=f"{PRIMARY}22")
    fig.update_yaxes(gridcolor=f"{PRIMARY}22")
    return fig


def list_models():
    if ollama is None:
        return []
    try:
        res = ollama.list()
        models = res["models"] if isinstance(res, dict) else res.models
        return [(m.get("model") or m.get("name")) if isinstance(m, dict) else m.model for m in models]
    except Exception:
        return []


def data_context(df: pd.DataFrame) -> str:
    return (
        f"Rows: {len(df)}, Columns: {list(df.columns)}\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n"
        f"Summary stats:\n{df.describe(include='all').T.head(15).to_string()}\n"
        f"Sample rows:\n{df.head(5).to_string()}"
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Setup")
    file = st.file_uploader("Upload CSV", type="csv")
    models = list_models()
    model = st.selectbox("Ollama model", models) if models else st.text_input("Ollama model", "llama3.2")
    if not models:
        st.warning("Ollama not detected. Run `ollama serve` and `ollama pull llama3.2`.")

df = pd.read_csv(file) if file else sample_data()
for c in df.columns:
    if "date" in c.lower():
        df[c] = pd.to_datetime(df[c], errors="coerce")
num_cols = df.select_dtypes("number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

tab1, tab2, tab3 = st.tabs(["Overview", "Explore", "AI Analyst"])

# ---------- Overview ----------
with tab1:
    stats = [
        (f"{len(df):,}", "Rows"),
        (len(df.columns), "Columns"),
        (int(df.isna().sum().sum()), "Missing values"),
        (int(df.duplicated().sum()), "Duplicates"),
    ]
    for col, (num, lbl) in zip(st.columns(4), stats):
        col.markdown(f'<div class="card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Data preview")
    st.dataframe(df.head(50), use_container_width=True)
    if len(num_cols) > 1:
        st.subheader("Correlation heatmap")
        fig = px.imshow(df[num_cols].corr(), text_auto=".2f",
                        color_continuous_scale=[TEAL, BG, AMBER], zmin=-1, zmax=1)
        st.plotly_chart(themed(fig), use_container_width=True)

# ---------- Explore ----------
with tab2:
    kind = st.radio("Chart", ["Histogram", "Bar (grouped)", "Scatter", "Line"], horizontal=True)
    if kind == "Histogram" and num_cols:
        x = st.selectbox("Column", num_cols)
        fig = px.histogram(df, x=x, nbins=30, color_discrete_sequence=[PRIMARY])
    elif kind == "Bar (grouped)" and cat_cols and num_cols:
        g = st.selectbox("Group by", cat_cols)
        v = st.selectbox("Value", num_cols)
        agg = df.groupby(g, as_index=False)[v].sum()
        fig = px.bar(agg, x=g, y=v, color=g, color_discrete_sequence=SEQ)
    elif kind == "Scatter" and len(num_cols) > 1:
        x = st.selectbox("X", num_cols)
        y = st.selectbox("Y", num_cols, index=1)
        c = st.selectbox("Color by", [None] + cat_cols)
        fig = px.scatter(df, x=x, y=y, color=c, color_discrete_sequence=SEQ, opacity=0.75)
    elif kind == "Line" and num_cols:
        x = st.selectbox("X axis", df.columns)
        y = st.selectbox("Y axis", num_cols)
        fig = px.line(df.sort_values(x), x=x, y=y, color_discrete_sequence=[TEAL])
    else:
        fig = None
        st.info("This dataset doesn't have the right column types for that chart.")
    if fig:
        st.plotly_chart(themed(fig), use_container_width=True)

# ---------- AI Analyst ----------
with tab3:
    st.subheader("Ask your data")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    q1, q2, q3 = st.columns(3)
    quick = None
    if q1.button("Summarize dataset"):
        quick = "Give a concise summary of this dataset and its key patterns."
    if q2.button("Find data issues"):
        quick = "List data quality issues and how to clean them."
    if q3.button("Suggest analyses"):
        quick = "Suggest 5 useful analyses or ML models for this data."

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("e.g. Which region performs best and why?") or quick
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        system = (
            "You are a senior data scientist. Answer using ONLY the dataset info below. "
            "Be concise, use bullet points, and suggest Python/pandas code when helpful.\n\n"
            + data_context(df)
        )
        msgs = [{"role": "system", "content": system}] + st.session_state.chat

        def stream():
            for chunk in ollama.chat(model=model, messages=msgs, stream=True):
                yield chunk["message"]["content"]

        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(stream())
                st.session_state.chat.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Could not reach Ollama: {e}")