"""
Dashboard Streamlit — Exploração do dataset dog-and-cat-data
Para rodar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import kagglehub
from pathlib import Path

st.set_page_config(page_title="Dog & Cat Data — Dashboard", layout="wide")



# ---------------------------------------------------------------------------
# Imagem de capa (fundo da página)
# ---------------------------------------------------------------------------

import base64

CAPA_PATH = "assets/capa.png"

def set_background(image_path: str):
    path = Path(image_path)
    if not path.exists():
        st.warning(f"Imagem de capa não encontrada em: {image_path}")
        return
    encoded = base64.b64encode(path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        :root {{
            --page-bg: #f8fafc;
            --panel-bg: rgba(255, 255, 255, 0.94);
            --panel-strong: #ffffff;
            --panel-soft: #f8fafc;
            --text-dark: #1e293b;
            --text-soft: #475569;
            --ink: #0f172a;
            --accent: #2563eb;
            --accent-strong: #1d4ed8;
            --accent-soft: #eff6ff;
            --blue: #2563eb;
            --blue-soft: #eff6ff;
            --green: #0f766e;
            --green-soft: #f0fdfa;
            --orange: #ea580c;
            --orange-soft: #fff7ed;
            --border-soft: #e2e8f0;
            --shadow-soft: rgba(15, 23, 42, 0.06);
        }}

        .stApp {{
            background-image: linear-gradient(135deg, rgba(255, 255, 255, 0.42), rgba(248, 250, 252, 0.58)),
                              url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: var(--text-dark);
            color-scheme: light;
        }}

        .stApp > header {{
            background-color: transparent;
        }}

        .block-container {{
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 1.5rem 1.75rem 1.25rem;
            box-shadow: 0 6px 18px var(--shadow-soft);
            backdrop-filter: blur(4px);
        }}

        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stCaption, .stDataFrame, .stSelectbox, .stSlider, .stMultiSelect,
        .stTabs [data-baseweb="tab-list"] button,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] div,
        [data-testid="stDataFrame"] *,
        .stDataFrame *,
        .stTable *,
        [role="listbox"] *,
        [role="option"] *,
        [data-baseweb="select"] *,
        [data-baseweb="slider"] *,
        [data-testid="stSidebar"] [role="option"] *,
        [data-testid="stSidebar"] [role="listbox"] *,
        div[role="combobox"],
        div[role="button"],
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stSlider > div > div {{
            color: var(--text-dark) !important;
        }}

        .stMarkdown p, .stCaption, .stDataFrame, .stSelectbox, .stSlider, .stMultiSelect {{
            color: var(--text-soft) !important;
        }}

        [data-testid="stDataFrame"],
        .stDataFrame,
        [data-testid="stDataFrame"] table,
        .stDataFrame table {{
            background: rgba(255, 255, 255, 0.96) !important;
            border: 1px solid rgba(148, 163, 184, 0.30);
            border-radius: 14px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
            color: #111827 !important;
        }}

        [data-testid="stDataFrame"] td,
        [data-testid="stDataFrame"] th,
        .stDataFrame td,
        .stDataFrame th {{
            color: #111827 !important;
            background: white !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="slider"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div,
        [data-testid="stSidebar"] .stSelectbox > div,
        [data-testid="stSidebar"] .stMultiSelect > div,
        [data-testid="stSidebar"] .stSlider > div,
        [data-testid="stSidebar"] .stNumberInput > div,
        [data-testid="stSidebar"] .stDateInput > div,
        [data-testid="stSidebar"] .stCheckbox > div,
        [data-testid="stSidebar"] .stRadio > div,
        [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"],
        [data-testid="stSlider"],
        [data-testid="stBaseInput"],
        [data-testid="stBaseSelect"],
        [data-testid="stBaseSlider"],
        div[role="combobox"],
        div[role="listbox"],
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [data-testid="stSidebar"] [role="listbox"],
        [data-testid="stSidebar"] [role="button"],
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] textarea {{
            background: white !important;
            border: 1px solid rgba(148, 163, 184, 0.45) !important;
            color: #111827 !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }}

        [role="option"],
        [role="listbox"] [role="option"] {{
            background: white !important;
            color: #111827 !important;
        }}
        [role="option"][aria-selected="true"] {{
            background: rgba(47, 111, 237, 0.12) !important;
            color: #111827 !important;
        }}

        input, select, textarea {{
            background: white !important;
            color: #111827 !important;
        }}

        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stNumberInput label,
        .stDateInput label,
        .stRadio label,
        .stCheckbox label {{
            color: #111827 !important;
            font-weight: 600;
        }}

        [data-testid="stMetricContainer"] {{
            background: var(--panel-strong);
            border: 1px solid var(--border-soft);
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            box-shadow: 0 3px 10px var(--shadow-soft);
        }}

        [data-testid="stMetricLabel"] {{
            font-weight: 600;
            color: var(--text-soft) !important;
        }}

        [data-testid="stMetricValue"] {{
            color: var(--ink) !important;
            font-weight: 700;
        }}

        .stButton > button {{
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}

        .stButton > button:hover {{
            background: #1d4ed8;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background(CAPA_PATH)

# ---------------------------------------------------------------------------
# Carregamento dos dados (com cache para não baixar/ler de novo a cada clique)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    dataset_path = Path(kagglehub.dataset_download("newbie2016/dog-and-cat-data"))

    # Procura o primeiro arquivo tabular (csv ou excel) dentro da pasta baixada
    candidates = list(dataset_path.rglob("*.csv")) + \
                 list(dataset_path.rglob("*.xlsx")) + \
                 list(dataset_path.rglob("*.xls"))

    if not candidates:
        return None, dataset_path

    file_path = candidates[0]
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    return df, file_path


st.title("🐶🐱 Dashboard — Dog & Cat Data")

with st.spinner("Carregando dados..."):
    df, source_path = load_data()

if df is None:
    st.error(f"Nenhum arquivo .csv/.xlsx encontrado em: {source_path}")
    st.stop()

# Métrica derivada usada na análise original do notebook.
# Ela precisa ser criada no app porque não vem pronta no arquivo baixado.
care_inputs = [
    "Avg exercise_minutes_daily",
    "Avg social_interaction_hours_daily",
    "Avg grooming_hours_monthly",
]
if "estimated_daily_care_minutes" not in df.columns and all(col in df.columns for col in care_inputs):
    df["estimated_daily_care_minutes"] = (
        df["Avg exercise_minutes_daily"]
        + 60 * df["Avg social_interaction_hours_daily"]
        + 2 * df["Avg grooming_hours_monthly"]
    )

st.caption(f"Fonte: `{source_path.name}` — {df.shape[0]} linhas × {df.shape[1]} colunas")
st.caption("* Hipoalergênica: classificação relativa, não garantia de ausência de alergias.")

# ---------------------------------------------------------------------------
# Sidebar — filtros dinâmicos baseados no tipo de cada coluna
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")

SIDEBAR_IMAGE_PATH = "assets/sidebar.png"

def set_sidebar_background(image_path: str):
    path = Path(image_path)
    if not path.exists():
        st.sidebar.warning(f"Imagem não encontrada em: {image_path}")
        return

    encoded = base64.b64encode(path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        /* =========================================================
           IMAGEM DE FUNDO DA SIDEBAR
           ========================================================= */
        [data-testid="stSidebar"] {{
            background-image:
                linear-gradient(
                    180deg,
                    rgba(255, 255, 255, 0.72),
                    rgba(255, 255, 255, 0.78)
                ),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(255, 255, 255, 0.28);
        }}

        /* =========================================================
           CAMPOS DOS FILTROS
           ========================================================= */
        [data-testid="stSidebar"] [data-baseweb="select"] {{
            background: #ffffff !important;
            color: #111827 !important;
            border-radius: 8px !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="input"] {{
            background: #ffffff !important;
            color: #111827 !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="input"] > div {{
            background: #ffffff !important;
            color: #111827 !important;
        }}

        /* Texto interno dos dropdowns */
        [data-testid="stSidebar"] [data-baseweb="select"] span {{
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            background: transparent !important;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] input {{
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            background: #ffffff !important;
            opacity: 1 !important;
        }}

        /* Placeholder "Selecione o filtro" */
        [data-testid="stSidebar"] input[placeholder="Selecione o filtro"] {{
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            background: #ffffff !important;
            opacity: 1 !important;
        }}

        [data-testid="stSidebar"] input[placeholder="Selecione o filtro"]::placeholder {{
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            opacity: 1 !important;
        }}

        /* Seta */
        [data-testid="stSidebar"] [data-baseweb="select"] svg {{
            fill: #111827 !important;
            color: #111827 !important;
        }}

        /* =========================================================
           MENU ABERTO
           ========================================================= */
        [data-baseweb="popover"] {{
            background: #ffffff !important;
        }}

        [data-baseweb="popover"] [role="option"] {{
            background: #ffffff !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }}

        [data-baseweb="popover"] [role="option"]:hover {{
            background: #eff6ff !important;
            color: #111827 !important;
        }}

        /* =========================================================
           RÓTULOS DOS FILTROS
           ========================================================= */
        [data-testid="stSidebar"] label {{
            color: #1e293b !important;
            font-weight: 600 !important;
            text-shadow: none !important;
        }}

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p {{
            color: #1e293b !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_sidebar_background(SIDEBAR_IMAGE_PATH)


filtered_df = df.copy()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Nomes mais amigáveis para apresentação
NOMES_AMIGAVEIS = {
    "species": "Espécie",
    "breed_name": "Raça",
    "senior_friendly_label": "Adequada para idosos",
    "adaptability": "Adaptabilidade",
    "urban_friendly_label": "Adequada para ambientes urbanos",
    "family_compatibility_label": "Compatibilidade com famílias",
    "obesity_prone_clean": "Propensão à obesidade",
    "hypoallergenic": "Hipoalergênica*",
    "heat_tolerance": "Tolerância ao calor",
    "cold_tolerance": "Tolerância ao frio",
    "humidity_tolerance": "Tolerância à umidade",
    "Avg exercise_minutes_daily": "Exercício diário (min)",
    "Avg grooming_hours_monthly": "Cuidados de higiene (h/mês)",
    "Avg social_interaction_hours_daily": "Interação social (h/dia)",
    "Avg total_monthly_cost_brl": "Custo mensal (R$)",
    "Avg total_monthly_cost_inr": "Custo mensal (INR)",
    "estimated_daily_care_minutes": "Cuidados diários (min)",
    "care_time_minutes_daily": "Cuidados diários (min)",
    "life_span": "Expectativa de vida",
    "health_issues": "Problemas de saúde",
    "urban_friendly": "Adequação urbana",
    "family_compatibility_score": "Compatibilidade familiar",
}

def nome_amigavel(coluna):
    return NOMES_AMIGAVEIS.get(coluna, coluna.replace("_", " ").replace("-", " ").strip().capitalize())

MAX_FILTERS = 5 # limite total de filtros exibidos na sidebar

# Colunas categóricas elegíveis (mais de 1 e no máximo 50 valores únicos)
eligible_categorical = [
    col for col in categorical_cols
    if 1 < df[col].dropna().nunique()
    and (df[col].dropna().nunique() <= 50 or col.lower() in ("species", "breed", "breed_name"))
]

preferred_categorical = [
    col for name in ("species", "breed_name", "breed")
    for col in eligible_categorical
    if col.lower() == name
]
remaining_categorical = [
    col for col in eligible_categorical if col not in preferred_categorical
]
# Colunas numéricas elegíveis (com variação de valores)
eligible_numeric = [
    col for col in numeric_cols
    if df[col].min() < df[col].max()
]

# Intercala categóricas e numéricas até atingir o limite de MAX_FILTERS
filter_cols = [("cat", col) for col in preferred_categorical]
i = j = 0
while len(filter_cols) < MAX_FILTERS and (i < len(remaining_categorical) or j < len(eligible_numeric)):
    if i < len(remaining_categorical):
        filter_cols.append(("cat", remaining_categorical[i]))
        i += 1
    if len(filter_cols) < MAX_FILTERS and j < len(eligible_numeric):
        filter_cols.append(("num", eligible_numeric[j]))
        j += 1

for kind, col in filter_cols:
    if kind == "cat":
        unique_vals = filtered_df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(nome_amigavel(col), sorted(map(str, unique_vals)), placeholder="Selecione o filtro")
        if selected:
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]
    else:
        col_min, col_max = float(df[col].min()), float(df[col].max())
        val_range = st.sidebar.slider(
            nome_amigavel(col), min_value=col_min, max_value=col_max, value=(col_min, col_max)
        )
        filtered_df = filtered_df[
            (filtered_df[col] >= val_range[0]) & (filtered_df[col] <= val_range[1])
        ]

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(filtered_df)} de {len(df)} linhas após os filtros")

# ---------------------------------------------------------------------------
# KPIs rápidos
# ---------------------------------------------------------------------------
kpi_cols = st.columns(4)
kpi_cols[0].metric("Linhas (filtradas)", len(filtered_df))
kpi_cols[1].metric("Colunas", df.shape[1])
kpi_cols[2].metric("Colunas numéricas", len(numeric_cols))
kpi_cols[3].metric("Colunas categóricas", len(categorical_cols))

# ---------------------------------------------------------------------------
# Gráficos — visão geral do contexto
# ---------------------------------------------------------------------------
st.subheader("Visão geral")


def encontrar_coluna(possiveis):
    for coluna in possiveis:
        if coluna in filtered_df.columns:
            return coluna
    return None


def estilo_base(chart, titulo, altura=340):
    """Padroniza tipografia, eixos e aparência dos gráficos Altair."""
    return (
        chart
        .properties(title=titulo, height=altura)
        .configure(background="#ffffff")
        .configure_title(
            color="#0f172a",
            fontSize=18,
            fontWeight=700,
            anchor="start",
            offset=14,
        )
        .configure_axis(
            labelColor="#475569",
            titleColor="#334155",
            labelFontSize=12,
            titleFontSize=13,
            titleFontWeight=600,
            gridColor="#e2e8f0",
            gridOpacity=0.55,
            domainColor="#cbd5e1",
            tickColor="#cbd5e1",
        )
        .configure_legend(
            labelColor="#475569",
            titleColor="#334155",
            labelFontSize=12,
            titleFontSize=12,
            orient="bottom",
        )
        .configure_view(strokeWidth=0)
    )


col_especie = encontrar_coluna(["species"])
col_custo = encontrar_coluna(["Avg total_monthly_cost_inr", "Avg total_monthly_cost_brl"])
col_cuidado = encontrar_coluna(["estimated_daily_care_minutes"])

# Dois gráficos de visão geral, lado a lado em telas largas.
grafico_col1, grafico_col2 = st.columns(2)

# ============================================================
# 1. Custo mensal médio por espécie
# ============================================================
with grafico_col1:
    if col_especie and col_custo:
        dados_custo = (
            filtered_df[[col_especie, col_custo]]
            .dropna()
            .groupby(col_especie, as_index=False)[col_custo]
            .mean()
            .rename(columns={col_especie: "Espécie", col_custo: "Custo mensal médio"})
            .sort_values("Custo mensal médio", ascending=False)
        )

        moeda = "INR" if col_custo.endswith("_inr") else "R$"

        barras_custo = (
            alt.Chart(dados_custo)
            .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
            .encode(
                x=alt.X("Espécie:N", title=None, sort="-y"),
                y=alt.Y(
                    "Custo mensal médio:Q",
                    title=f"Custo mensal médio ({moeda})",
                    scale=alt.Scale(zero=True),
                ),
                color=alt.Color("Espécie:N", title=None, legend=None),
                tooltip=[
                    alt.Tooltip("Espécie:N", title="Espécie"),
                    alt.Tooltip(
                        "Custo mensal médio:Q",
                        title=f"Custo médio ({moeda})",
                        format=",.1f",
                    ),
                ],
            )
        )

        rotulos_custo = (
            alt.Chart(dados_custo)
            .mark_text(dy=-10, fontSize=13, fontWeight=600, color="#334155")
            .encode(
                x=alt.X("Espécie:N", sort="-y"),
                y=alt.Y("Custo mensal médio:Q"),
                text=alt.Text("Custo mensal médio:Q", format=".0f"),
            )
        )

        grafico_custo = estilo_base(
            barras_custo + rotulos_custo,
            "Custo mensal médio por espécie",
            altura=330,
        )
        st.altair_chart(grafico_custo, use_container_width=True)

# ============================================================
# 2. Tempo diário estimado de cuidados por espécie
# ============================================================
with grafico_col2:
    if col_especie and col_cuidado:
        dados_cuidado = (
            filtered_df[[col_especie, col_cuidado]]
            .dropna()
            .groupby(col_especie, as_index=False)[col_cuidado]
            .mean()
            .rename(
                columns={
                    col_especie: "Espécie",
                    col_cuidado: "Minutos de cuidados por dia",
                }
            )
            .sort_values("Minutos de cuidados por dia", ascending=False)
        )

        barras_cuidado = (
            alt.Chart(dados_cuidado)
            .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
            .encode(
                x=alt.X("Espécie:N", title=None, sort="-y"),
                y=alt.Y(
                    "Minutos de cuidados por dia:Q",
                    title="Minutos por dia",
                    scale=alt.Scale(zero=True),
                ),
                color=alt.Color("Espécie:N", title=None, legend=None),
                tooltip=[
                    alt.Tooltip("Espécie:N", title="Espécie"),
                    alt.Tooltip(
                        "Minutos de cuidados por dia:Q",
                        title="Média diária",
                        format=".1f",
                    ),
                ],
            )
        )

        rotulos_cuidado = (
            alt.Chart(dados_cuidado)
            .mark_text(dy=-10, fontSize=13, fontWeight=600, color="#334155")
            .encode(
                x=alt.X("Espécie:N", sort="-y"),
                y=alt.Y("Minutos de cuidados por dia:Q"),
                text=alt.Text("Minutos de cuidados por dia:Q", format=".0f"),
            )
        )

        grafico_cuidado = estilo_base(
            barras_cuidado + rotulos_cuidado,
            "Tempo diário estimado de cuidados",
            altura=330,
        )
        st.altair_chart(grafico_cuidado, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabela de dados (apoio)
# ---------------------------------------------------------------------------
st.subheader("Dados filtrados")
tabela_exibicao = filtered_df.rename(columns=nome_amigavel)
st.dataframe(
    tabela_exibicao.style.set_properties(
        **{"color": "#1f2937", "background-color": "white"}
    ),
    use_container_width=True
)
