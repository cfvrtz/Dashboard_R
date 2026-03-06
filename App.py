import streamlit as st
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import plotly.graph_objects as go
import plotly.express as px
import requests
import tempfile

# =============================================================================
# CONFIG
# =============================================================================
DATA_URL = "https://github.com/cfvrtz/Dashboard_R/releases/download/v1.0/retiros_base_2601.parquet"

# ─────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────
AZUL_OSCURO = "#009FE3"
AZUL_MEDIO  = "#55B9E0"
AZUL_CLARO  = "#FCDB00"
ACENTO      = "#1abc9c"
GRIS_FONDO  = "#f0f4f8"
BLANCO      = "#ffffff"

TIPO_COLORS = {
    "Lunes a Viernes": AZUL_CLARO,
    "Sábado":          "#e67e22",
    "Domingo":         "#e74c3c",
    "Feriado":         ACENTO,
}

DIAS_SEMANA_LABEL = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
DIAS_SEMANA_COLORS = ["#009FE3","#FCDB00","#e67e22","#e74c3c","#8e44ad","#1abc9c","#2c3e50"]

TIPO_ORDER = ["Lunes a Viernes", "Sábado", "Domingo", "Feriado"]
DIA_ORDER  = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

# =============================================================================
# HOLIDAYS CL (con fallback si no existe la librería)
# =============================================================================
def _build_cl_holidays():
    import datetime as _dt
    try:
        import holidays as _hd
        cl = set()
        for yr in range(2018, 2031):
            cl.update(_hd.Chile(years=yr).keys())
        return cl
    except Exception:
        pass

    # Feriados fijos (fallback)
    fixed = [(1,1),(5,1),(5,21),(6,29),(7,16),(8,15),
             (9,18),(9,19),(10,12),(10,31),(11,1),(12,8),(12,25)]
    result = set()
    for yr in range(2018, 2031):
        for m, d in fixed:
            try:
                result.add(_dt.date(yr, m, d))
            except ValueError:
                pass
    return result

CL_HOLIDAYS = _build_cl_holidays()

# =============================================================================
# HELPERS
# =============================================================================
def periodo_label(ym):
    meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    y, m = divmod(int(ym), 100)
    return f"{meses[m-1]} {y}"

def kpi(col, label, value, sub="", sub2="", acento=False):
    cls = "kpi-card acento" if acento else "kpi-card"
    sub2_html = f'<div class="kpi-sub2">{sub2}</div>' if sub2 else ""
    col.markdown(f"""<div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {sub2_html}
    </div>""", unsafe_allow_html=True)

# =============================================================================
# DOWNLOAD / LOAD
# =============================================================================
@st.cache_resource(show_spinner="Descargando datos (primera vez ~1 min)…")
def get_parquet_path():
    resp = requests.get(DATA_URL, timeout=300, stream=True)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
        if chunk:
            tmp.write(chunk)
    tmp.close()
    return tmp.name

@st.cache_data(show_spinner=False)
def get_retiros(path):
    return sorted(
        pq.read_table(path, columns=["retiro"])
          .to_pandas()["retiro"]
          .dropna()
          .unique()
          .tolist()
    )

@st.cache_data(show_spinner=False)
def get_claves(path, retiro):
    t = pq.read_table(path, columns=["retiro","clave"], filters=[("retiro","=",retiro)])
    return sorted(t.to_pandas()["clave"].dropna().unique().tolist())

@st.cache_data(show_spinner="Cargando datos del retiro…")
def load_filtered(path, retiro, clave):
    t = pq.read_table(path, filters=[("retiro","=",retiro),("clave","=",clave)])
    df = t.to_pandas()

    if df.empty:
        return df

    # Limpieza base
    if "medida_kwh" in df.columns:
        df["medida_kwh"] = pd.to_numeric(df["medida_kwh"], errors="coerce").abs().fillna(0)

    df["hora_mensual"] = pd.to_numeric(df["hora_mensual"], errors="coerce")
    df["anio_mes"]     = pd.to_numeric(df["anio_mes"], errors="coerce")

    # Derivadas simples
    df["hora_dia"] = (df["hora_mensual"] - 1) % 24
    df["mes"]      = (df["anio_mes"] % 100).astype("Int64")
    df["semestre"] = df["mes"].apply(lambda m: "Oct–Mar" if m in [10,11,12,1,2,3] else "Abr–Sep")
    df["periodo_label"] = df["anio_mes"].apply(periodo_label)

    # ── FECHA ROBUSTA (FIX del error) ────────────────────────────────────────
    # Construye: base = YYYYMM01 y suma (hora_mensual-1) horas.
    # Filtra horas fuera del mes (si la base viene "padded" a 744 siempre, etc.)
    anio_mes = df["anio_mes"].astype("Int64")
    hora_m   = df["hora_mensual"].astype("Int64")

    base = pd.to_datetime(anio_mes.astype(str) + "01", format="%Y%m%d", errors="coerce")
    max_horas_mes = base.dt.days_in_month * 24

    valid = (
        base.notna() &
        hora_m.notna() &
        (hora_m >= 1) &
        (hora_m <= max_horas_mes)
    )

    # OJO: si tenías padding, acá se descartan filas inválidas.
    df = df.loc[valid].copy()
    base = base.loc[valid]
    hora_m = hora_m.loc[valid]

    df["fecha_hora"] = base + pd.to_timedelta(hora_m - 1, unit="h")
    df["fecha"] = df["fecha_hora"].dt.normalize()
    df["dia_del_mes"] = df["fecha_hora"].dt.day

    df["dia_semana"] = df["fecha_hora"].dt.dayofweek
    df["dia_semana_label"] = df["dia_semana"].map(DIAS_SEMANA_LABEL)

    es_feriado = df["fecha_hora"].dt.date.isin(CL_HOLIDAYS)

    # Vectorizado (rápido)
    df["tipo_detallado"] = np.select(
        [es_feriado, df["dia_semana"].eq(5), df["dia_semana"].eq(6)],
        ["Feriado", "Sábado", "Domingo"],
        default="Lunes a Viernes"
    )

    return df

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="Dashboard Retiros", layout="wide", page_icon="⚡")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .main {{ background-color: {GRIS_FONDO}; }}
    .block-container {{ padding-top: 1.5rem; }}
    .header-box {{
        background: linear-gradient(135deg, {AZUL_OSCURO} 0%, {AZUL_MEDIO} 60%, {AZUL_CLARO} 100%);
        border-radius: 12px; padding: 22px 32px; margin-bottom: 24px; color: white;
        box-shadow: 0 4px 16px rgba(15,41,66,.25);
    }}
    .header-box h2 {{ margin:0; font-size:1.4rem; font-weight:700; letter-spacing:-.02em; }}
    .header-box p  {{ margin:6px 0 0 0; font-size:.88rem; opacity:.80; }}
    .header-box b  {{ opacity:1; font-weight:600; }}
    .kpi-card {{
        background:{BLANCO}; border-radius:10px; padding:18px 16px 14px;
        box-shadow:0 2px 8px rgba(0,0,0,.07); text-align:center;
        border-top:3px solid {AZUL_CLARO};
    }}
    .kpi-label {{ font-size:.68rem; color:#7f8c8d; text-transform:uppercase;
                  letter-spacing:.08em; font-weight:600; margin-bottom:6px; }}
    .kpi-value {{ font-size:1.55rem; font-weight:700; color:{AZUL_OSCURO}; line-height:1.1; }}
    .kpi-sub   {{ font-size:.72rem; color:#95a5a6; margin-top:4px; }}
    .kpi-sub2  {{ font-size:.70rem; color:#bdc3c7; margin-top:2px; font-style:italic; }}
    .kpi-card.acento {{ border-top-color:{ACENTO}; }}
    .kpi-card.acento .kpi-value {{ color:#0e6655; }}
    .section-title {{
        font-size:.9rem; font-weight:700; color:{AZUL_OSCURO};
        border-left:4px solid {ACENTO}; padding-left:10px;
        margin:28px 0 14px 0; text-transform:uppercase; letter-spacing:.05em;
    }}
    section[data-testid="stSidebar"] {{ background:{AZUL_OSCURO}; }}
    section[data-testid="stSidebar"] * {{ color:white !important; }}
    section[data-testid="stSidebar"] .stSelectbox > div > div {{
        background:rgba(255,255,255,.1) !important;
        border:1px solid rgba(255,255,255,.2) !important;
        border-radius:8px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ─── Descargar ───
try:
    parquet_path = get_parquet_path()
except Exception as e:
    st.error(f"Error al descargar el archivo: {e}")
    st.stop()

# ─── Sidebar ───
with st.sidebar:
    st.markdown("### ⚡ Dashboard Retiros")
    st.markdown("---")
    retiros    = get_retiros(parquet_path)
    sel_retiro = st.selectbox("Retiro", retiros)
    claves     = get_claves(parquet_path, sel_retiro)
    sel_clave  = st.selectbox("Clave", claves)
    st.markdown("---")
    st.caption(f"📦 {len(retiros):,} retiros disponibles")

# ─── Datos filtrados ───
df = load_filtered(parquet_path, sel_retiro, sel_clave)

if df.empty:
    st.warning("No hay datos para esta selección (o se descartaron todas las horas inválidas del mes).")
    st.stop()

# ─── Header ───
barra_val = df["barra"].iloc[0] if "barra" in df.columns else "—"
suministradores = df["suministrador"].dropna().unique().tolist() if "suministrador" in df.columns else []
sum_str = " &nbsp;·&nbsp; ".join(f"<b>{s}</b>" for s in sorted(suministradores))

st.markdown(f"""
<div class="header-box">
  <h2>⚡ &nbsp;{sel_retiro}</h2>
  <p>Clave: <b>{sel_clave}</b> &nbsp;·&nbsp; Barra: <b>{barra_val}</b></p>
  <p style="margin-top:8px; font-size:.85rem;">
    🏢 Suministrador{'es' if len(suministradores)>1 else ''}: {sum_str}
  </p>
</div>""", unsafe_allow_html=True)

# =============================================================================
# KPIs
# =============================================================================
mensual = df.groupby(["anio_mes","periodo_label"])["medida_kwh"].sum().reset_index()
mensual.columns = ["ym","label","total_kwh"]
mensual["total_mwh"] = mensual["total_kwh"] / 1000

total_anual  = mensual["total_kwh"].sum() / 1_000_000
promedio_mes = mensual["total_kwh"].mean() / 1000
min_mes      = mensual.loc[mensual["total_kwh"].idxmin()]
max_mes      = mensual.loc[mensual["total_kwh"].idxmax()]

hora_pico_max = int(df[df["anio_mes"]==max_mes["ym"]].groupby("hora_dia")["medida_kwh"].mean().idxmax())
hora_pico_min = int(df[df["anio_mes"]==min_mes["ym"]].groupby("hora_dia")["medida_kwh"].mean().idxmax())

solar_kwh = df[df["hora_dia"].between(8,17)]["medida_kwh"].sum()
pct_solar  = solar_kwh / df["medida_kwh"].sum() * 100 if df["medida_kwh"].sum() > 0 else 0

mes_col    = (df["anio_mes"] % 100).astype(int)
punta_mask = mes_col.between(4,9) & df["hora_dia"].between(18,22)
punta_kwh  = df[punta_mask]["medida_kwh"].sum()
punta_pct  = punta_kwh / df["medida_kwh"].sum() * 100 if df["medida_kwh"].sum() > 0 else 0

k1,k2,k3,k4,k5,k6 = st.columns(6)
kpi(k1, "Energía Anual",        f"{total_anual:.3f} GWh")
kpi(k2, "Promedio Mensual",     f"{promedio_mes:,.0f} MWh")
kpi(k3, "Menor Consumo",        f"{min_mes['total_kwh']/1000:,.0f} MWh",
    sub=min_mes['label'], sub2=f"Hora pico: {hora_pico_min:02d}:00 h")
kpi(k4, "Mayor Consumo",        f"{max_mes['total_kwh']/1000:,.0f} MWh",
    sub=max_mes['label'], sub2=f"Hora pico: {hora_pico_max:02d}:00 h")
kpi(k5, "Bloque Solar 08–17h",  f"{pct_solar:.1f}%", sub=f"{solar_kwh/1_000_000:.2f} GWh")
kpi(k6, "H. Punta Abr–Sep 18–22h", f"{punta_kwh/1_000_000:.3f} GWh",
    sub=f"{punta_pct:.1f}% del total", acento=True)

# =============================================================================
# Consumo Mensual
# =============================================================================
st.markdown('<div class="section-title">Consumo Mensual</div>', unsafe_allow_html=True)

mensual_sum = df.groupby(["anio_mes","periodo_label","suministrador"])["medida_kwh"].sum().reset_index()
mensual_sum["total_mwh"] = mensual_sum["medida_kwh"] / 1000
sums_list   = sorted(mensual_sum["suministrador"].dropna().unique())
colores_sum = px.colors.qualitative.Set2

fig_bar = go.Figure()
for i, s in enumerate(sums_list):
    d = mensual_sum[mensual_sum["suministrador"]==s].sort_values("anio_mes")
    fig_bar.add_trace(go.Bar(
        x=d["periodo_label"], y=d["total_mwh"], name=s,
        marker_color=colores_sum[i % len(colores_sum)],
        hovertemplate="%{x}<br>" + s + ": %{y:,.0f} MWh<extra></extra>",
    ))

fig_bar.update_layout(
    barmode="stack", height=350, margin=dict(t=20,b=10,l=50,r=20),
    plot_bgcolor=BLANCO, paper_bgcolor=GRIS_FONDO,
    yaxis=dict(title="MWh", gridcolor="#e8ecf0", tickformat=",.0f"),
    xaxis=dict(tickangle=-30),
    legend=dict(orientation="h", y=-0.25, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
)

for _, row in mensual.iterrows():
    fig_bar.add_annotation(
        x=row["label"], y=row["total_mwh"], text=f"{row['total_mwh']:,.0f}",
        showarrow=False, yanchor="bottom", font=dict(size=10, color=AZUL_OSCURO), yshift=3,
    )

st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# Curvas por semestre (4 tipos de día)
# =============================================================================
st.markdown('<div class="section-title">Curvas de Consumo por Hora y Tipo de Día</div>',
            unsafe_allow_html=True)

col_oct, col_abr = st.columns(2)
for col_ui, sem in [(col_oct,"Oct–Mar"),(col_abr,"Abr–Sep")]:
    df_sem = df[df["semestre"] == sem]
    if df_sem.empty:
        col_ui.info(f"Sin datos para {sem}")
        continue

    curva = df_sem.groupby(["hora_dia","tipo_detallado"])["medida_kwh"].mean().reset_index()
    top   = curva["medida_kwh"].max() or 1

    fig_c = go.Figure()
    for tipo in TIPO_ORDER:
        d = curva[curva["tipo_detallado"]==tipo].sort_values("hora_dia")
        if d.empty:
            continue
        fig_c.add_trace(go.Scatter(
            x=d["hora_dia"], y=d["medida_kwh"]/top*100,
            mode="lines", name=tipo,
            line=dict(color=TIPO_COLORS[tipo], width=2.5),
            hovertemplate="%{x}h: %{y:.1f}%<extra>" + tipo + "</extra>",
        ))

    fig_c.update_layout(
        title=dict(text=sem, font=dict(size=13, color=AZUL_OSCURO, family="DM Sans")),
        height=300, margin=dict(t=40,b=10,l=45,r=10),
        plot_bgcolor=BLANCO, paper_bgcolor=GRIS_FONDO,
        yaxis=dict(title="%", ticksuffix="%", range=[0,110], gridcolor="#e8ecf0"),
        xaxis=dict(title="Hora", dtick=2, range=[-0.5,23.5], gridcolor="#e8ecf0",
                   tickvals=list(range(0,24,2))),
        legend=dict(font=dict(size=10), orientation="h", y=-0.3, bgcolor="rgba(0,0,0,0)"),
    )
    col_ui.plotly_chart(fig_c, use_container_width=True)

# =============================================================================
# Consumo Promedio por Día de Semana
# =============================================================================
st.markdown('<div class="section-title">Consumo Promedio por Hora y Día de la Semana [kWh]</div>',
            unsafe_allow_html=True)

diario = df.groupby(["hora_dia","dia_semana_label"])["medida_kwh"].mean().reset_index()

fig_d = go.Figure()
for i, dia in enumerate(DIA_ORDER):
    d = diario[diario["dia_semana_label"]==dia].sort_values("hora_dia")
    if d.empty:
        continue
    dash = "dot" if dia in ["Sábado","Domingo"] else "solid"
    fig_d.add_trace(go.Scatter(
        x=d["hora_dia"], y=d["medida_kwh"],
        mode="lines+markers", name=dia,
        line=dict(color=DIAS_SEMANA_COLORS[i], width=2.2, dash=dash),
        marker=dict(size=4),
        hovertemplate="%{x}h: %{y:,.1f} kWh<extra>" + dia + "</extra>",
    ))

fig_d.update_layout(
    height=360, margin=dict(t=20,b=10,l=65,r=20),
    plot_bgcolor=BLANCO, paper_bgcolor=GRIS_FONDO,
    yaxis=dict(title="kWh", gridcolor="#e8ecf0", tickformat=",.0f"),
    xaxis=dict(title="Hora", dtick=1, range=[-0.5,23.5], gridcolor="#e8ecf0",
               tickvals=list(range(0,24))),
    legend=dict(font=dict(size=11), orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_d, use_container_width=True)

# =============================================================================
# Tabla histórico
# =============================================================================
st.markdown('<div class="section-title">Histórico por Período</div>', unsafe_allow_html=True)

hist = df.groupby(["anio_mes","periodo_label"]).agg(
    energia_kwh    =("medida_kwh","sum"),
    potencia_max_kw=("medida_kwh","max"),
).reset_index().sort_values("anio_mes", ascending=False)

solar_pm = df[df["hora_dia"].between(8,17)].groupby("anio_mes")["medida_kwh"].sum()/1000
noche_pm = df[~df["hora_dia"].between(8,17)].groupby("anio_mes")["medida_kwh"].sum()/1000

hist["Energía MWh"]     = (hist["energia_kwh"]/1000).map("{:,.1f}".format)
hist["Solar 08–17 MWh"] = hist["anio_mes"].map(solar_pm).map(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
hist["Noche 18–07 MWh"] = hist["anio_mes"].map(noche_pm).map(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
hist["Potencia Máx kW"] = hist["potencia_max_kw"].map("{:,.2f}".format)

st.dataframe(
    hist[["periodo_label","Energía MWh","Solar 08–17 MWh","Noche 18–07 MWh","Potencia Máx kW"]]
        .rename(columns={"periodo_label":"Período"}),
    use_container_width=True, hide_index=True
)

st.markdown("---")
st.caption("Dashboard de Retiros · Datos cargados desde GitHub Releases")
