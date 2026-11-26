# -*- coding: utf-8 -*-
"""
Dashboard climático interactivo (Streamlit + Plotly)
Ejecutar:  streamlit run app.py   ->  http://localhost:8501
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Dashboard climático", layout="wide")

_csvs = sorted(Path(__file__).parent.glob("*.csv"), key=lambda q: q.stat().st_mtime, reverse=True)
CSV_POR_DEFECTO = _csvs[0] if _csvs else Path(__file__).parent / "Dataset Limpio2.csv"
data = pd.read_csv(CSV_POR_DEFECTO)

RENAME = {
    "DIRECCIÓN DEL VIENTO (m/s)": "dir",   # en realidad son grados (0-360)
    "VELOCIDAD DEL VIENTO (m/s)": "vel",
    "TEMPERATURA (°C)": "temp",
    "HUMEDAD (%)": "hum",
    "PRESIÓN (hPa)": "pres",
    "PRECIPITACIÓN (mm)": "prec",
}
data = pd.read_csv(CSV_POR_DEFECTO)

data = data.rename(columns={
    "TEMPERATURA (°C)": "temp",
    "HUMEDAD (%)": "hum",
    "PRESIÓN (hPa)": "pres",
    "VELOCIDAD DEL VIENTO (m/s)": "vel",
    "PRECIPITACIÓN (mm)": "prec"
})


LIMITES = {}

for variable in ["temp", "hum", "pres", "vel"]:
    q1 = data[variable].quantile(0.25)
    q3 = data[variable].quantile(0.75)

    riq = q3 - q1

    limite_inf = q1 - 1.5 * riq
    limite_sup = q3 + 1.5 * riq

    LIMITES[variable] = (limite_inf, limite_sup)

# Ajustes físicos
LIMITES["hum"] = (max(0, LIMITES["hum"][0]), 100)
LIMITES["vel"] = (0, LIMITES["vel"][1])

# Precipitación: calcular cuartiles solo cuando llueve
lluvia = data.loc[data["prec"] > 0, "prec"]

q1 = lluvia.quantile(0.25)
q3 = lluvia.quantile(0.75)
riq = q3 - q1

LIMITES["prec"] = (0, q3 + 1.5 * riq)

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS_MES = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
SECT = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]

C_MAX, C_MED, C_MIN = "#d1495b", "#edae49", "#00798c"
C_LLUVIA, C_VIENTO, C_TREND = "#3b82c4", "#2a9d8f", "#8d6cab"



@st.cache_data(show_spinner="Cargando y limpiando datos…")
def cargar_datos(fuente, marca, limpiar):
    """`marca` solo sirve para invalidar la caché cuando cambia el archivo."""
    if isinstance(fuente, bytes):
        fuente = io.BytesIO(fuente)
    df = pd.read_csv(fuente, encoding="utf-8-sig").rename(columns=RENAME)
    df["ESTACION"] = df["ESTACION"].astype(str).str.strip()
    df["t"] = pd.to_datetime(df["FECHA CONVERTIDA"])
    df = df.drop_duplicates(["ESTACION", "t"]).sort_values(["ESTACION", "t"]).reset_index(drop=True)

    informe = {}
    caido = df["hum"] < 1  # humedad 0 => sensor caído, se invalida también la temperatura
    informe["Sensor caído (humedad = 0)"] = int(caido.sum())
    if limpiar:
        df.loc[caido, ["temp", "hum"]] = np.nan
    nombres = {"temp": "Temperatura", "hum": "Humedad", "pres": "Presión",
               "vel": "Velocidad del viento", "prec": "Precipitación"}
    cero_exacto = df["temp"] == 0  # 0.00 °C exacto con 2 decimales = fallo del sensor
    informe["Temperatura exactamente 0.00 (fallo)"] = int(cero_exacto.sum())
    if limpiar:
        df.loc[cero_exacto, "temp"] = np.nan
    for col, (lo, hi) in LIMITES.items():
        malo = (df[col] < lo) | (df[col] > hi)
        informe[f"{nombres[col]} fuera de [{lo}, {hi}]"] = int(malo.sum())
        if limpiar:
            df.loc[malo, col] = np.nan

    df["fecha"] = df["t"].dt.normalize()
    df["mes_ts"] = df["t"].dt.to_period("M").dt.to_timestamp()
    df["anio"] = df["t"].dt.year
    df["mes"] = df["t"].dt.month
    df["hora"] = df["t"].dt.hour

    # Cambio de presión en las últimas 3 h y lluvia en las próximas 3 h (por estación)
    base = df[["ESTACION", "t", "pres", "prec"]]
    lag = base[["ESTACION", "t", "pres"]].copy()
    lag["t"] = lag["t"] + pd.Timedelta(hours=3)
    lag = lag.rename(columns={"pres": "pres_3h"})
    df["dP3"] = df["pres"] - df[["ESTACION", "t"]].merge(lag, on=["ESTACION", "t"], how="left")["pres_3h"].values
    futuros = []
    for k in (1, 2, 3):
        f = base[["ESTACION", "t", "prec"]].copy()
        f["t"] = f["t"] - pd.Timedelta(hours=k)
        futuros.append(df[["ESTACION", "t"]].merge(f, on=["ESTACION", "t"], how="left")["prec"].values)
    mx = pd.DataFrame(np.vstack(futuros).T).max(axis=1, skipna=True).values
    df["lluvia_3h"] = np.where(np.isnan(mx), np.nan, (mx >= 0.1).astype(float))
    return df, informe


@st.cache_data(show_spinner=False)
def construir_diario(_d, clave):
    g = _d.groupby(["ESTACION", "fecha"])
    dias = g.agg(
        tmean=("temp", "mean"), tmax=("temp", "max"), tmin=("temp", "min"), nt=("temp", "count"),
        hum=("hum", "mean"), prec=("prec", "sum"), nprec=("prec", "count"),
        vmean=("vel", "mean"), vmax=("vel", "max"), nv=("vel", "count"),
    ).reset_index()
    # Un día solo cuenta si tiene al menos 20 horas válidas de esa variable
    dias.loc[dias.nprec < 20, "prec"] = np.nan
    dias.loc[dias.nt < 20, ["tmean", "tmax", "tmin"]] = np.nan
    dias.loc[dias.nv < 20, ["vmean", "vmax"]] = np.nan
    dias["anio"] = dias.fecha.dt.year
    dias["mes"] = dias.fecha.dt.month
    return dias


def interpreta(r):
    a = abs(r)
    fuerza = ("muy débil" if a < .2 else "débil" if a < .4 else "moderada" if a < .6
              else "fuerte" if a < .8 else "muy fuerte")
    return f"{fuerza} y {'positiva' if r > 0 else 'negativa'}"


def lluvia_mensual_climatologica(dias):
    """mm/mes promedio por mes calendario (promedia años y estaciones)."""
    x = dias.dropna(subset=["prec"])
    m = x.groupby(["ESTACION", "anio", "mes"]).prec.mean().reset_index()
    m = m.groupby("mes").prec.mean()
    return (m * DIAS_MES[m.index - 1]).reindex(range(1, 13))


# ----------------------------------------------------------------------------
# Barra lateral: datos y filtros
# ----------------------------------------------------------------------------
st.sidebar.title("Filtros")

if CSV_POR_DEFECTO.exists():
    fuente, marca = str(CSV_POR_DEFECTO), CSV_POR_DEFECTO.stat().st_mtime
    st.sidebar.caption(f"Datos: {CSV_POR_DEFECTO.name}")
else:
    up = st.sidebar.file_uploader("Sube el CSV", type="csv")
    if up is None:
        st.info("Coloca `Dataset Limpio2.csv` junto a `app.py` o súbelo en la barra lateral.")
        st.stop()
    fuente, marca = up.getvalue(), up.name

limpiar = st.sidebar.toggle(
    "Excluir valores imposibles (errores de sensor)", value=True,
    help="Pone en vacío los valores fuera de límites físicos razonables "
         "(p. ej. 82 °C, 749 mm en una hora).")
df, informe = cargar_datos(fuente, marca, limpiar)

estaciones = sorted(df.ESTACION.unique())
sel_est = st.sidebar.multiselect("Estaciones", estaciones, default=estaciones)
if not sel_est:
    st.warning("Selecciona al menos una estación.")
    st.stop()

fmin, fmax = df.fecha.min().date(), df.fecha.max().date()
rango = st.sidebar.date_input("Rango de fechas", (fmin, fmax), min_value=fmin, max_value=fmax)
f0, f1 = (rango[0], rango[1]) if isinstance(rango, (tuple, list)) and len(rango) == 2 else (fmin, fmax)

with st.sidebar.expander("Calidad de datos"):
    st.caption(f"{len(df):,} registros horarios · {df.ESTACION.nunique()} estaciones")
    for k, v in informe.items():
        st.write(f"- {k}: **{v if isinstance(v, str) else format(v, ',')}**")
    st.caption("Estos valores se " + ("excluyen" if limpiar else "NO se excluyen (filtro apagado)") + ".")
    st.caption("La columna de dirección dice «m/s» pero son grados (0–360°).")

d = df[df.ESTACION.isin(sel_est) & (df.fecha >= pd.Timestamp(f0)) & (df.fecha <= pd.Timestamp(f1))]
if d.empty:
    st.warning("No hay datos para ese filtro.")
    st.stop()

clave = (tuple(sorted(sel_est)), str(f0), str(f1), limpiar, marca if isinstance(marca, str) else float(marca))
dias = construir_diario(d, clave)
clima_mm = lluvia_mensual_climatologica(dias)
lluvia_anual = float(clima_mm.dropna().mean() * 12) if clima_mm.notna().any() else np.nan

# ----------------------------------------------------------------------------
# Encabezado y KPIs
# ----------------------------------------------------------------------------
st.title("Dashboard climático de las estaciones")

k = st.columns(5)
k[0].metric("Temp. promedio", f"{d.temp.mean():.1f} °C")
k[1].metric("Temp. máx. / mín.", f"{d.temp.max():.1f} / {d.temp.min():.1f} °C")
k[2].metric("Humedad media", f"{d.hum.mean():.0f} %")
k[3].metric("Lluvia anual estimada", f"{lluvia_anual:,.0f} mm" if not np.isnan(lluvia_anual) else "—")
k[4].metric("Viento medio", f"{d.vel.mean():.2f} m/s")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Clima", "Rosa de los vientos", "Relaciones", "Tendencias", "Días extremos", "Métricas"])

# ----------------------------------------------------------------------------
# 1. Clima
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("1. ¿Cómo es el clima de este lugar?")
    gran = st.radio("Temperatura por", ["Día", "Mes", "Año"], horizontal=True, index=1)
    col = {"Día": "fecha", "Mes": "mes_ts", "Año": "anio"}[gran]
    g = d.groupby(["ESTACION", col]).temp.agg(Media="mean", Máxima="max", Mínima="min").reset_index()
    serie = g.groupby(col)[["Media", "Máxima", "Mínima"]].mean().reset_index()
    fig = go.Figure()
    for nombre, color in (("Máxima", C_MAX), ("Media", C_MED), ("Mínima", C_MIN)):
        fig.add_scatter(x=serie[col], y=serie[nombre], name=nombre, mode="lines+markers" if gran != "Día" else "lines",
                        line=dict(color=color, width=2))
    fig.update_layout(height=380, yaxis_title="°C", xaxis_title=gran, hovermode="x unified",
                      margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.1))
    if gran == "Año":
        fig.update_xaxes(dtick=1)
    st.plotly_chart(fig)
    cap = "Promedio de todos los años y estaciones seleccionadas."
    if gran == "Año":
        mpa = d.groupby("anio").mes.nunique()
        cap += " Meses con datos por año: " + ", ".join(f"{a}: {n}" for a, n in mpa.items()) + \
               " (los años incompletos no son comparables entre sí)."
    st.caption(cap)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**¿Mes más lluvioso y más seco?**")
        if clima_mm.notna().any():
            mm = clima_mm.dropna()
            mx_m, mn_m = mm.idxmax(), mm.idxmin()
            a, b = st.columns(2)
            a.metric("Más lluvioso", MESES[mx_m - 1], f"{mm.max():.0f} mm/mes", delta_color="off")
            b.metric("Más seco", MESES[mn_m - 1], f"{mm.min():.0f} mm/mes", delta_color="off")
            colores = [("#1f5fa8" if m == mx_m else "#e07a5f" if m == mn_m else "#9ec5e8") for m in mm.index]
            f2 = go.Figure(go.Bar(x=[MESES[m - 1] for m in mm.index], y=mm.values, marker_color=colores,
                                  hovertemplate="%{x}: %{y:.0f} mm<extra></extra>"))
            f2.update_layout(height=320, yaxis_title="mm/mes (promedio)", margin=dict(t=10, b=10))
            st.plotly_chart(f2)
    with c2:
        st.markdown("**¿A qué hora hace más calor y más frío?**")
        h = d.groupby(["ESTACION", "hora"]).temp.mean().groupby("hora").mean()
        a, b = st.columns(2)
        a.metric("Más calor", f"{h.idxmax():02d}:00", f"{h.max():.1f} °C", delta_color="off")
        b.metric("Más frío", f"{h.idxmin():02d}:00", f"{h.min():.1f} °C", delta_color="off")
        f3 = go.Figure(go.Scatter(x=h.index, y=h.values, mode="lines+markers", line=dict(color=C_MAX),
                                  hovertemplate="%{x}:00 → %{y:.1f} °C<extra></extra>"))
        f3.update_layout(height=320, xaxis_title="Hora del día", yaxis_title="°C", margin=dict(t=10, b=10))
        f3.update_xaxes(dtick=2)
        st.plotly_chart(f3)

    piv = d.groupby(["hora", "mes"]).temp.mean().unstack("mes")
    piv.columns = [MESES[m - 1] for m in piv.columns]
    f4 = px.imshow(piv, aspect="auto", color_continuous_scale="RdYlBu_r",
                   labels=dict(x="Mes", y="Hora", color="°C"), origin="lower")
    f4.update_layout(height=380, title="Temperatura media por hora y mes", margin=dict(t=40, b=10))
    st.plotly_chart(f4)

# ----------------------------------------------------------------------------
# 2. Rosa de los vientos
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("2. Rosa de los vientos")
    w = d[["dir", "vel", "hora"]].dropna()
    calma = w[w.vel < 0.5]
    w = w[w.vel >= 0.5].copy()
    w["sec"] = (((w["dir"] + 11.25) % 360) // 22.5).astype(int)
    edges = [0.5, 1, 2, 3, 5, 8, 10, np.inf]
    labels = ["0.5–1", "1–2", "2–3", "3–5", "5–8", "8–10", "> 10"]
    w["bin"] = pd.cut(w.vel, edges, labels=labels, right=False)
    tab = pd.crosstab(w.sec, w.bin).reindex(index=range(16), columns=labels, fill_value=0)
    pct = tab / len(w) * 100
    dom = int(tab.sum(axis=1).idxmax())
    dom_pct = tab.sum(axis=1).max() / len(w) * 100

    umb_v = st.slider("Umbral de vendaval (m/s)", 5.0, 20.0, 10.0, 0.5)
    n_vend = int((d.vel > umb_v).sum())
    dias_vend = int((dias.vmax > umb_v).sum())

    m = st.columns(5)
    m[0].metric("El viento viene más del", f"{SECT[dom]} ({dom * 22.5:.0f}°)", f"{dom_pct:.1f} % de las horas", delta_color="off")
    m[1].metric("Velocidad media", f"{d.vel.mean():.2f} m/s")
    m[2].metric("Velocidad máx. registrada", f"{d.vel.max():.1f} m/s")
    m[3].metric(f"Horas con vendaval (> {umb_v:g} m/s)", f"{n_vend:,}", f"{n_vend / max(d.vel.notna().sum(), 1) * 100:.3f} % del tiempo", delta_color="off")
    m[4].metric("Calma (< 0.5 m/s)", f"{len(calma) / (len(calma) + len(w)) * 100:.1f} %")

    c1, c2 = st.columns([3, 2])
    with c1:
        colores = px.colors.sample_colorscale("Viridis", [i / (len(labels) - 1) for i in range(len(labels))])
        fr = go.Figure()
        for lab, colr in zip(labels, colores):
            fr.add_trace(go.Barpolar(r=pct[lab].values, theta=[i * 22.5 for i in range(16)], name=f"{lab} m/s",
                                     marker_color=colr, customdata=np.array(SECT),
                                     hovertemplate="%{customdata}: %{r:.2f} %<extra>" + lab + " m/s</extra>"))
        fr.update_layout(
            height=520, margin=dict(t=30, b=10), legend_title="Velocidad",
            polar=dict(angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                                        tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                        ticktext=["N", "NE", "E", "SE", "S", "SO", "O", "NO"]),
                       radialaxis=dict(ticksuffix="%", angle=45)))
        st.plotly_chart(fr)
        st.caption("Una rosa de vientos representa la frecuencia de dirección y velocidad del viento.")
    with c2:
        porh = d.assign(vend=(d.vel > umb_v)).groupby("hora").agg(vel=("vel", "mean"), vend=("vend", "sum"))
        f5 = make_subplots(specs=[[{"secondary_y": True}]])
        f5.add_bar(x=porh.index, y=porh.vend, secondary_y=True, showlegend=False)
        f5.add_scatter(x=porh.index, y=porh.vel, name="Velocidad media", line=dict(color=C_VIENTO), secondary_y=False)
        f5.update_layout(height=300, title="Viento por hora del día", margin=dict(t=40, b=10),
                         legend=dict(orientation="h", y=-0.25))
        f5.update_yaxes(title_text="m/s", secondary_y=False)
        f5.update_yaxes(title_text="horas", secondary_y=True)
        st.plotly_chart(f5)

# ----------------------------------------------------------------------------
# 3. Relaciones
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("3. Relación entre variables")
    def dispersion(xcol, ycol, xt, yt, color):
        z = d[[xcol, ycol]].dropna()
        if len(z) < 30:
            return None, np.nan, np.nan
        r = z[xcol].corr(z[ycol])
        pend, inter = np.polyfit(z[xcol], z[ycol], 1)
        s = z.sample(min(len(z), 6000), random_state=1)
        fg = go.Figure(go.Scattergl(x=s[xcol], y=s[ycol], mode="markers", marker=dict(size=4, opacity=.25, color=color),
                                    name="Horas"))
        xs = np.array([z[xcol].min(), z[xcol].max()])
        fg.add_scatter(x=xs, y=pend * xs + inter, mode="lines", line=dict(color="#e63946", width=3), name="Tendencia")
        fg.update_layout(height=380, xaxis_title=xt, yaxis_title=yt, margin=dict(t=10, b=10), showlegend=False)
        return fg, r, pend

    s1, s2 = st.tabs(["Humedad ↔ Temperatura", "Presión ↔ Lluvia"])
    with s1:
        fg, r, p = dispersion("hum", "temp", "Humedad (%)", "Temperatura (°C)", C_MIN)
        if fg is None:
            st.warning("Hay muy pocos datos en este filtro para dibujar la relación.")
        else:
            st.plotly_chart(fg)
            st.info(f"Correlación r = **{r:.2f}** ({interpreta(r)}). Por cada +10 % de humedad, la temperatura cambia "
                    f"**{p * 10:+.2f} °C** en promedio. Ojo: ambas siguen el ciclo diario (de noche hay más humedad y menos calor), "
                    "así que no prueba causalidad.")
    

with tab4:
    st.subheader("4. Tendencias y climograma")
    var = st.radio("Variable de la tendencia anual", ["Temperatura (°C)", "Precipitación (mm/mes)"],
                   horizontal=True)
    vcol, agg, unidad = {"Temperatura (°C)": ("temp", "mean", "°C"),
                         "Precipitación (mm/mes)": ("prec", "sum", "mm/año")}[var]

    dm = dias.copy()
    dm["mes_ts"] = dm.fecha.dt.to_period("M").dt.to_timestamp()
    mm_ = dm.groupby(["ESTACION", "mes_ts"]).agg(
        temp=("tmean", "mean"), n_t=("tmean", "count"), prec_d=("prec", "mean"), n_p=("prec", "count")).reset_index()
    mm_["prec"] = mm_.prec_d * mm_.mes_ts.dt.days_in_month
    mm_.loc[mm_.n_t < 15, "temp"] = np.nan
    mm_.loc[mm_.n_p < 15, "prec"] = np.nan
    mm_["mes"], mm_["anio"] = mm_.mes_ts.dt.month, mm_.mes_ts.dt.year
    sel = mm_[["ESTACION", "mes_ts", "mes", "anio", vcol]].dropna().copy()

    ya = pd.Series(dtype=float)
    if not sel.empty:
        anual = sel.groupby(["ESTACION", "anio"]).agg(v=(vcol, agg), n=(vcol, "count")).reset_index()
        ya = anual[anual.n == 12].groupby("anio").v.mean()

    c1, c2 = st.columns(2)
    st.caption("Representa los años de muestra que tienen nuestra base de datos(3.5 años).")
    with c1:
        if len(ya) >= 2:
            pend_a, inter_a = np.polyfit(ya.index.astype(float), ya.values, 1)
            fy = go.Figure()
            fy.add_scatter(x=ya.index, y=ya.values, mode="lines+markers", name="Anual", line=dict(color=C_MED, width=3))
            fy.add_scatter(x=ya.index, y=pend_a * ya.index + inter_a, mode="lines", name="Tendencia",
                           line=dict(color=C_TREND, dash="dash"))
            fy.update_layout(height=380, title="Valor anual (solo años completos) y línea de tendencia",
                             yaxis_title=unidad, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2))
            fy.update_xaxes(dtick=1)
            st.plotly_chart(fy)
        else:
            st.info("Hay menos de 2 años completos en el filtro: no se puede dibujar la tendencia anual.")
    with c2:
        t_mes = dias.groupby("mes").tmean.mean().reindex(range(1, 13))
        fcl = make_subplots(specs=[[{"secondary_y": True}]])
        fcl.add_bar(x=MESES, y=clima_mm.values, name="Lluvia (mm/mes)", marker_color=C_LLUVIA, secondary_y=False)
        fcl.add_scatter(x=MESES, y=t_mes.values, name="Temperatura (°C)", line=dict(color=C_MAX, width=3), secondary_y=True)
        fcl.update_layout(height=380, title="Climograma: lluvia y temperatura por mes",
                          margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2))
        fcl.update_yaxes(title_text="mm/mes", secondary_y=False)
        fcl.update_yaxes(title_text="°C", secondary_y=True)
        st.plotly_chart(fcl)
        

# ----------------------------------------------------------------------------
# 5. Días extremos
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("5. Días extremos (KPIs climáticos)")
    a, b, c, e = st.columns(4)
    u_h0 = a.number_input("Helada fuerte: T mín. diaria <", value=0.0, step=0.5, format="%.1f")
    u_h1 = b.number_input("Helada / frío: T mín. diaria <", value=5.0, step=0.5, format="%.1f")
    u_ll = c.number_input("Lluvia fuerte: > (mm/día)", value=20.0, step=1.0, format="%.1f")
    u_vf = e.number_input("Viento fuerte: máx. diaria > (m/s)", value=10.0, step=0.5, format="%.1f")

    ev = dias.assign(h0=dias.tmin < u_h0, h1=dias.tmin < u_h1, ll=dias.prec > u_ll, vf=dias.vmax > u_vf)
    por_est = ev.groupby("ESTACION").agg(
        dias_temp=("tmin", "count"), helada_fuerte=("h0", "sum"), helada_frio=("h1", "sum"),
        dias_lluvia=("prec", "count"), lluvia_fuerte=("ll", "sum"),
        dias_viento=("vmax", "count"), viento_fuerte=("vf", "sum"))
    prom = por_est.mean()
    etiqueta = "promedio por estación" if len(sel_est) > 1 else "en el periodo"

    kk = st.columns(4)
    kk[0].metric(f"Días con T mín. < {u_h0:g} °C", f"{prom.helada_fuerte:.0f}", etiqueta, delta_color="off")
    kk[1].metric(f"Días con T mín. < {u_h1:g} °C", f"{prom.helada_frio:.0f}", etiqueta, delta_color="off")
    kk[2].metric(f"Días con lluvia > {u_ll:g} mm", f"{prom.lluvia_fuerte:.0f}", etiqueta, delta_color="off")
    kk[3].metric(f"Días con viento > {u_vf:g} m/s", f"{prom.viento_fuerte:.0f}", etiqueta, delta_color="off")

    opciones = {f"T mín. < {u_h1:g} °C": "h1", f"T mín. < {u_h0:g} °C": "h0",
                f"Lluvia > {u_ll:g} mm": "ll", f"Viento > {u_vf:g} m/s": "vf"}
    tipo = st.radio("Ver evento", list(opciones), horizontal=True)
    cc = opciones[tipo]
    n_est = max(len(sel_est), 1)
    c1, c2 = st.columns(2)
    por_anio = ev.groupby("anio")[cc].sum() / n_est
    fa = go.Figure(go.Bar(x=por_anio.index.astype(str), y=por_anio.values, marker_color=C_MAX,
                          hovertemplate="%{x}: %{y:.1f} días<extra></extra>"))
    fa.update_layout(height=300, title="Por año (días acumulados, promedio por estación)", margin=dict(t=40, b=10))
    c1.plotly_chart(fa)
    por_mes = ev.groupby("mes")[cc].sum().reindex(range(1, 13), fill_value=0) / n_est
    fm = go.Figure(go.Bar(x=MESES, y=por_mes.values, marker_color=C_LLUVIA,
                          hovertemplate="%{x}: %{y:.1f} días<extra></extra>"))
    fm.update_layout(height=300, title="Por mes calendario (días acumulados, promedio por estación)", margin=dict(t=40, b=10))
    c2.plotly_chart(fm)
    st.caption("Los años 2022 y 2026 solo tienen parte de los meses, por eso suman menos días.")
with tab6:
    st.subheader("Métricas de rendimiento")
    st.caption("Comparación entre la ejecución")

    t_seq = 1.7279
    t_par = 2.5585
    speedup = 0.6752
    eficiencia = 16.88

    # Tarjetas principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tiempo secuencial", f"{t_seq:.4f} s")
    col2.metric(
        "Tiempo paralelo",
        f"{t_par:.4f} s",
        delta=f"{t_par - t_seq:+.4f} s",
        delta_color="inverse",
    )
    col3.metric(
        "Aceleración (Speedup)",
        f"{speedup:.4f}",
        delta=f"{speedup - 1:+.4f} vs. ideal 1.0",
        delta_color="inverse",
    )
    col4.metric("Eficiencia", f"{eficiencia:.2f} %")

    st.divider()

    # Barra de eficiencia
    st.markdown("**Nivel de eficiencia**")
    st.progress(min(eficiencia / 100, 1.0))

    st.divider()

    # Interpretación
    if speedup < 1:
        st.info(
            "La versión paralela fue más lenta que la secuencial. "
            "Esto suele deberse al costo de crear procesos o hilos"
            "De La parte secuencial que no se puede paralelizar,"
            "o a la sobrecarga de comunicación entre procesos."
        )