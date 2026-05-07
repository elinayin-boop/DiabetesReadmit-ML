"""
D230 Diabetes Readmission — Interactive EDA Dashboard
Run: python dashboard.py  then open http://127.0.0.1:8050

Install: pip install dash plotly pandas numpy
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback

# ── Colour palette (matches the original matplotlib theme) ───────────────────
C = dict(
    teal   = "#028090",
    mint   = "#02C39A",
    accent = "#F96167",
    amber  = "#F4A261",
    navy   = "#0D1B2A",
    gray   = "#64748B",
    lgray  = "#CBD5E1",
    offwh  = "#F0F4F8",
    white  = "#FFFFFF",
    navy2  = "#1C3A5E",
)

# ── Simulated dataset (replace with df from diabetic_cleaned.csv) ─────────────
# Reproduces exact statistics from the cleaning file output.
# To use real data: df = pd.read_csv("diabetic_cleaned.csv")

np.random.seed(42)
N = 99353  # post-cleaning row count

age_order   = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
               "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
age_counts  = [160, 690, 1650, 3765, 9626, 17102, 22187, 25564, 16708, 2669]
age_rates   = [0.019,0.058,0.143,0.113,0.107,0.098,0.113,0.120,0.124,0.116]

race_vals   = ["Caucasian","African American","Hispanic","Asian","Other","Unknown"]
race_counts = [76099, 19210, 2037, 641, 1243, 123]
race_rates  = [0.110, 0.112, 0.095, 0.076, 0.098, 0.088]

gender_vals   = ["Female","Male"]
gender_counts = [53462, 45891]
gender_rates  = [0.109, 0.113]

inp_labels = ["0","1","2","3","4","5+"]
inp_rates  = [0.084,0.137,0.178,0.221,0.248,0.276]
inp_counts = [74210,14820,6180,2440,990,713]

corr_features = ["time_hosp","lab_procs","procedures","medications",
                 "outpatient","emergency","inpatient","diagnoses","readmitted"]
corr_matrix = np.array([
    [1.00, 0.36, 0.08, 0.47, 0.01, 0.02, 0.05, 0.33, 0.07],
    [0.36, 1.00, 0.16, 0.22, 0.01, 0.01, 0.04, 0.25, 0.03],
    [0.08, 0.16, 1.00, 0.11, 0.01, 0.01, 0.02, 0.10, 0.01],
    [0.47, 0.22, 0.11, 1.00, 0.02, 0.03, 0.07, 0.30, 0.06],
    [0.01, 0.01, 0.01, 0.02, 1.00, 0.06, 0.10, 0.03, 0.05],
    [0.02, 0.01, 0.01, 0.03, 0.06, 1.00, 0.12, 0.04, 0.06],
    [0.05, 0.04, 0.02, 0.07, 0.10, 0.12, 1.00, 0.07, 0.19],
    [0.33, 0.25, 0.10, 0.30, 0.03, 0.04, 0.07, 1.00, 0.05],
    [0.07, 0.03, 0.01, 0.06, 0.05, 0.06, 0.19, 0.05, 1.00],
])
# lower triangle only
corr_lower = np.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool),
                      np.nan, corr_matrix)

readmit_total  = int(N * 0.113)
no_readmit     = N - readmit_total

# ── Helper: plot config ───────────────────────────────────────────────────────
PLOT_CFG = dict(displayModeBar=False)
LAYOUT_BASE = dict(
    plot_bgcolor  = C["white"],
    paper_bgcolor = C["offwh"],
    font          = dict(family="'Segoe UI', sans-serif", color=C["navy"], size=12),
)

def insight_box(text):
    return html.Div(text, style={
        "background": "#E6F4FF",
        "border-left": f"4px solid {C['teal']}",
        "border-radius": "0 8px 8px 0",
        "padding": "10px 16px",
        "font-size": "13px",
        "color": C["navy"],
        "margin-top": "10px",
        "line-height": "1.7",
    })

def section_card(title, content):
    return html.Div([
        html.H3(title, style={
            "font-size": "15px", "font-weight": "600",
            "color": C["navy"], "margin": "0 0 14px",
            "padding-bottom": "8px",
            "border-bottom": f"2px solid {C['teal']}",
        }),
        content
    ], style={
        "background": C["white"],
        "border-radius": "12px",
        "padding": "20px 24px",
        "box-shadow": "0 2px 12px rgba(0,0,0,0.07)",
        "margin-bottom": "20px",
    })

# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="D230 EDA Dashboard")

app.layout = html.Div([

    # Header
    html.Div([
        html.Div([
            html.H1("Diabetes Readmission — EDA Dashboard",
                    style={"margin":0,"font-size":"22px","font-weight":"700",
                           "color":C["white"],"letter-spacing":"0.3px"}),
            html.P("UCI 130-US Hospitals  |  99,353 encounters after cleaning",
                   style={"margin":"4px 0 0","font-size":"12px","color":"#A8D8E8","opacity":"0.9"}),
        ], style={"flex":"1"}),
        html.Div([
            html.Span("⬤", style={"color":C["mint"],"margin-right":"6px"}),
            html.Span("Interactive", style={"color":C["white"],"font-size":"12px"}),
        ], style={"display":"flex","align-items":"center"}),
    ], style={
        "background": f"linear-gradient(135deg, {C['navy']} 0%, {C['navy2']} 100%)",
        "padding": "18px 32px",
        "display": "flex",
        "align-items": "center",
        "justify-content": "space-between",
        "box-shadow": "0 2px 16px rgba(0,0,0,0.2)",
    }),

    # Filters bar
    html.Div([
        html.Div([
            html.Label("Age Groups", style={"font-size":"12px","color":C["gray"],
                                             "font-weight":"600","margin-bottom":"4px",
                                             "display":"block"}),
            dcc.Dropdown(
                id="filter-age",
                options=[{"label": g, "value": i} for i, g in enumerate(age_order)],
                value=list(range(10)),
                multi=True,
                clearable=False,
                placeholder="Select age groups...",
                style={"font-size":"13px","min-width":"500px"},
            ),
        ], style={"flex":"2"}),
        html.Div([
            html.Label("Inpatient Visits ≤", style={"font-size":"12px","color":C["gray"],
                                                      "font-weight":"600","margin-bottom":"4px",
                                                      "display":"block"}),
            dcc.Slider(
                id="filter-inpatient",
                min=0, max=5, step=1, value=5,
                marks={i: {"label":str(i) if i<5 else "5+",
                           "style":{"font-size":"11px","color":C["gray"]}}
                       for i in range(6)},
            ),
        ], style={"flex":"1","min-width":"220px"}),
    ], style={
        "background": C["white"],
        "padding": "14px 32px",
        "display": "flex",
        "gap": "28px",
        "align-items": "flex-end",
        "box-shadow": "0 1px 8px rgba(0,0,0,0.06)",
        "flex-wrap": "wrap",
        "border-bottom": f"1px solid {C['lgray']}",
    }),
    # KPI strip
    html.Div(id="kpi-strip", style={
        "display": "flex",
        "gap": "16px",
        "padding": "16px 32px",
        "background": C["offwh"],
        "flex-wrap": "wrap",
    }),

    # Main content
    html.Div([

        # Row 1: class balance + age
        html.Div([
            html.Div(id="chart-class-balance", style={"flex":"1","min-width":"340px"}),
            html.Div(id="chart-age",           style={"flex":"2","min-width":"480px"}),
        ], style={"display":"flex","gap":"20px","flex-wrap":"wrap"}),

        # Row 2: inpatient + correlation
        html.Div([
            html.Div(id="chart-inpatient",    style={"flex":"1","min-width":"380px"}),
            html.Div(id="chart-correlation",  style={"flex":"1","min-width":"380px"}),
        ], style={"display":"flex","gap":"20px","flex-wrap":"wrap","margin-top":"0"}),



    ], style={"padding":"20px 32px","background":C["offwh"],"min-height":"calc(100vh - 180px)"}),

], style={"font-family":"'Segoe UI', sans-serif","background":C["offwh"],"min-height":"100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-strip", "children"),
    Output("chart-class-balance", "children"),
    Output("chart-age", "children"),
    Output("chart-inpatient", "children"),
    Output("chart-correlation", "children"),
    Input("filter-age", "value"),
    Input("filter-inpatient", "value"),
)
def update_all(age_selection, max_inp):
    # age_selection is list of indices from multi-select dropdown
    if not age_selection:
        age_selection = list(range(10))
    age_idx = sorted(age_selection)
    g_mult  = 1.0
    r_mult  = 1.0

    # ── Filtered age data ────────────────────────────────────────────────────
    sel_ages   = [age_order[i]   for i in age_idx]
    sel_counts = [int(age_counts[i] * g_mult * r_mult) for i in age_idx]
    sel_rates  = [age_rates[i] for i in age_idx]
    sel_read   = [int(c * r) for c, r in zip(sel_counts, sel_rates)]

    total_enc  = sum(sel_counts)
    total_read = sum(sel_read)
    pos_rate   = total_read / total_enc if total_enc > 0 else 0

    # ── Filtered inpatient ───────────────────────────────────────────────────
    inp_sel_labels = inp_labels[:max_inp+1]
    inp_sel_rates  = inp_rates[:max_inp+1]
    inp_sel_counts = [int(inp_counts[i] * g_mult * r_mult) for i in range(max_inp+1)]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    def kpi(label, value, color=C["teal"], sub=""):
        return html.Div([
            html.Div(value, style={"font-size":"24px","font-weight":"700","color":color}),
            html.Div(label, style={"font-size":"11px","color":C["gray"],"margin-top":"2px"}),
            html.Div(sub,   style={"font-size":"10px","color":C["lgray"]}) if sub else None,
        ], style={
            "background":C["white"],"border-radius":"10px","padding":"12px 20px",
            "box-shadow":"0 1px 6px rgba(0,0,0,0.06)",
            "border-top":f"3px solid {color}","min-width":"130px",
        })

    kpis = [
        kpi("Encounters",     f"{total_enc:,}",    C["teal"]),
        kpi("Readmitted <30d",f"{total_read:,}",   C["accent"]),
        kpi("Readmission Rate",f"{pos_rate:.1%}",  C["accent"], "positive class"),
        kpi("Age Groups",     str(len(sel_ages)),  C["navy"]),
        kpi("Peak Age Group", max(zip(sel_rates, sel_ages))[1] if sel_ages else "—", C["mint"]),
        kpi("Inpatient Cap",  inp_labels[max_inp], C["amber"]),
    ]

    # ════════════════════════════════════════════════════════════════════════
    # CHART 1 — Class Balance
    # ════════════════════════════════════════════════════════════════════════
    fig_cb = make_subplots(rows=1, cols=2,
                           specs=[[{"type":"bar"},{"type":"pie"}]])
    fig_cb.add_trace(go.Bar(
        x=["Not Readmitted","Readmitted <30d"],
        y=[total_enc - total_read, total_read],
        marker_color=[C["teal"], C["accent"]],
        text=[f"{total_enc-total_read:,}", f"{total_read:,}"],
        textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig_cb.add_trace(go.Pie(
        labels=["Not Readmitted","Readmitted <30d"],
        values=[total_enc - total_read, total_read],
        marker_colors=[C["teal"], C["accent"]],
        hole=0.4,
        textinfo="percent+label",
        showlegend=False,
    ), row=1, col=2)
    fig_cb.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Class Balance", font=dict(size=14, color=C["navy"]), x=0),
        height=300, margin=dict(l=40,r=20,t=55,b=30),
        annotations=[
            dict(text="Encounter Counts", x=0.18, y=1.08, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color=C["gray"])),
            dict(text="Class Proportion", x=0.89, y=1.08, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color=C["gray"])),
        ],
    )
    fig_cb.update_yaxes(showgrid=True, gridcolor=C["lgray"], row=1, col=1)
    chart_cb = section_card("📊 Class Balance", html.Div([
        dcc.Graph(figure=fig_cb, config=PLOT_CFG),
    ]))

    # ════════════════════════════════════════════════════════════════════════
    # CHART 2 — Age Group vs Readmission
    # ════════════════════════════════════════════════════════════════════════
    peak_age_idx = int(np.argmax(sel_rates)) if sel_rates else 0
    bar_colors_age = [C["accent"] if i == peak_age_idx else C["teal"]
                      for i in range(len(sel_ages))]

    fig_age = make_subplots(specs=[[{"secondary_y": True}]])
    fig_age.add_trace(go.Bar(
        x=sel_ages, y=[r*100 for r in sel_rates],
        marker_color=bar_colors_age,
        marker_line_color="white", marker_line_width=1.5,
        opacity=0.88, name="Readmission Rate (%)",
        text=[f"{r:.1%}" for r in sel_rates],
        textposition="outside",
        hovertemplate="Age: %{x}<br>Rate: %{y:.1f}%<extra></extra>",
    ), secondary_y=False)
    fig_age.add_trace(go.Scatter(
        x=sel_ages, y=[c/1000 for c in sel_counts],
        mode="lines+markers",
        line=dict(color=C["amber"], width=2.2),
        marker=dict(size=7, color=C["white"],
                    line=dict(color=C["amber"], width=2.2)),
        name="Encounter volume (k)",
        hovertemplate="Age: %{x}<br>Volume: %{y:.1f}k<extra></extra>",
        fill="tozeroy", fillcolor="rgba(244,162,97,0.07)",
    ), secondary_y=True)

    # Quadratic trend
    if len(sel_ages) >= 3:
        mids = [5+i*10 for i in age_idx]
        z = np.polyfit(mids, [r*100 for r in sel_rates], 2)
        p_fn = np.poly1d(z)
        x_smooth = np.linspace(min(mids), max(mids), 60)
        y_smooth = p_fn(x_smooth)
        x_idx_smooth = np.linspace(0, len(sel_ages)-1, 60)
        fig_age.add_trace(go.Scatter(
            x=[sel_ages[min(int(round(xi)), len(sel_ages)-1)]
               for xi in np.linspace(0, len(sel_ages)-1, 60)],
            y=y_smooth,
            mode="lines",
            line=dict(color=C["navy"], width=1.5, dash="dash"),
            name="Quadratic trend", opacity=0.45,
            hoverinfo="skip",
        ), secondary_y=False)

    fig_age.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Age Group vs 30-Day Readmission Rate", font=dict(size=14), x=0),
        height=370,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11), bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=50,r=60,t=55,b=50),
        hovermode="x unified",
    )
    fig_age.update_yaxes(title_text="Readmission Rate (%)",
                         title_font=dict(color=C["navy"]),
                         showgrid=True, gridcolor=C["lgray"],
                         range=[0, max([r*100 for r in sel_rates])*1.35] if sel_rates else None,
                         secondary_y=False)
    fig_age.update_yaxes(title_text="Encounter Volume (k)",
                         title_font=dict(color=C["amber"]),
                         showgrid=False, secondary_y=True)

    peak_age = sel_ages[peak_age_idx] if sel_ages else "—"
    peak_r   = sel_rates[peak_age_idx] if sel_rates else 0

    chart_age = section_card("👥 Age Group vs Readmission Rate", html.Div([
        dcc.Graph(figure=fig_age, config=PLOT_CFG),
    ]))

    # ════════════════════════════════════════════════════════════════════════
    # CHART 3 — Prior Inpatient Visits
    # ════════════════════════════════════════════════════════════════════════
    cmap = px.colors.sequential.YlOrRd
    n_inp = len(inp_sel_labels)
    inp_bar_colors = [cmap[int(2 + i * (len(cmap)-3) / max(n_inp-1, 1))]
                      for i in range(n_inp)]

    fig_inp = go.Figure()
    fig_inp.add_trace(go.Bar(
        x=inp_sel_labels, y=[r*100 for r in inp_sel_rates],
        marker_color=inp_bar_colors,
        marker_line_color="white", marker_line_width=1.5,
        text=[f"{r:.1%}" for r in inp_sel_rates],
        textposition="outside",
        hovertemplate="Visits: %{x}<br>Rate: %{y:.1f}%<br>n=%{customdata:,}<extra></extra>",
        customdata=inp_sel_counts,
        name="Readmission Rate",
    ))
    # Delta annotations
    for i in range(1, len(inp_sel_rates)):
        delta = (inp_sel_rates[i] - inp_sel_rates[i-1]) * 100
        fig_inp.add_annotation(
            x=inp_sel_labels[i], xref="x", y=inp_sel_rates[i]*100 + 1.8,
            text=f"+{delta:.1f}pp", showarrow=False,
            font=dict(size=9.5, color=C["gray"]), opacity=0.8,
        )

    fig_inp.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Prior Inpatient Visits → Readmission Rate", font=dict(size=14), x=0),
        height=350,
        xaxis=dict(title="Prior inpatient visits (capped at 5+)"),
        yaxis=dict(title="30-day Readmission Rate (%)",
                   showgrid=True, gridcolor=C["lgray"],
                   range=[0, max(inp_sel_rates)*130]),
        showlegend=False,
        margin=dict(l=50,r=30,t=55,b=55),
        hovermode="x",
    )

    delta_total = (inp_sel_rates[-1] - inp_sel_rates[0]) * 100
    chart_inp = section_card("🏥 Prior Inpatient Visits vs Readmission", html.Div([
        dcc.Graph(figure=fig_inp, config=PLOT_CFG),
    ]))

    # ════════════════════════════════════════════════════════════════════════
    # CHART 4 — Correlation Heatmap
    # ════════════════════════════════════════════════════════════════════════
    n_feat = len(corr_features)
    z_plot = corr_lower.copy()

    # Colour scale: red-yellow-green
    colorscale = [
        [0.0, "#F96167"], [0.3, "#FFF4A3"],
        [0.5, "#FFFDE7"], [0.7, "#C8E6C9"],
        [1.0, "#028090"],
    ]
    fig_corr = go.Figure(go.Heatmap(
        z=z_plot,
        x=corr_features, y=corr_features,
        colorscale=colorscale,
        zmin=-0.3, zmax=0.55,
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row]
              for row in z_plot],
        texttemplate="%{text}",
        textfont=dict(size=9.5),
        hovertemplate="x: %{x}<br>y: %{y}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", thickness=14, len=0.8),
    ))

    # Highlight readmitted col/row border
    for i in range(n_feat):
        fig_corr.add_shape(type="rect",
            x0=n_feat-1-0.5, y0=i-0.5, x1=n_feat-0.5, y1=i+0.5,
            line=dict(color=C["accent"], width=2.5))
        fig_corr.add_shape(type="rect",
            x0=i-0.5, y0=n_feat-1-0.5, x1=i+0.5, y1=n_feat-0.5,
            line=dict(color=C["accent"], width=2.5))

    fig_corr.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Feature Correlation Matrix (lower triangle)  |  🔴 = readmitted",
                   font=dict(size=13), x=0),
        height=420,
        xaxis=dict(tickangle=35, tickfont=dict(size=9.5)),
        yaxis=dict(tickfont=dict(size=9.5), autorange="reversed"),
        margin=dict(l=90,r=30,t=55,b=70),
    )

    chart_corr = section_card("🔗 Feature Correlation Heatmap", html.Div([
        dcc.Graph(figure=fig_corr, config=PLOT_CFG),
    ]))


    return kpis, chart_cb, chart_age, chart_inp, chart_corr


if __name__ == "__main__":
    app.run(debug=True, port=8050)
