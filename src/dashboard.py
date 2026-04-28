"""
Plotly Dash Dashboard

Run:
    cd /path/to/airbnb-price-predictor
    python src/dashboard.py

Then open http://localhost:8050 in your browser.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, dcc, html
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to project root - run from project root)
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Load data & models at startup (no retraining)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df_pl = pl.read_parquet(DATA_DIR / "airbnb_featured.parquet")
df    = df_pl.to_pandas()
print(f"  Loaded {len(df):,} listings")

print("Loading models...")
ridge_data = pickle.load(open(MODELS_DIR / "ridge_model_ridgecv.pkl", "rb"))
rf_data    = pickle.load(open(MODELS_DIR / "rf_model.pkl",    "rb"))
xgb_data   = pickle.load(open(MODELS_DIR / "xgb_model.pkl",  "rb"))

# Support both old format (bare pipeline) and new format (dict with pipeline key)
ridge_pipe = ridge_data if not isinstance(ridge_data, dict) else ridge_data.get("pipeline", ridge_data)
rf_pipe    = rf_data["pipeline"]
xgb_pipe   = xgb_data["pipeline"]

# Ridge was saved as a bare pipeline — extract its expected feature columns
ridge_feature_cols = list(ridge_pipe.feature_names_in_)
print("  Models loaded.")

# Pre-compute feature importance data (no retraining needed)
def _clean_feature_name(name):
    """Make feature names readable for charts."""
    if name.startswith("pca_amenity_"):
        n = int(name.split("_")[-1])
        return f"Amenity Factor {n+1}"
    return name.replace("_", " ").title()

# SHAP mean |value| per feature (top 20)
_shap_vals  = np.abs(np.array(xgb_data["shap_values"]))
_shap_names = [_clean_feature_name(n) for n in xgb_data["shap_feature_names"]]
_shap_mean  = _shap_vals.mean(axis=0)
_shap_df    = pd.DataFrame({"feature": _shap_names, "importance": _shap_mean}) \
                .sort_values("importance", ascending=False).head(20) \
                .sort_values("importance")

# RF impurity-based feature importance (top 20)
_rf_imp    = rf_data["pipeline"].named_steps["rf"].feature_importances_
_rf_names  = [_clean_feature_name(n) for n in rf_data["feature_names"]]
_rf_df     = pd.DataFrame({"feature": _rf_names, "importance": _rf_imp}) \
               .sort_values("importance", ascending=False).head(20) \
               .sort_values("importance")

CITY_LABELS  = {"nyc": "New York City", "la": "Los Angeles", "chi": "Chicago"}
CITY_COLORS  = {"nyc": "#E07B54", "la": "#16A085", "chi": "#2980B9"}
PLOTLY_TPL   = "plotly_white"

# Pre-compute model metrics on a consistent test set
DROP_COLS = {"id", "host_id", "price_usd", "log_price"}
y_all = df["log_price"]
X_all = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
_, X_test_raw, _, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)

def get_metrics(pipe, X_test, y_test):
    try:
        y_pred = pipe.predict(X_test)
        return {
            "RMSE (log)": round(mean_squared_error(y_test, y_pred) ** 0.5, 4),
            "MAE (log)":  round(mean_absolute_error(y_test, y_pred), 4),
            "R²":         round(r2_score(y_test, y_pred), 4),
            "RMSE (USD)": round(mean_squared_error(np.exp(y_test), np.exp(y_pred)) ** 0.5, 2),
        }
    except Exception:
        return {"RMSE (log)": "-", "MAE (log)": "-", "R²": "-", "RMSE (USD)": "-"}

ridge_metrics = get_metrics(ridge_pipe, X_test_raw, y_test)
rf_metrics    = get_metrics(rf_pipe,    X_test_raw[rf_data["NUMERIC_COLS"] + rf_data["CATEGORICAL_COLS"]], y_test)
xgb_metrics   = get_metrics(xgb_pipe,  X_test_raw[xgb_data["NUMERIC_COLS"] + xgb_data["CATEGORICAL_COLS"]], y_test)

# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood options per city
# ─────────────────────────────────────────────────────────────────────────────
nbhd_by_city = {
    city: sorted(df[df["city"] == city]["neighbourhood_cleansed"].dropna().unique().tolist())
    for city in ["nyc", "la", "chi"]
}

# ─────────────────────────────────────────────────────────────────────────────
# App design
# ─────────────────────────────────────────────────────────────────────────────
FONTS = "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap"

C_BG     = "#F7F4EF"
C_CARD   = "#FFFFFF"
C_DARK   = "#1C1C1C"
C_ACCENT = "#7B5C9E"
C_TEAL   = "#00A699"
C_MUTED  = "#888888"
C_BORDER = "#E8E4DD"

FONT = "'Outfit', system-ui, sans-serif"

CARD = {
    "background": C_CARD,
    "borderRadius": "12px",
    "padding": "24px",
    "border": f"1px solid {C_BORDER}",
    "marginBottom": "16px",
}
LABEL = {
    "fontFamily": FONT,
    "fontWeight": "600",
    "fontSize": "11px",
    "letterSpacing": "0.08em",
    "textTransform": "uppercase",
    "color": C_MUTED,
    "marginBottom": "6px",
    "marginTop": "18px",
    "display": "block",
}
TAB_STYLE = {
    "fontFamily": FONT,
    "fontWeight": "500",
    "fontSize": "13px",
    "padding": "12px 24px",
    "color": C_MUTED,
    "border": "none",
    "borderBottom": "3px solid transparent",
    "background": "transparent",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color": C_DARK,
    "fontWeight": "700",
    "borderBottom": f"3px solid {C_ACCENT}",
}

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    title="Airbnb Price Predictor",
    suppress_callback_exceptions=True,
    external_stylesheets=[FONTS],
)

app.layout = html.Div(
    style={"fontFamily": FONT, "background": C_BG, "minHeight": "100vh"},
    children=[

        # Header
        html.Div(style={
            "background": C_DARK,
            "padding": "36px 48px 28px",
            "borderBottom": f"4px solid {C_ACCENT}",
        }, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-end"}, children=[
                html.Div([
                    html.Div("AIRBNB PRICE PREDICTOR", style={
                        "fontFamily": FONT,
                        "fontSize": "10px",
                        "letterSpacing": "0.25em",
                        "color": C_ACCENT,
                        "fontWeight": "700",
                        "marginBottom": "10px",
                    }),
                    html.H1("What's Your Listing Worth?", style={
                        "fontFamily": FONT,
                        "fontSize": "38px",
                        "color": "#F5F5F0",
                        "margin": 0,
                        "fontWeight": "800",
                        "lineHeight": "1.1",
                    }),
                ]),
                html.Div(style={"textAlign": "right"}, children=[
                    html.Div("New York City · Los Angeles · Chicago", style={
                        "color": "#aaa", "fontSize": "13px", "fontFamily": FONT,
                    }),
                    html.Div("CIS 2450 · Big Data Analytics", style={
                        "color": "#555", "fontSize": "11px", "marginTop": "5px",
                        "fontFamily": FONT, "letterSpacing": "0.05em",
                    }),
                ]),
            ]),
        ]),

        # Tab bar
        html.Div(style={
            "background": C_CARD,
            "borderBottom": f"1px solid {C_BORDER}",
            "paddingLeft": "32px",
        }, children=[
            dcc.Tabs(id="tabs", value="tab-predict", style={"border": "none"}, children=[
                dcc.Tab(label="Price Predictor",  value="tab-predict",  style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="EDA Explorer",     value="tab-eda",      style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Model Comparison", value="tab-models",   style=TAB_STYLE, selected_style=TAB_SELECTED),
            ]),
        ]),

        html.Div(id="tab-content", style={"padding": "28px 40px", "maxWidth": "1400px", "margin": "0 auto"}),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# Tab routing
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-predict":
        return render_predictor()
    elif tab == "tab-eda":
        return render_eda()
    else:
        return render_models()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Price Predictor
# ─────────────────────────────────────────────────────────────────────────────
def render_predictor():
    return html.Div([
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "360px 1fr",
            "gap": "24px",
            "alignItems": "start",
        }, children=[

            # Input form
            html.Div(style=CARD, children=[
                html.Div("Listing Details", style={
                    "fontFamily": FONT, "fontSize": "20px", "fontWeight": "700",
                    "color": C_DARK, "marginBottom": "4px", "marginTop": 0,
                }),
                html.Div("Configure your listing to get an instant price estimate.",
                         style={"fontSize": "13px", "color": C_MUTED, "marginBottom": "8px"}),

                html.Hr(style={"border": "none", "borderTop": f"1px solid {C_BORDER}", "margin": "16px 0"}),

                html.Label("City", style=LABEL),
                dcc.Dropdown(id="inp-city",
                    options=[{"label": v, "value": k} for k, v in CITY_LABELS.items()],
                    value="nyc", clearable=False,
                    style={"fontFamily": FONT, "fontSize": "13px"}),

                html.Label("Neighbourhood", style=LABEL),
                dcc.Dropdown(id="inp-neighbourhood", value=None, clearable=True,
                             placeholder="Any neighbourhood",
                             style={"fontFamily": FONT, "fontSize": "13px"}),

                html.Label("Room Type", style=LABEL),
                dcc.Dropdown(id="inp-room-type",
                    options=[{"label": v, "value": v} for v in [
                        "Entire home/apt", "Private room", "Shared room", "Hotel room"]],
                    value="Entire home/apt", clearable=False,
                    style={"fontFamily": FONT, "fontSize": "13px"}),

                html.Hr(style={"border": "none", "borderTop": f"1px solid {C_BORDER}", "margin": "20px 0 4px"}),

                html.Label("Bedrooms", style=LABEL),
                dcc.Slider(id="inp-bedrooms", min=0, max=6, step=1, value=1,
                           marks={i: str(i) for i in range(7)},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Label("Bathrooms", style=LABEL),
                dcc.Slider(id="inp-bathrooms", min=0.5, max=5, step=0.5, value=1,
                           marks={i: str(i) for i in range(1, 6)},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Label("Accommodates (guests)", style=LABEL),
                dcc.Slider(id="inp-accommodates", min=1, max=16, step=1, value=2,
                           marks={i: str(i) for i in [1, 2, 4, 6, 8, 10, 12, 16]},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Hr(style={"border": "none", "borderTop": f"1px solid {C_BORDER}", "margin": "20px 0 4px"}),

                html.Label("Amenity Count", style=LABEL),
                dcc.Slider(id="inp-amenities", min=0, max=80, step=5, value=30,
                           marks={i: str(i) for i in [0, 20, 40, 60, 80]},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Label("Walk Score (0–100)", style=LABEL),
                dcc.Slider(id="inp-walkscore", min=0, max=100, step=5, value=70,
                           marks={i: str(i) for i in [0, 25, 50, 75, 100]},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Label("Review Score (0–5)", style=LABEL),
                dcc.Slider(id="inp-review", min=0, max=5, step=0.1, value=4.7,
                           marks={i: str(i) for i in [0, 1, 2, 3, 4, 5]},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Label("Years as Host", style=LABEL),
                dcc.Slider(id="inp-years-host", min=0, max=15, step=0.5, value=3,
                           marks={i: str(i) for i in [0, 3, 6, 9, 12, 15]},
                           tooltip={"placement": "bottom", "always_visible": False}),

                html.Hr(style={"border": "none", "borderTop": f"1px solid {C_BORDER}", "margin": "20px 0 16px"}),

                html.Label("Superhost?", style=LABEL),
                dcc.RadioItems(id="inp-superhost",
                    options=[{"label": "  Yes", "value": 1}, {"label": "  No", "value": 0}],
                    value=0, inline=True,
                    style={"fontFamily": FONT, "fontSize": "13px", "marginBottom": "20px"}),

                html.Button("Predict Price →", id="btn-predict", n_clicks=0, style={
                    "background": C_ACCENT,
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "13px 24px",
                    "fontSize": "13px",
                    "fontWeight": "700",
                    "fontFamily": FONT,
                    "letterSpacing": "0.04em",
                    "cursor": "pointer",
                    "width": "100%",
                }),
            ]),

            # Results panel
            html.Div(id="results-panel"),
        ]),
    ])


@app.callback(
    Output("inp-neighbourhood", "options"),
    Output("inp-neighbourhood", "value"),
    Input("inp-city", "value"),
)
def update_neighbourhood_options(city):
    options = [{"label": n, "value": n} for n in nbhd_by_city.get(city, [])]
    return options, None


@app.callback(
    Output("results-panel", "children"),
    Input("btn-predict", "n_clicks"),
    State("inp-city", "value"),
    State("inp-neighbourhood", "value"),
    State("inp-room-type", "value"),
    State("inp-bedrooms", "value"),
    State("inp-bathrooms", "value"),
    State("inp-accommodates", "value"),
    State("inp-amenities", "value"),
    State("inp-walkscore", "value"),
    State("inp-superhost", "value"),
    State("inp-review", "value"),
    State("inp-years-host", "value"),
    prevent_initial_call=True,
)
def predict_price(n_clicks, city, neighbourhood, room_type, bedrooms, bathrooms,
                  accommodates, amenity_count, walkscore, superhost, review_score, years_host):

    neighbourhood = neighbourhood or df[df["city"] == city]["neighbourhood_cleansed"].mode()[0]

    # Build input row with all features the XGBoost pipeline expects
    row = {c: np.nan for c in xgb_data["NUMERIC_COLS"] + xgb_data["CATEGORICAL_COLS"]}
    # Fill in user-provided values
    row.update({
        "city": city,
        "room_type": room_type,
        "neighbourhood_cleansed": neighbourhood,
        "neighbourhood_group_cleansed": np.nan,
        "property_type": "Entire rental unit",
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "bathroom_shared": 0,
        "accommodates": accommodates,
        "amenity_count": amenity_count,
        "walkscore": walkscore,
        "transit_score": walkscore * 0.9,
        "bike_score": walkscore * 0.7,
        "host_is_superhost": superhost,
        "review_scores_rating": review_score,
        "review_scores_accuracy": review_score,
        "review_scores_cleanliness": review_score,
        "review_scores_checkin": review_score,
        "review_scores_communication": review_score,
        "review_scores_location": review_score,
        "review_scores_value": review_score,
        "years_as_host": years_host,
        "number_of_reviews": 20,
        "reviews_per_month": 1.5,
        "availability_365": 180,
        "beds": bedrooms + 1,
        "minimum_nights": 2,
        "maximum_nights": 365,
        "instant_bookable": 0,
        "host_listings_count": 1,
        "host_identity_verified": 1,
        "calculated_host_listings_count": 1,
        # interaction terms
        "bedrooms_x_accommodates": bedrooms * accommodates,
        "review_x_superhost": review_score * superhost,
        "amenity_count_x_accommodates": amenity_count * accommodates,
        "bedrooms_x_walkability": bedrooms * walkscore,
        "tenure_x_superhost": years_host * superhost,
        "bathrooms_x_accommodates": bathrooms * accommodates,
        "review_x_amenity_count": review_score * amenity_count,
        "availability_x_reviews_per_month": 180 * 1.5,
    })

    # Use median lat/lon for the neighbourhood (fall back to city median if neighbourhood unknown)
    city_df  = df[df["city"] == city]
    nbhd_df  = city_df[city_df["neighbourhood_cleansed"] == neighbourhood]
    ref_df   = nbhd_df if len(nbhd_df) > 0 else city_df
    row["latitude"]  = ref_df["latitude"].median()
    row["longitude"] = ref_df["longitude"].median()

    # Fill PCA amenity columns with 0 (average listing)
    for col in xgb_data["NUMERIC_COLS"]:
        if col.startswith("pca_amenity_") and np.isnan(row.get(col, np.nan)):
            row[col] = 0.0

    # Get predictions from all three models
    # Ridge expects 343 cols (including 191 raw amenity dummies) — fill missing with 0
    ridge_row = {c: row.get(c, 0) for c in ridge_feature_cols}
    X_ridge   = pd.DataFrame([ridge_row])[ridge_feature_cols]

    preds = {}
    for name, pipe, X_in in [
        ("Ridge",         ridge_pipe, X_ridge),
        ("Random Forest", rf_pipe,    pd.DataFrame([row])[rf_data["NUMERIC_COLS"] + rf_data["CATEGORICAL_COLS"]]),
        ("XGBoost",       xgb_pipe,   pd.DataFrame([row])[xgb_data["NUMERIC_COLS"] + xgb_data["CATEGORICAL_COLS"]]),
    ]:
        try:
            log_pred = pipe.predict(X_in)[0]
            preds[name] = np.exp(log_pred)
        except Exception:
            preds[name] = None

    xgb_pred = preds.get("XGBoost")
    if xgb_pred is None:
        return html.Div(html.P("Prediction failed - check inputs.", style={"color": "red"}), style=CARD)

    # Confidence range: ±1 RMSE in log space, back-transformed
    rmse_log     = xgb_data["metrics"]["test_rmse"]
    log_pred_val = np.log(xgb_pred)
    low  = np.exp(log_pred_val - rmse_log)
    high = np.exp(log_pred_val + rmse_log)

    # Prediction card
    pred_card = html.Div(style=CARD, children=[
        html.Div("Estimated Nightly Price", style={
            "fontFamily": FONT, "fontSize": "10px", "letterSpacing": "0.25em",
            "textTransform": "uppercase", "color": C_MUTED,
            "fontWeight": "600", "marginBottom": "12px",
        }),
        html.Div(style={"display": "flex", "alignItems": "flex-end", "gap": "40px", "flexWrap": "wrap"}, children=[
            html.Div([
                html.Div(f"${xgb_pred:,.0f}", style={
                    "fontFamily": FONT, "fontSize": "72px", "fontWeight": "800",
                    "color": C_ACCENT, "lineHeight": "1", "letterSpacing": "-0.02em",
                }),
                html.Div(f"per night  ·  Range: ${low:,.0f} – ${high:,.0f}",
                         style={"fontSize": "13px", "color": C_MUTED, "marginTop": "6px"}),
                html.Div("XGBoost model · ±1 RMSE confidence range",
                         style={"fontSize": "11px", "color": "#bbb", "marginTop": "3px"}),
            ]),
            html.Div(style={"borderLeft": f"2px solid {C_BORDER}", "paddingLeft": "32px"}, children=[
                html.Div("Model Comparison", style={
                    "fontFamily": FONT, "fontSize": "10px", "letterSpacing": "0.2em",
                    "textTransform": "uppercase", "color": C_MUTED,
                    "fontWeight": "600", "marginBottom": "12px",
                }),
                html.Table(style={"fontFamily": FONT, "fontSize": "13px", "borderCollapse": "collapse"}, children=[
                    html.Tbody([
                        html.Tr([
                            html.Td(label, style={"paddingRight": "20px", "paddingBottom": "8px", "color": C_MUTED}),
                            html.Td(
                                f"${preds[key]:,.0f}" if preds.get(key) else "-",
                                style={
                                    "fontWeight": "700",
                                    "color": C_ACCENT if key == "XGBoost" else C_DARK,
                                    "paddingBottom": "8px",
                                }
                            ),
                        ])
                        for label, key in [("Ridge", "Ridge"), ("Random Forest", "Random Forest"), ("XGBoost ★", "XGBoost")]
                    ])
                ]),
            ]),
        ]),
    ])

    # Nearby listings map
    price_low  = max(0, xgb_pred - 75)
    price_high = xgb_pred + 75
    nearby = city_df[
        (city_df["price_usd"] >= price_low) &
        (city_df["price_usd"] <= price_high) &
        city_df["latitude"].notna()
    ].sample(min(500, len(city_df)), random_state=42)

    city_center = {"nyc": (40.7128, -74.0060), "la": (34.0522, -118.2437), "chi": (41.8781, -87.6298)}
    lat_c, lon_c = city_center[city]

    nearby = nearby.rename(columns={
        "price_usd": "Price (USD)",
        "room_type": "Room Type",
        "neighbourhood_cleansed": "Neighbourhood",
    })

    map_fig = px.scatter_map(
        nearby, lat="latitude", lon="longitude", color="Price (USD)",
        color_continuous_scale="RdYlGn_r",
        hover_data=["Price (USD)", "Room Type", "Neighbourhood"],
        zoom=11, center={"lat": lat_c, "lon": lon_c},
        map_style="carto-positron",
        title=f"Comparable Listings · {CITY_LABELS[city]} · ${price_low:.0f}–${price_high:.0f}/night",
        opacity=0.8, template=PLOTLY_TPL, height=560,
    )
    map_fig.update_layout(
        coloraxis_colorbar_title="Price (USD)",
        margin={"t": 44, "b": 0, "l": 0, "r": 0},
        font={"family": FONT},
    )

    map_card = html.Div(style=CARD, children=[
        html.Div("Comparable Nearby Listings", style={
            "fontFamily": FONT, "fontSize": "18px", "fontWeight": "700",
            "color": C_DARK, "marginBottom": "12px",
        }),
        dcc.Graph(figure=map_fig, config={"displayModeBar": False}),
    ])

    return html.Div([pred_card, map_card])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: EDA Explorer
# ─────────────────────────────────────────────────────────────────────────────
def _caption(text):
    return html.P(text, style={
        "fontSize": "12px", "color": C_MUTED, "marginTop": "8px",
        "marginBottom": 0, "lineHeight": "1.5", "fontFamily": FONT,
    })

def _chart_card(fig, caption):
    return html.Div(style=CARD, children=[
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        _caption(caption),
    ])

def render_eda():
    chart_font = {"family": FONT, "size": 12}
    title_font = {"family": FONT, "size": 16, "color": C_DARK}

    sample = df.sample(10000, random_state=42)

    price_fig = px.histogram(
        sample, x="price_usd", nbins=60,
        color="city", color_discrete_map=CITY_COLORS,
        category_orders={"city": ["la", "nyc", "chi"]},
        title="Price Distribution by City",
        labels={"price_usd": "Nightly Price (USD)", "city": "City"},
        template=PLOTLY_TPL, barmode="overlay", opacity=0.65,
    )
    price_fig.update_xaxes(range=[0, 600])
    price_fig.update_layout(font=chart_font, title_font=title_font)

    # Log price KDE by city (violin)
    violin_fig = px.violin(
        sample, x="city", y="log_price",
        color="city", color_discrete_map=CITY_COLORS,
        box=True, points=False,
        title="Log-Price Distribution by City",
        labels={"log_price": "log(price)", "city": "City"},
        template=PLOTLY_TPL,
    )
    violin_fig.update_layout(font=chart_font, title_font=title_font)

    ws_df = df[df["walkscore"].notna()].sample(min(5000, df["walkscore"].notna().sum()), random_state=42)
    ws_fig = px.scatter(
        ws_df, x="walkscore", y="log_price", color="city",
        color_discrete_map=CITY_COLORS, opacity=0.3, trendline="ols",
        title="Walk Score vs Log-Price",
        labels={"walkscore": "Walk Score", "log_price": "log(price)"},
        template=PLOTLY_TPL,
    )
    ws_fig.update_layout(font=chart_font, title_font=title_font)

    room_fig = px.box(
        sample, x="room_type", y="price_usd", color="city",
        color_discrete_map=CITY_COLORS,
        category_orders={"room_type": ["Entire home/apt", "Private room", "Hotel room", "Shared room"]},
        title="Price by Room Type",
        labels={"price_usd": "Nightly Price (USD)", "room_type": "Room Type"},
        template=PLOTLY_TPL,
    )
    room_fig.update_yaxes(range=[0, 800])
    room_fig.update_layout(font=chart_font, title_font=title_font, boxmode="group")

    # Top 15 neighbourhoods - one chart per city to avoid shared categorical axis
    nbhd_stats = (
        df.groupby(["city", "neighbourhood_cleansed"])["price_usd"]
        .median().reset_index()
        .rename(columns={"price_usd": "median_price"})
    )
    nbhd_captions = {
        "nyc": "NYC listings skew highest, Manhattan neighbourhoods like Tribeca and Fort Wadsworth command top rates.",
        "la":  "LA's coastal and hillside neighbourhoods (Malibu, Beverly Crest) lead on median price.",
        "chi": "Chicago's Loop and Near North Side dominate, reflecting downtown demand.",
    }
    nbhd_figs = []
    for city_key, city_label in CITY_LABELS.items():
        top = (
            nbhd_stats[nbhd_stats["city"] == city_key]
            .sort_values("median_price", ascending=False)
            .head(15)
            .sort_values("median_price")
        )
        fig = px.bar(
            top, x="median_price", y="neighbourhood_cleansed", orientation="h",
            title=f"Top 15 Neighbourhoods · {city_label}",
            labels={"median_price": "Median Price (USD)", "neighbourhood_cleansed": ""},
            template=PLOTLY_TPL, height=420,
            color_discrete_sequence=[CITY_COLORS[city_key]],
        )
        fig.update_layout(font=chart_font, title_font=title_font, showlegend=False,
                          margin={"l": 0, "r": 8, "t": 44, "b": 0})
        nbhd_figs.append(html.Div(style=CARD, children=[
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            _caption(nbhd_captions[city_key]),
        ]))

    return html.Div([
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            _chart_card(price_fig,  "NYC and Chicago listings are more tightly clustered; LA has a longer right tail due to luxury coastal properties. Clipped at $600 for readability."),
            _chart_card(violin_fig, "Log-price is roughly symmetric in all three cities, validating the log-transform for regression. NYC has a slightly higher median."),
            _chart_card(ws_fig,     "Higher walk scores correlate weakly with higher prices. NYC listings (denser, more walkable) cluster in the upper-right. ~42% of listings are missing walk scores due to API rate limits."),
            _chart_card(room_fig,   "Entire homes command 2–3× the price of private rooms. Hotel rooms are competitive with entire homes in NYC. Shared rooms are rare and priced lowest across all cities."),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px"}, children=nbhd_figs),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
def render_models():
    chart_font = {"family": FONT, "size": 12}
    title_font = {"family": FONT, "size": 16, "color": C_DARK}
    model_colors = [CITY_COLORS["nyc"], CITY_COLORS["la"], CITY_COLORS["chi"]]

    metrics_df = pd.DataFrame({
        "Model":      ["Ridge (RidgeCV)", "Random Forest", "XGBoost (BayesOpt)"],
        "RMSE (log)": [ridge_metrics["RMSE (log)"], rf_metrics["RMSE (log)"],  xgb_metrics["RMSE (log)"]],
        "MAE (log)":  [ridge_metrics["MAE (log)"],  rf_metrics["MAE (log)"],   xgb_metrics["MAE (log)"]],
        "R²":         [ridge_metrics["R²"],         rf_metrics["R²"],          xgb_metrics["R²"]],
        "RMSE (USD)": [ridge_metrics["RMSE (USD)"], rf_metrics["RMSE (USD)"],  xgb_metrics["RMSE (USD)"]],
    })

    rmse_fig = px.bar(
        metrics_df, x="Model", y="RMSE (log)", color="Model",
        color_discrete_sequence=model_colors,
        title="Test RMSE (log-price) - Lower is Better",
        template=PLOTLY_TPL, text_auto=".4f",
    )
    rmse_fig.update_traces(textposition="outside", marker_line_width=0)
    rmse_fig.update_layout(showlegend=False, font=chart_font, title_font=title_font, plot_bgcolor="white")

    r2_fig = px.bar(
        metrics_df, x="Model", y="R²", color="Model",
        color_discrete_sequence=model_colors,
        title="Test R² - Higher is Better",
        template=PLOTLY_TPL, text_auto=".4f",
        range_y=[0.6, 1.0],
    )
    r2_fig.update_traces(textposition="outside", marker_line_width=0)
    r2_fig.update_layout(showlegend=False, font=chart_font, title_font=title_font, plot_bgcolor="white")

    header_style = {
        "textAlign": "left", "padding": "10px 16px",
        "borderBottom": f"2px solid {C_DARK}", "fontFamily": FONT,
        "fontWeight": "700", "fontSize": "11px", "letterSpacing": "0.1em",
        "textTransform": "uppercase", "color": C_MUTED,
    }
    def cell_style(i, col):
        is_xgb = i == 2
        return {
            "padding": "12px 16px", "fontFamily": FONT, "fontSize": "14px",
            "color": CITY_COLORS["chi"] if is_xgb else C_DARK,
            "fontWeight": "700" if is_xgb and col != "Model" else ("600" if col == "Model" else "400"),
            "borderBottom": f"1px solid {C_BORDER}",
            "background": "#FAFAF8" if i % 2 == 0 else C_CARD,
        }

    table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(html.Tr([html.Th(col, style=header_style) for col in metrics_df.columns])),
            html.Tbody([
                html.Tr([html.Td(metrics_df.iloc[i][col], style=cell_style(i, col)) for col in metrics_df.columns])
                for i in range(len(metrics_df))
            ]),
        ]
    )

    shap_fig = px.bar(
        _shap_df, x="importance", y="feature", orientation="h",
        title="SHAP Feature Importance - XGBoost (Top 20)",
        labels={"importance": "Mean |SHAP value|", "feature": ""},
        template=PLOTLY_TPL, color_discrete_sequence=[CITY_COLORS["chi"]],
        height=480,
    )
    shap_fig.update_layout(showlegend=False, font=chart_font, title_font=title_font,
                           margin={"l": 0, "r": 8, "t": 44, "b": 0})

    rf_imp_fig = px.bar(
        _rf_df, x="importance", y="feature", orientation="h",
        title="Feature Importance - Random Forest (Top 20)",
        labels={"importance": "Impurity-based Importance", "feature": ""},
        template=PLOTLY_TPL, color_discrete_sequence=[CITY_COLORS["la"]],
        height=480,
    )
    rf_imp_fig.update_layout(showlegend=False, font=chart_font, title_font=title_font,
                             margin={"l": 0, "r": 8, "t": 44, "b": 0})

    return html.Div([
        html.Div(style=CARD, children=[
            html.Div("Model Performance Summary", style={
                "fontFamily": FONT, "fontSize": "20px", "fontWeight": "700",
                "color": C_DARK, "marginBottom": "16px",
            }),
            table,
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            html.Div(style=CARD, children=[dcc.Graph(figure=rmse_fig, config={"displayModeBar": False})]),
            html.Div(style=CARD, children=[dcc.Graph(figure=r2_fig,   config={"displayModeBar": False})]),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            html.Div(style=CARD, children=[
                dcc.Graph(figure=shap_fig, config={"displayModeBar": False}),
                html.P("Mean absolute SHAP value - how much each feature shifts the predicted log-price on average. 'Amenity Factor N' = PCA component of 191 binary amenity dummies.",
                       style={"fontSize": "12px", "color": C_MUTED, "marginTop": "8px", "marginBottom": 0, "fontFamily": FONT}),
            ]),
            html.Div(style=CARD, children=[
                dcc.Graph(figure=rf_imp_fig, config={"displayModeBar": False}),
                html.P("Impurity-based importance - average reduction in node impurity from splits on each feature across all trees. 'Amenity Factor N' = PCA component of 191 binary amenity dummies.",
                       style={"fontSize": "12px", "color": C_MUTED, "marginTop": "8px", "marginBottom": 0, "fontFamily": FONT}),
            ]),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
