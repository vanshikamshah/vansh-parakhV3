# ============================================================
#  PARAKH — Investor Protection Dashboard  (self-contained)
#  No CSV files required — all data generated inline.
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

# ── Colour palette ───────────────────────────────────────────
TEAL   = "#00A896"
ORANGE = "#E87722"
GOLD   = "#C9A84C"
RED    = "#C0392B"
GREEN  = "#1A7F4B"
PURPLE = "#6A5ACD"
MODEL_COLORS = {
    "KNN": TEAL,
    "Decision Tree": ORANGE,
    "Random Forest": GREEN,
    "Gradient Boosted": PURPLE,
}

# ════════════════════════════════════════════════════════════
#  INLINE DATA GENERATION  (replaces CSV loading entirely)
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Generating Parakh dataset…")
def generate_data():
    rng = np.random.default_rng(42)
    N = 500   # investors

    # ── demographics ─────────────────────────────────────────
    ages = rng.integers(22, 65, N)
    genders = rng.choice(["Male","Female","Other"], N, p=[0.62,0.36,0.02])
    city_tiers = rng.choice(["Tier 1","Tier 2","Tier 3"], N, p=[0.45,0.35,0.20])
    INCOME_ORDER = ["<3L","3L-6L","6L-10L","10L-20L","20L-50L","50L+"]
    income_idx = rng.choice(len(INCOME_ORDER), N,
                            p=[0.10,0.22,0.25,0.22,0.14,0.07])
    income_brackets = [INCOME_ORDER[i] for i in income_idx]
    educations = rng.choice(
        ["High School","Graduate","Post-Graduate","Professional"],
        N, p=[0.12,0.38,0.32,0.18]
    )
    years_investing = np.clip(rng.integers(1, 25, N), 1, 24).astype(float)
    financial_literacy = np.clip(
        rng.normal(5.5, 2.0, N), 1, 10
    ).round(1)
    RISK_AP = ["Conservative","Moderate","Aggressive"]
    risk_ap = rng.choice(RISK_AP, N, p=[0.30,0.45,0.25])
    platforms = rng.choice(
        ["Zerodha","Groww","Upstox","HDFC Securities","ICICI Direct","Others"],
        N, p=[0.25,0.22,0.18,0.12,0.10,0.13]
    )

    # ── portfolio ─────────────────────────────────────────────
    port_base = rng.lognormal(12.5, 1.2, N)
    total_portfolio = np.clip(port_base, 10_000, 50_000_000).round(-2)

    direct_eq_pct = np.clip(rng.beta(2, 3, N), 0.05, 0.80)
    mf_pct        = np.clip(rng.beta(3, 2, N) * (1 - direct_eq_pct), 0.05, 0.70)
    epf_pct       = np.clip(rng.beta(2, 5, N) * 0.3, 0.0, 0.30)
    fd_pct        = np.clip(1 - direct_eq_pct - mf_pct - epf_pct, 0.0, 0.50)

    direct_equity_inr  = (total_portfolio * direct_eq_pct).round(-2)
    mutual_funds_inr   = (total_portfolio * mf_pct).round(-2)
    epf_inr            = (total_portfolio * epf_pct).round(-2)
    fd_inr             = (total_portfolio * fd_pct).round(-2)
    nps_inr            = (total_portfolio * rng.uniform(0, 0.10, N)).round(-2)
    gold_digital_inr   = (total_portfolio * rng.uniform(0, 0.05, N)).round(-2)
    gold_physical_inr  = (total_portfolio * rng.uniform(0, 0.08, N)).round(-2)
    ppf_inr            = (total_portfolio * rng.uniform(0, 0.06, N)).round(-2)
    real_estate_inr    = (total_portfolio * rng.uniform(0, 0.20, N)).round(-2)

    eq_total = direct_equity_inr + mutual_funds_inr
    equity_to_total = np.where(
        total_portfolio > 0, (eq_total / total_portfolio * 100), 0
    ).round(2)

    wealth_frag = np.clip(rng.normal(5, 1.5, N), 1, 10).round(1)
    pct_single_platform = np.clip(rng.uniform(30, 95, N), 30, 95).round(1)

    # ── MF diversification illusion ───────────────────────────
    num_mfs = rng.integers(1, 12, N)
    num_stocks = rng.integers(1, 30, N)
    perceived_div = np.clip(rng.normal(6.5, 1.5, N), 2, 10).round(1)
    # actual is lower due to overlap
    actual_div = np.clip(
        perceived_div - rng.uniform(1.5, 4.5, N), 1, 8
    ).round(1)
    div_gap = (perceived_div - actual_div).round(2)
    top15_cover = np.clip(rng.normal(70, 8, N), 45, 95).round(1)
    hhi = np.clip(rng.normal(0.18, 0.07, N), 0.02, 0.60).round(4)
    overlap_flag = (top15_cover > 65).astype(int)
    conc_flag    = (hhi > 0.20).astype(int)

    SECTORS = ["IT","Banking","FMCG","Pharma","Auto","Energy","Infra","Real Estate"]
    top_sector = rng.choice(SECTORS, N)
    top_sector_pct = np.clip(rng.normal(38, 10, N), 15, 75).round(1)

    # ── loss / risk ───────────────────────────────────────────
    # loss proportional to concentration + low literacy + aggressive risk
    loss_prob = 0.1 + 0.3 * conc_flag + 0.15 * (financial_literacy < 5) + \
                0.1 * (risk_ap == "Aggressive")
    total_loss = np.where(
        rng.random(N) < loss_prob,
        rng.lognormal(9, 1.2, N),
        0
    ).round(-2)
    risk_loss_ratio = np.where(
        total_portfolio > 0, (total_loss / total_portfolio * 100), 0
    ).round(2)

    # ── behavioural / awareness ───────────────────────────────
    knows_var       = rng.choice([0,1], N, p=[0.62,0.38])
    checks_corr     = rng.choice([0,1], N, p=[0.73,0.27])
    checks_sector   = rng.choice([0,1], N, p=[0.60,0.40])
    uses_stop_loss  = rng.choice([0,1], N, p=[0.68,0.32])
    simulates       = rng.choice([0,1], N, p=[0.78,0.22])
    tracks_all      = rng.choice([0,1], N, p=[0.55,0.45])

    risk_awareness = (
        knows_var + checks_corr + checks_sector +
        uses_stop_loss + simulates + tracks_all +
        (financial_literacy / 10 * 4)
    ).clip(0, 10).round(1)

    heard_aa  = rng.choice([0,1], N, p=[0.58,0.42])
    has_aa    = np.where(heard_aa == 1,
                         rng.choice([0,1], N, p=[0.40,0.60]), 0)

    # ── Parakh adoption ───────────────────────────────────────
    parakh_logit = (
        -1.5
        + 0.04 * (ages - 35)
        + 0.15 * (financial_literacy - 5)
        + 0.5 * (conc_flag)
        + 0.3 * has_aa
        + 0.3 * (div_gap > 2.5).astype(int)
        + rng.normal(0, 0.5, N)
    )
    parakh_prob = 1 / (1 + np.exp(-parakh_logit))
    would_use_parakh = (parakh_prob > 0.5).astype(int)
    # bump to ~63% adoption
    deficit = int(0.63 * N) - would_use_parakh.sum()
    if deficit > 0:
        no_idx = np.where(would_use_parakh == 0)[0]
        flip_idx = rng.choice(no_idx, min(deficit, len(no_idx)), replace=False)
        would_use_parakh[flip_idx] = 1

    wtp = np.where(
        would_use_parakh == 1,
        np.clip(rng.normal(199, 80, N), 49, 499),
        np.clip(rng.normal(79,  50, N),  0, 249)
    ).round(0)

    PAIN_POINTS = [
        "No single view of all assets",
        "Cannot assess real diversification",
        "Lack of risk alerts",
        "Over-reliance on broker advice",
        "No pre-trade simulation",
    ]
    pain_point_idx = rng.integers(0, 5, N)
    main_pain_point = [PAIN_POINTS[i] for i in pain_point_idx]

    parakh_need = (
        0.3 * hhi / hhi.max()
        + 0.25 * (div_gap / div_gap.max())
        + 0.2 * (1 - financial_literacy / 10)
        + 0.15 * (risk_loss_ratio / (risk_loss_ratio.max() + 1e-6))
        + 0.1 * (1 - risk_awareness / 10)
    )
    parakh_need = (parakh_need / parakh_need.max() * 10).round(2)

    ids = [f"INV{str(i+1).zfill(4)}" for i in range(N)]

    m1 = pd.DataFrame({
        "investor_id":                    ids,
        "age":                            ages,
        "gender":                         genders,
        "city_tier":                      city_tiers,
        "income_bracket":                 income_brackets,
        "education":                      educations,
        "years_investing":                years_investing,
        "financial_literacy_score":       financial_literacy,
        "self_reported_risk_appetite":    risk_ap,
        "primary_platform":               platforms,
        "total_portfolio_value_inr":      total_portfolio,
        "num_mutual_funds":               num_mfs,
        "num_direct_stocks":              num_stocks,
        "perceived_diversification_score":perceived_div,
        "actual_diversification_score":   actual_div,
        "diversification_gap":            div_gap,
        "top15_stocks_cover_pct_of_mf":  top15_cover,
        "herfindahl_index":               hhi,
        "overlap_risk_flag":              overlap_flag,
        "high_concentration_flag":        conc_flag,
        "top_sector":                     top_sector,
        "top_sector_concentration_pct":   top_sector_pct,
        "total_loss_last_18mo_inr":       total_loss,
        "risk_loss_ratio":                risk_loss_ratio,
        "direct_equity_inr":              direct_equity_inr,
        "mutual_funds_inr":               mutual_funds_inr,
        "epf_inr":                        epf_inr,
        "nps_inr":                        nps_inr,
        "gold_digital_inr":               gold_digital_inr,
        "gold_physical_inr":              gold_physical_inr,
        "fixed_deposits_inr":             fd_inr,
        "ppf_inr":                        ppf_inr,
        "real_estate_estimate_inr":       real_estate_inr,
        "equity_to_total_wealth_pct":     equity_to_total,
        "wealth_fragmentation_score":     wealth_frag,
        "pct_wealth_on_single_platform":  pct_single_platform,
        "knows_what_var_is":              knows_var,
        "checks_portfolio_correlation":   checks_corr,
        "checks_sector_concentration":    checks_sector,
        "uses_stop_loss":                 uses_stop_loss,
        "simulates_before_trading":       simulates,
        "tracks_all_assets_in_one_place": tracks_all,
        "risk_awareness_score_out_of_10": risk_awareness,
        "heard_of_account_aggregator":    heard_aa,
        "has_aa_consent":                 has_aa,
        "would_use_parakh_simulator":     would_use_parakh,
        "willingness_to_pay_monthly_inr": wtp,
        "main_pain_point":                main_pain_point,
        "parakh_need_score":              parakh_need,
    })

    # ── Trade-level data (M2) ─────────────────────────────────
    TRADES_PER = rng.integers(4, 12, N)          # per investor
    T = TRADES_PER.sum()

    inv_rep   = np.repeat(np.arange(N), TRADES_PER)
    trade_ids = [f"TRD{str(i+1).zfill(6)}" for i in range(T)]

    DECISION_BASIS = [
        "Own Research","Technical Analysis","Broker Recommendation",
        "News/Media","FOMO","Social Media Tip","Astrology/Gut"
    ]
    basis_probs = [0.18,0.20,0.20,0.15,0.12,0.10,0.05]
    trade_basis_idx = rng.choice(len(DECISION_BASIS), T, p=basis_probs)
    trade_basis = [DECISION_BASIS[i] for i in trade_basis_idx]

    # loss rate by basis (matches spec: gut ~64%, own research ~18%)
    BASIS_LOSS_RATE = {
        "Own Research":0.18, "Technical Analysis":0.28,
        "Broker Recommendation":0.38, "News/Media":0.42,
        "FOMO":0.55, "Social Media Tip":0.60, "Astrology/Gut":0.64
    }
    loss_occ = np.array([
        int(rng.random() < BASIS_LOSS_RATE[b]) for b in trade_basis
    ])

    TRADE_TYPES  = ["Buy","Sell","SIP","SWP"]
    ASSET_CLASSES = ["Equity","Mutual Fund","Gold","Crypto","Debt","NPS"]

    trade_types   = rng.choice(TRADE_TYPES, T, p=[0.50,0.25,0.15,0.10])
    asset_classes = rng.choice(ASSET_CLASSES, T, p=[0.40,0.30,0.08,0.07,0.10,0.05])
    trade_amounts = np.clip(rng.lognormal(9.5, 1.3, T), 500, 2_000_000).round(-2)
    market_event  = rng.choice([0,1], T, p=[0.72,0.28])
    checked_risk  = rng.choice([0,1], T, p=[0.905,0.095])   # ~9.5%
    uninformed    = np.isin(
        trade_basis_idx,
        [DECISION_BASIS.index(b) for b in
         ["FOMO","Social Media Tip","Astrology/Gut"]]
    ).astype(int)
    regret = np.where(
        loss_occ == 1,
        np.clip(rng.normal(6.5, 1.5, T), 2, 10),
        np.clip(rng.normal(3.0, 1.5, T), 1, 8)
    ).round(1)

    m2 = pd.DataFrame({
        "trade_id":                       trade_ids,
        "investor_id":                    [ids[i] for i in inv_rep],
        "trade_type":                     trade_types,
        "asset_class":                    asset_classes,
        "decision_basis":                 trade_basis,
        "trade_amount_inr":               trade_amounts,
        "market_event_flag":              market_event,
        "checked_risk_before_trade":      checked_risk,
        "uninformed_trade_flag":          uninformed,
        "loss_occurred":                  loss_occ,
        "post_trade_regret_score":        regret,
        # join columns from m1
        "age":                            ages[inv_rep],
        "years_investing":                years_investing[inv_rep],
        "financial_literacy_score":       financial_literacy[inv_rep],
        "risk_awareness_score_out_of_10": risk_awareness[inv_rep],
        "income_bracket":                 [INCOME_ORDER[income_idx[i]] for i in inv_rep],
        "self_reported_risk_appetite":    risk_ap[inv_rep],
        "city_tier":                      city_tiers[inv_rep],
        "has_aa_consent":                 has_aa[inv_rep],
        "would_use_parakh_simulator":     would_use_parakh[inv_rep],
        "willingness_to_pay_monthly_inr": wtp[inv_rep],
        "total_portfolio_value_inr":      total_portfolio[inv_rep],
        "herfindahl_index":               hhi[inv_rep],
    })

    return m1, m2


# ── load once ─────────────────────────────────────────────────
m1, m2 = generate_data()


# ════════════════════════════════════════════════════════════
#  ML HELPERS
# ════════════════════════════════════════════════════════════
INCOME_ORDER = ["<3L","3L-6L","6L-10L","10L-20L","20L-50L","50L+"]

def prep_model_a(df):
    """Trade-level → predict loss_occurred."""
    d = df.copy()
    d["trade_type_enc"]  = d["trade_type"].map(
        {"Buy":0,"Sell":1,"SIP":2,"SWP":3}).fillna(0)
    d["asset_class_enc"] = d["asset_class"].map(
        {"Equity":0,"Mutual Fund":1,"Gold":2,"Crypto":3,"Debt":4,"NPS":5}).fillna(0)
    d["log_trade"]  = np.log1p(d["trade_amount_inr"])
    basis_risk = {
        "Own Research":0,"Technical Analysis":1,"Broker Recommendation":2,
        "News/Media":3,"FOMO":4,"Social Media Tip":5,"Astrology/Gut":6
    }
    d["basis_risk_enc"] = d["decision_basis"].map(basis_risk).fillna(3)
    d["log_port"]   = np.log1p(d["total_portfolio_value_inr"])
    d["income_enc"] = d["income_bracket"].apply(
        lambda x: INCOME_ORDER.index(x) if x in INCOME_ORDER else 0)
    d["city_enc"]   = d["city_tier"].map(
        {"Tier 1":0,"Tier 2":1,"Tier 3":2}).fillna(0)
    d["risk_ap_enc"]= d["self_reported_risk_appetite"].map(
        {"Conservative":0,"Moderate":1,"Aggressive":2}).fillna(1)

    feats = [
        "trade_type_enc","asset_class_enc","log_trade","basis_risk_enc",
        "market_event_flag","checked_risk_before_trade","age","years_investing",
        "financial_literacy_score","risk_awareness_score_out_of_10","log_port",
        "uninformed_trade_flag","city_enc","income_enc","risk_ap_enc",
        "has_aa_consent","would_use_parakh_simulator","willingness_to_pay_monthly_inr"
    ]
    X = d[feats].fillna(0).values
    y = d["loss_occurred"].values
    return X, y, feats

def prep_model_b(df):
    """Investor-level → predict would_use_parakh_simulator."""
    d = df.copy()
    d["income_enc"] = d["income_bracket"].apply(
        lambda x: INCOME_ORDER.index(x) if x in INCOME_ORDER else 0)
    d["risk_ap_enc"]= d["self_reported_risk_appetite"].map(
        {"Conservative":0,"Moderate":1,"Aggressive":2}).fillna(1)
    d["city_enc"]   = d["city_tier"].map(
        {"Tier 1":0,"Tier 2":1,"Tier 3":2}).fillna(0)
    d["log_port"]   = np.log1p(d["total_portfolio_value_inr"])
    d["log_loss"]   = np.log1p(d["total_loss_last_18mo_inr"])

    feats = [
        "age","years_investing","financial_literacy_score",
        "risk_awareness_score_out_of_10","income_enc","risk_ap_enc","city_enc",
        "log_port","log_loss","herfindahl_index","diversification_gap",
        "top15_stocks_cover_pct_of_mf","overlap_risk_flag","high_concentration_flag",
        "equity_to_total_wealth_pct","wealth_fragmentation_score",
        "pct_wealth_on_single_platform","knows_what_var_is",
        "checks_portfolio_correlation","checks_sector_concentration",
        "uses_stop_loss","simulates_before_trading","tracks_all_assets_in_one_place",
        "heard_of_account_aggregator","has_aa_consent",
        "willingness_to_pay_monthly_inr","parakh_need_score","risk_loss_ratio"
    ]
    X = d[feats].fillna(0).values
    y = d["would_use_parakh_simulator"].values
    return X, y, feats


@st.cache_resource(show_spinner="Training ML models…")
def train_all_models(X_arr, y_tup, label="model"):
    y = np.array(y_tup)
    seed = 42
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y, test_size=0.2, random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    sc   = StandardScaler()
    Xs_tr = sc.fit_transform(X_tr)
    Xs_te = sc.transform(X_te)

    mdls = {
        "KNN":            KNeighborsClassifier(n_neighbors=9),
        "Decision Tree":  DecisionTreeClassifier(
                              max_depth=6, min_samples_leaf=20, random_state=seed),
        "Random Forest":  RandomForestClassifier(
                              n_estimators=200, max_depth=8,
                              min_samples_leaf=10, random_state=seed),
        "Gradient Boosted": GradientBoostingClassifier(
                              n_estimators=200, learning_rate=0.08,
                              max_depth=5, random_state=seed),
    }

    results = {}
    for name, mdl in mdls.items():
        Xtr = Xs_tr if name == "KNN" else X_tr
        Xte = Xs_te if name == "KNN" else X_te
        mdl.fit(Xtr, y_tr)
        y_pred = mdl.predict(Xte)
        y_prob = mdl.predict_proba(Xte)[:, 1]

        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        total = tn + fp + fn + tp
        fpr, tpr, _ = roc_curve(y_te, y_prob)

        fi = None
        if hasattr(mdl, "feature_importances_"):
            fi = mdl.feature_importances_

        results[name] = {
            "acc":  accuracy_score(y_te, y_pred),
            "prec": precision_score(y_te, y_pred, zero_division=0),
            "rec":  recall_score(y_te, y_pred, zero_division=0),
            "f1":   f1_score(y_te, y_pred, zero_division=0),
            "auc":  roc_auc_score(y_te, y_prob),
            "fp_pct": fp / total * 100 if total else 0,
            "fn_pct": fn / total * 100 if total else 0,
            "cm":   cm,
            "fpr":  fpr,
            "tpr":  tpr,
            "feat_imp": fi,
        }
    return results


# ════════════════════════════════════════════════════════════
#  PAGE CONFIG & NAVIGATION
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Parakh – Investor Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] {background: #0D2B45;}
  [data-testid="stSidebar"] * {color: #E8F4FD !important;}
  .metric-card {
      background: white; border-radius:12px; padding:20px;
      box-shadow:0 2px 8px rgba(0,0,0,.08); text-align:center;
  }
  .metric-value {font-size:2rem; font-weight:700; color:#00A896;}
  .metric-label {font-size:.85rem; color:#6B7280; margin-top:4px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛡️ Parakh")
    st.markdown("*Empowering India's Retail Investors*")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Descriptive Analytics",
        "🔬 Diagnostic Analytics",
        "🤖 ML Classification",
        "💡 Business Insights",
    ])
    st.divider()
    st.caption(f"Dataset: {len(m1):,} investors · {len(m2):,} trades")


# ════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🛡️ Parakh — Investor Protection Dashboard")
    st.markdown(
        "Built on India's **Account Aggregator** infrastructure, Parakh helps "
        "retail investors see the diversification illusion, understand real risk, "
        "and simulate portfolio impact before trading."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    conc_pct = m1["high_concentration_flag"].mean() * 100
    adopt_pct = m1["would_use_parakh_simulator"].mean() * 100
    check_pct = (m2["checked_risk_before_trade"].mean() * 100)
    loss_pct  = m2["loss_occurred"].mean() * 100

    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [conc_pct, adopt_pct, check_pct, loss_pct],
        ["Concentration Risk","Parakh Adoption","Check Risk Before Trade","Trades Resulted in Loss"]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val:.1f}%</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Why Parakh?")
    col_l, col_r = st.columns(2)
    with col_l:
        st.info(
            "**The Diversification Illusion**\n\n"
            "Retail investors hold 5-10 mutual funds thinking they're diversified. "
            "But the top-15 stocks account for ~70% of most MF portfolios, creating "
            "massive hidden overlap. Parakh's HHI engine reveals the true exposure."
        )
        avg_gap = m1["diversification_gap"].mean()
        st.metric("Avg. Diversification Gap", f"{avg_gap:.1f} / 10",
                  help="Perceived minus Actual diversification score")
    with col_r:
        st.warning(
            "**Uninformed Trading**\n\n"
            f"Over {m2['uninformed_trade_flag'].mean()*100:.0f}% of trades are driven by "
            "FOMO, social-media tips, or astrology — the highest-loss decision bases. "
            "Parakh's pre-trade simulator bridges the awareness gap."
        )
        st.metric("Avg. Trade Loss Rate (Gut/FOMO)",
                  f"{m2[m2['decision_basis'].isin(['Astrology/Gut','FOMO'])]['loss_occurred'].mean()*100:.1f}%")


# ════════════════════════════════════════════════════════════
#  PAGE 2 — DESCRIPTIVE ANALYTICS
# ════════════════════════════════════════════════════════════
elif page == "📊 Descriptive Analytics":
    st.title("📊 Descriptive Analytics")
    st.caption("7 core questions about our investor population")

    # Q1 — Demographic profile
    st.subheader("Q1 · Who are our investors?")
    c1, c2, c3 = st.columns(3)
    with c1:
        cnt = m1["gender"].value_counts().reset_index()
        fig = px.pie(cnt, names="gender", values="count",
                     color_discrete_sequence=[TEAL,ORANGE,GOLD],
                     title="Gender Split")
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cnt2 = m1["city_tier"].value_counts().reset_index()
        fig2 = px.bar(cnt2, x="city_tier", y="count",
                      color="city_tier",
                      color_discrete_sequence=[TEAL,ORANGE,GOLD],
                      title="City Tier Distribution")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    with c3:
        fig3 = px.histogram(m1, x="age", nbins=20,
                            color_discrete_sequence=[PURPLE],
                            title="Age Distribution")
        st.plotly_chart(fig3, use_container_width=True)

    # Q2 — Portfolio composition
    st.subheader("Q2 · How is wealth allocated?")
    asset_cols = ["direct_equity_inr","mutual_funds_inr","fixed_deposits_inr",
                  "epf_inr","gold_physical_inr","gold_digital_inr",
                  "nps_inr","ppf_inr","real_estate_estimate_inr"]
    asset_labels = ["Direct Equity","Mutual Funds","Fixed Deposits",
                    "EPF","Gold (Physical)","Gold (Digital)",
                    "NPS","PPF","Real Estate"]
    totals = [m1[c].sum() for c in asset_cols]
    fig_alloc = px.pie(
        names=asset_labels, values=totals,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Aggregate Wealth Allocation Across 500 Investors"
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    # Q3 — Diversification gap
    st.subheader("Q3 · How severe is the Diversification Illusion?")
    c1, c2 = st.columns(2)
    with c1:
        fig_gap = px.histogram(m1, x="diversification_gap", nbins=25,
                               color_discrete_sequence=[RED],
                               title="Distribution of Diversification Gap")
        fig_gap.add_vline(x=m1["diversification_gap"].mean(),
                          line_dash="dash", line_color=TEAL,
                          annotation_text=f"Mean={m1['diversification_gap'].mean():.2f}")
        st.plotly_chart(fig_gap, use_container_width=True)
    with c2:
        fig_hhi = px.histogram(m1, x="herfindahl_index", nbins=25,
                               color_discrete_sequence=[ORANGE],
                               title="HHI Distribution (>0.20 = Concentrated)")
        fig_hhi.add_vline(x=0.20, line_dash="dash", line_color=RED,
                          annotation_text="Concentration threshold")
        st.plotly_chart(fig_hhi, use_container_width=True)

    # Q4 — Risk behaviour
    st.subheader("Q4 · How risk-aware are investors?")
    behaviour_cols = [
        "knows_what_var_is","checks_portfolio_correlation",
        "checks_sector_concentration","uses_stop_loss",
        "simulates_before_trading","tracks_all_assets_in_one_place"
    ]
    behaviour_labels = [
        "Knows VaR","Checks Correlation","Checks Sector","Uses Stop-Loss",
        "Simulates Before Trade","Tracks All Assets"
    ]
    pcts = [m1[c].mean()*100 for c in behaviour_cols]
    fig_risk = go.Figure(go.Bar(
        x=pcts, y=behaviour_labels, orientation="h",
        marker_color=TEAL,
        text=[f"{v:.1f}%" for v in pcts], textposition="outside"
    ))
    fig_risk.update_layout(
        title="% Investors Practising Each Risk Behaviour",
        xaxis_title="Percentage (%)", xaxis_range=[0,100]
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Q5 — Trade patterns
    st.subheader("Q5 · What drives trade decisions?")
    c1, c2 = st.columns(2)
    with c1:
        basis_cnt = m2["decision_basis"].value_counts().reset_index()
        fig_basis = px.bar(basis_cnt, x="count", y="decision_basis",
                           orientation="h",
                           color_discrete_sequence=[PURPLE],
                           title="Trades by Decision Basis")
        st.plotly_chart(fig_basis, use_container_width=True)
    with c2:
        loss_by_basis = (
            m2.groupby("decision_basis")["loss_occurred"].mean() * 100
        ).reset_index()
        loss_by_basis.columns = ["Basis","Loss Rate (%)"]
        loss_by_basis = loss_by_basis.sort_values("Loss Rate (%)", ascending=True)
        fig_lr = px.bar(loss_by_basis, x="Loss Rate (%)", y="Basis",
                        orientation="h",
                        color="Loss Rate (%)",
                        color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],
                        title="Loss Rate by Decision Basis (%)")
        st.plotly_chart(fig_lr, use_container_width=True)

    # Q6 — Parakh adoption
    st.subheader("Q6 · Who would adopt Parakh?")
    c1, c2 = st.columns(2)
    with c1:
        adopt_by_lit = m1.groupby(
            pd.cut(m1["financial_literacy_score"], bins=[0,3,6,10],
                   labels=["Low (0-3)","Medium (3-6)","High (6-10)"])
        )["would_use_parakh_simulator"].mean() * 100
        fig_lit = px.bar(x=adopt_by_lit.index.astype(str),
                         y=adopt_by_lit.values,
                         color_discrete_sequence=[TEAL],
                         title="Parakh Adoption by Financial Literacy")
        fig_lit.update_layout(yaxis_title="% Would Adopt")
        st.plotly_chart(fig_lit, use_container_width=True)
    with c2:
        adopt_by_city = (
            m1.groupby("city_tier")["would_use_parakh_simulator"].mean() * 100
        ).reset_index()
        fig_city = px.bar(adopt_by_city, x="city_tier",
                          y="would_use_parakh_simulator",
                          color_discrete_sequence=[ORANGE],
                          title="Parakh Adoption by City Tier")
        fig_city.update_layout(yaxis_title="% Would Adopt")
        st.plotly_chart(fig_city, use_container_width=True)

    # Q7 — WTP
    st.subheader("Q7 · Willingness to Pay")
    fig_wtp = px.box(m1, x="self_reported_risk_appetite",
                     y="willingness_to_pay_monthly_inr",
                     color="self_reported_risk_appetite",
                     color_discrete_sequence=[GREEN,TEAL,ORANGE],
                     title="WTP (₹/month) by Risk Appetite")
    fig_wtp.update_layout(showlegend=False,
                          yaxis_title="₹/month",
                          xaxis_title="Risk Appetite")
    st.plotly_chart(fig_wtp, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 3 — DIAGNOSTIC ANALYTICS
# ════════════════════════════════════════════════════════════
elif page == "🔬 Diagnostic Analytics":
    st.title("🔬 Diagnostic Analytics")
    st.caption("Root-cause analysis: what drives loss and concentration risk?")

    # ── Loss drivers ──────────────────────────────────────────
    st.subheader("Loss Rate Decomposition")

    tab1, tab2, tab3 = st.tabs(["By Asset Class","By Income","By Experience"])

    with tab1:
        lr_asset = (
            m2.groupby("asset_class")["loss_occurred"].mean() * 100
        ).reset_index()
        lr_asset.columns = ["Asset Class","Loss Rate (%)"]
        lr_asset = lr_asset.sort_values("Loss Rate (%)", ascending=False)
        fig = px.bar(lr_asset, x="Asset Class", y="Loss Rate (%)",
                     color="Loss Rate (%)",
                     color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],
                     title="Loss Rate by Asset Class")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        lr_inc = (
            m2.groupby("income_bracket")["loss_occurred"].mean() * 100
        ).reindex(INCOME_ORDER).reset_index()
        lr_inc.columns = ["Income Bracket","Loss Rate (%)"]
        fig2 = px.bar(lr_inc, x="Income Bracket", y="Loss Rate (%)",
                      color_discrete_sequence=[PURPLE],
                      title="Loss Rate by Income Bracket")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        m2["exp_bin"] = pd.cut(m2["years_investing"],
                                bins=[0,2,5,10,25],
                                labels=["0-2 yrs","2-5 yrs","5-10 yrs","10+ yrs"])
        lr_exp = (
            m2.groupby("exp_bin", observed=True)["loss_occurred"].mean() * 100
        ).reset_index()
        lr_exp.columns = ["Experience","Loss Rate (%)"]
        fig3 = px.bar(lr_exp, x="Experience", y="Loss Rate (%)",
                      color_discrete_sequence=[ORANGE],
                      title="Loss Rate by Investing Experience")
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── Concentration risk anatomy ────────────────────────────
    st.subheader("Concentration Risk Anatomy")
    c1, c2 = st.columns(2)
    with c1:
        fig_sc = px.scatter(
            m1, x="herfindahl_index", y="total_loss_last_18mo_inr",
            color="high_concentration_flag",
            color_discrete_map={0:GREEN, 1:RED},
            opacity=0.6,
            labels={"high_concentration_flag":"High Conc.", 
                    "herfindahl_index":"HHI",
                    "total_loss_last_18mo_inr":"Loss (INR)"},
            title="HHI vs. Loss Incurred"
        )
        fig_sc.update_yaxes(type="log")
        st.plotly_chart(fig_sc, use_container_width=True)
    with c2:
        conc_by_city = (
            m1.groupby("city_tier")["high_concentration_flag"].mean() * 100
        ).reset_index()
        conc_by_city.columns = ["City Tier","% High Concentration"]
        fig_conc = px.bar(conc_by_city, x="City Tier",
                          y="% High Concentration",
                          color_discrete_sequence=[RED],
                          title="Concentration Risk by City Tier")
        st.plotly_chart(fig_conc, use_container_width=True)

    st.divider()

    # ── Financial literacy vs risk awareness ──────────────────
    st.subheader("Literacy × Awareness vs. Outcomes")
    fig_bubble = px.scatter(
        m1.sample(min(300, len(m1)), random_state=1),
        x="financial_literacy_score",
        y="risk_awareness_score_out_of_10",
        size="total_loss_last_18mo_inr",
        color="would_use_parakh_simulator",
        color_discrete_map={0:RED, 1:TEAL},
        opacity=0.7,
        labels={"would_use_parakh_simulator":"Would Use Parakh"},
        title="Financial Literacy vs Risk Awareness (bubble = loss amount)"
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Market event impact ───────────────────────────────────
    st.subheader("Market Event Impact on Losses")
    me_tab = m2.groupby("market_event_flag")["loss_occurred"].mean().reset_index()
    me_tab["market_event_flag"] = me_tab["market_event_flag"].map(
        {0:"No Event", 1:"Market Event"}
    )
    me_tab.columns = ["Condition","Loss Rate"]
    fig_me = px.bar(me_tab, x="Condition", y="Loss Rate",
                    color_discrete_sequence=[TEAL, RED],
                    title="Loss Rate: Market Event vs Normal")
    fig_me.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_me, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 4 — ML CLASSIFICATION
# ════════════════════════════════════════════════════════════
elif page == "🤖 ML Classification":
    st.title("🤖 ML Classification")

    model_choice = st.radio(
        "Select Classification Target",
        ["Model A — Trade Loss Prediction (loss_occurred)",
         "Model B — Parakh Adoption Prediction (would_use_parakh_simulator)"],
        horizontal=True
    )

    if "Model A" in model_choice:
        X, y, feats = prep_model_a(m2)
        label = "Model A (Trade-Level)"
    else:
        X, y, feats = prep_model_b(m1)
        label = "Model B (Investor-Level)"

    results = train_all_models(X, tuple(y), label=label)

    # ── Performance summary table ─────────────────────────────
    st.subheader(f"Performance Summary — {label}")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy":  f"{r['acc']*100:.1f}%",
            "Precision": f"{r['prec']*100:.1f}%",
            "Recall":    f"{r['rec']*100:.1f}%",
            "F1":        f"{r['f1']*100:.1f}%",
            "ROC-AUC":   f"{r['auc']:.3f}",
            "False+ (%)":f"{r['fp_pct']:.1f}%",
            "False- (%)":f"{r['fn_pct']:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.divider()

    # ── ROC curves ────────────────────────────────────────────
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash="dash", color="grey"))
    for name, r in results.items():
        fig_roc.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"],
            name=f"{name} (AUC={r['auc']:.3f})",
            line=dict(color=MODEL_COLORS[name], width=2.5)
        ))
    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.divider()

    # ── Confusion matrices ────────────────────────────────────
    st.subheader("Confusion Matrices")
    cols = st.columns(4)
    for (name, r), col in zip(results.items(), cols):
        cm = r["cm"]
        if cm.size == 4:
            labels = ["No Loss","Loss"] if "Model A" in model_choice else ["No Adopt","Adopt"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=labels, y=labels,
                color_continuous_scale=[[0,"#EAF4FB"],[1,MODEL_COLORS[name]]],
                title=name
            )
            fig_cm.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                coloraxis_showscale=False,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            col.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── Feature importance ────────────────────────────────────
    st.subheader("Feature Importance (Tree Models)")
    fi_models = {n: r for n, r in results.items() if r["feat_imp"] is not None}
    if fi_models:
        fi_tabs = st.tabs(list(fi_models.keys()))
        for tab, (name, r) in zip(fi_tabs, fi_models.items()):
            with tab:
                fi_df = pd.DataFrame(
                    {"Feature": feats, "Importance": r["feat_imp"]}
                ).sort_values("Importance", ascending=False).head(15)
                fig_fi = px.bar(fi_df, x="Importance", y="Feature",
                                orientation="h",
                                color_discrete_sequence=[MODEL_COLORS[name]],
                                title=f"Top-15 Features — {name}")
                fig_fi.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance not available for KNN.")


# ════════════════════════════════════════════════════════════
#  PAGE 5 — BUSINESS INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.title("💡 Business Insights")

    # ── Parakh Need Score segmentation ───────────────────────
    st.subheader("Investor Segments by Parakh Need Score")
    m1["need_segment"] = pd.cut(
        m1["parakh_need_score"],
        bins=[0, 3, 6, 10],
        labels=["Low Need","Medium Need","High Need"]
    )
    seg_data = m1["need_segment"].value_counts().reset_index()
    seg_data.columns = ["Segment","Count"]
    c1, c2 = st.columns(2)
    with c1:
        fig_seg = px.pie(seg_data, names="Segment", values="Count",
                         color_discrete_sequence=[GREEN, GOLD, RED],
                         title="Investor Need Segments")
        st.plotly_chart(fig_seg, use_container_width=True)
    with c2:
        wtp_by_seg = m1.groupby("need_segment", observed=True)[
            "willingness_to_pay_monthly_inr"
        ].median().reset_index()
        wtp_by_seg.columns = ["Segment","Median WTP (₹/mo)"]
        fig_wtp2 = px.bar(wtp_by_seg, x="Segment", y="Median WTP (₹/mo)",
                          color="Segment",
                          color_discrete_sequence=[GREEN, GOLD, RED],
                          title="Median WTP by Need Segment")
        fig_wtp2.update_layout(showlegend=False)
        st.plotly_chart(fig_wtp2, use_container_width=True)

    st.divider()

    # ── Revenue potential ─────────────────────────────────────
    st.subheader("Revenue Potential Model")
    total_addressable = len(m1)
    adopters = m1[m1["would_use_parakh_simulator"] == 1]
    avg_wtp  = adopters["willingness_to_pay_monthly_inr"].mean()
    monthly_rev = len(adopters) * avg_wtp

    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Potential Adopters (sample)", f"{len(adopters):,}",
               f"{len(adopters)/total_addressable*100:.1f}% of cohort")
    ci2.metric("Avg Monthly WTP", f"₹{avg_wtp:,.0f}")
    ci3.metric("Monthly Revenue (sample)", f"₹{monthly_rev:,.0f}")

    st.info(
        "**Scaling to India's ~90M retail investors**: If even 5% mirror our cohort's "
        f"adoption rate ({len(adopters)/total_addressable*100:.0f}%) and pay "
        f"₹{avg_wtp:.0f}/month, monthly ARR potential exceeds "
        f"₹{(0.05 * 90_000_000 * (len(adopters)/total_addressable) * avg_wtp / 1e7):.0f} Cr."
    )

    st.divider()

    # ── Pain points ───────────────────────────────────────────
    st.subheader("Top Pain Points — Why Parakh is Needed")
    pain_cnt = m1["main_pain_point"].value_counts().reset_index()
    pain_cnt.columns = ["Pain Point","Count"]
    fig_pain = px.bar(pain_cnt, x="Count", y="Pain Point",
                      orientation="h",
                      color="Count",
                      color_continuous_scale=[[0,TEAL],[1,RED]],
                      title="Distribution of Primary Pain Points")
    fig_pain.update_layout(yaxis=dict(autorange="reversed"),
                           coloraxis_showscale=False)
    st.plotly_chart(fig_pain, use_container_width=True)

    st.divider()

    # ── AA ecosystem ──────────────────────────────────────────
    st.subheader("Account Aggregator Ecosystem Readiness")
    c1, c2 = st.columns(2)
    with c1:
        aa_data = pd.DataFrame({
            "Status": ["Heard of AA","Has AA Consent","Not Heard"],
            "Count": [
                m1["heard_of_account_aggregator"].sum(),
                m1["has_aa_consent"].sum(),
                (~m1["heard_of_account_aggregator"].astype(bool)).sum()
            ]
        })
        fig_aa = px.funnel(aa_data, x="Count", y="Status",
                           color_discrete_sequence=[TEAL],
                           title="AA Awareness → Consent Funnel")
        st.plotly_chart(fig_aa, use_container_width=True)
    with c2:
        adopt_aa = m1.groupby("has_aa_consent")[
            "would_use_parakh_simulator"
        ].mean().reset_index()
        adopt_aa["has_aa_consent"] = adopt_aa["has_aa_consent"].map(
            {0:"No AA Consent", 1:"Has AA Consent"}
        )
        adopt_aa.columns = ["AA Status","Parakh Adoption Rate"]
        fig_aa2 = px.bar(adopt_aa, x="AA Status",
                         y="Parakh Adoption Rate",
                         color_discrete_sequence=[TEAL, GREEN],
                         title="Parakh Adoption: AA Consent vs Not")
        fig_aa2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_aa2, use_container_width=True)

    st.divider()

    # ── Strategic recommendations ─────────────────────────────
    st.subheader("Strategic Recommendations")
    st.markdown("""
| Priority | Insight | Recommendation |
|:---:|---|---|
| 🔴 | ~68% investors face concentration risk (HHI > 0.20) | Launch HHI Exposure Report as free-tier hook |
| 🔴 | Gut/FOMO trades have 55-64% loss rate | Pre-trade simulator with loss probability score |
| 🟡 | <10% check risk before trading | In-app nudges + AA consent onboarding |
| 🟡 | ~70% MF overlap in top-15 stocks | Overlap heat-map as signature Parakh feature |
| 🟢 | Tier 2/3 cities show strong adoption potential | Vernacular mobile-first UX |
| 🟢 | AA consent correlates with higher Parakh adoption | Partner with NSDL/CDSL for easy AA sign-up |
""")

    st.divider()
    st.markdown(
        "**Parakh** is built on the premise that informed investors make better decisions. "
        "The data shows a clear gap between *perceived* and *actual* risk — "
        "bridging it is the product opportunity."
    )
    st.caption("Parakh Dashboard · Data Analytics PBL · SP Jain Singapore")

