"""
app.py — Content Portfolio Strategy Dashboard

Reads data/catalog_history.csv (grown daily by the GitHub Actions workflow / fetch_new_titles.py)
and displays a lightweight overview immediately. The full analysis (hypothesis tests, modeling,
clustering, economics, churn link, memo) is compute-heavier, so it only runs when you click
"Run Full Analysis" — not on every page load.
"""
import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import simulate_bootstrap_catalog, run_full_pipeline

sns.set_style("whitegrid")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "catalog_history.csv")
META_PATH = os.path.join(os.path.dirname(__file__), "data", "last_pull.json")

st.set_page_config(page_title="Content Portfolio Strategy Dashboard", layout="wide")


@st.cache_data(ttl=3600)
def load_catalog():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            df = simulate_bootstrap_catalog(n=400, seed=42)
        return df
    return simulate_bootstrap_catalog(n=400, seed=42)


def load_pull_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return None


st.title("🎬 Content Portfolio Strategy Dashboard")
st.caption("Renew, cancel, or invest more? A data-driven greenlighting framework, refreshed daily with real TMDB releases.")

catalog_meta = load_catalog()
pull_meta = load_pull_metadata()

# ---------------- Sidebar: data freshness + manual pull ----------------
with st.sidebar:
    st.header("Data status")
    n_real = int((catalog_meta.get('source', pd.Series(dtype=str)) == 'real_tmdb_pull').sum()) if 'source' in catalog_meta else 0
    n_sim = len(catalog_meta) - n_real
    st.metric("Total titles", len(catalog_meta))
    st.metric("Real TMDB titles", n_real)
    st.metric("Simulated bootstrap titles", n_sim)

    if pull_meta:
        st.caption(f"Last automated pull: {pull_meta.get('last_pull_date', 'unknown')}")
        st.caption(f"Titles added in that pull: {pull_meta.get('n_titles_added_this_pull', 0)}")
    else:
        st.caption("No automated pull has run yet — showing bootstrap data only. "
                   "The daily GitHub Actions workflow will populate real titles over time.")

    st.divider()
    st.subheader("Manual pull (testing)")
    st.caption("The scheduled daily pull happens via GitHub Actions, not this app. Use this only "
               "to test a pull without waiting for the schedule.")
    manual_key = st.text_input("TMDB API key", type="password", help="Only used for this manual test pull; not stored.")
    if st.button("Pull latest titles now"):
        if not manual_key:
            st.error("Enter a TMDB API key first.")
        else:
            import fetch_new_titles
            os.environ["TMDB_API_KEY"] = manual_key
            with st.spinner("Pulling from TMDB..."):
                fetch_new_titles.main()
            st.cache_data.clear()
            st.success("Pull complete — reloading data.")
            st.rerun()

# ---------------- Lightweight overview (always shown, no heavy compute) ----------------
st.subheader("Catalog overview")
col1, col2, col3 = st.columns(3)
col1.metric("Titles in catalog", len(catalog_meta))
col2.metric("Genres represented", catalog_meta['genre'].nunique())
col3.metric("Median budget ($M)", round(catalog_meta['production_budget_musd'].median(), 1))

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    catalog_meta['genre'].value_counts().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Titles by genre')
    st.pyplot(fig)
with c2:
    if 'pulled_date' in catalog_meta.columns and catalog_meta['pulled_date'].notna().any():
        by_date = catalog_meta.dropna(subset=['pulled_date']).groupby('pulled_date').size().cumsum()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        by_date.plot(ax=ax, marker='o', color='seagreen')
        ax.set_title('Cumulative real titles pulled over time')
        ax.set_ylabel('Total real titles')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("Catalog growth chart will appear once at least one real TMDB pull has run.")

st.divider()

# ---------------- Full analysis (on-demand only) ----------------
st.subheader("Full analysis")
st.caption("Hypothesis tests, predictive modeling, clustering, economics, and the content-churn "
           "link. Runs on whatever's currently in the catalog above — click to (re)run after new "
           "titles have been added.")

if 'pipeline_result' not in st.session_state:
    st.session_state.pipeline_result = None

run_clicked = st.button("▶ Run Full Analysis", type="primary")
if run_clicked:
    with st.spinner("Running hypothesis tests, modeling, clustering, and economics..."):
        st.session_state.pipeline_result = run_full_pipeline(catalog_meta, pulled_metadata=pull_meta)
    st.success("Analysis complete.")

result = st.session_state.pipeline_result

if result is None:
    st.info("Click **Run Full Analysis** above to generate hypothesis tests, model results, "
            "archetypes, and the renewal recommendations.")
else:
    tabs = st.tabs(["Hypothesis Tests", "Predictive Model", "Archetypes", "Economics & Recommendations",
                    "Content ↔ Churn", "Executive Memo"])

    with tabs[0]:
        st.dataframe(result['hypothesis_results'], use_container_width=True)
        st.caption(f"Cohen's d for star power effect: {result['cohens_d']:.3f}" if not np.isnan(result['cohens_d']) else "")
        st.caption("Benjamini-Hochberg correction applied across all tests — check `significant_after_bh` "
                   "rather than raw p-values.")

    with tabs[1]:
        st.dataframe(result['model_results'], use_container_width=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        result['feature_importance'].set_index('feature')['importance_mean'].sort_values().plot(
            kind='barh', ax=ax, color='darkorange')
        ax.set_title(f"Permutation importance ({result['best_model_name']})")
        st.pyplot(fig)

    with tabs[2]:
        arch_counts = result['catalog']['archetype'].value_counts()
        st.bar_chart(arch_counts)
        st.dataframe(result['cluster_profile'], use_container_width=True)

    with tabs[3]:
        st.caption(f"Auto-calibrated churn-reduction constant: {result['calibrated_constant']:.6f} "
                   "(anchored so the median title breaks even)")
        rec_counts = result['catalog']['recommendation'].value_counts().reindex(
            ['Renew / Invest More', 'Renew (Monitor)', 'Renegotiate Terms', 'Cancel / Do Not Renew'])
        st.bar_chart(rec_counts)

        st.markdown("**Sensitivity: % of titles recommended for renewal**")
        pivot = result['sensitivity'].pivot(index='ltv_multiplier', columns='cost_multiplier',
                                             values='pct_titles_recommended_renew')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, cbar_kws={'label': '% -> renew'})
        st.pyplot(fig)

        st.markdown("**Top opportunities**")
        st.dataframe(result['top_titles'], use_container_width=True)
        st.markdown("**Titles to sunset**")
        st.dataframe(result['bottom_titles'], use_container_width=True)

    with tabs[4]:
        st.dataframe(result['churn_coef'], use_container_width=True)
        st.caption(f"Churn model AUC: {result['churn_auc']:.3f}")
        st.markdown("**A/B test power analysis (churn as guardrail metric)**")
        power_df = pd.DataFrame([
            {"MDE (pp)": mde * 100, "Required n per arm": n}
            for mde, n in result['power_analysis'].items()
        ])
        st.dataframe(power_df, use_container_width=True)

    with tabs[5]:
        st.text(result['memo'])
        st.download_button("Download memo as .txt", result['memo'], file_name="executive_memo.txt")

st.divider()
st.caption("Engagement, subscriber, and churn data are simulated using realistic benchmark ranges "
           "— no public API publishes real streaming viewership. Pre-release metadata (genre, "
           "budget, cast popularity, critic score) for TMDB-sourced titles is real.")
