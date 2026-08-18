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

from pipeline import simulate_bootstrap_catalog, run_full_pipeline, ensure_columns

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
        return ensure_columns(df)
    return ensure_columns(simulate_bootstrap_catalog(n=400, seed=42))


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
        st.caption(f"Revenue backfilled in that pull: {pull_meta.get('n_titles_revenue_refreshed_this_pull', 0)}")
    else:
        st.caption("No automated pull has run yet — showing bootstrap data only. "
                   "The daily GitHub Actions workflow will populate real titles over time.")

    st.divider()
    st.subheader("TMDB API key")
    st.caption("Used for the manual pull below and the title search feature. Not stored anywhere "
               "persistent — only kept for this browser session.")
    api_key_input = st.text_input("TMDB API key", type="password", value=st.session_state.get('tmdb_key', ''))
    st.session_state.tmdb_key = api_key_input

    st.divider()
    st.subheader("Manual pull (testing)")
    st.caption("The scheduled daily pull happens via GitHub Actions, not this app. Use this only "
               "to test a pull without waiting for the schedule.")
    if st.button("Pull latest titles now"):
        if not st.session_state.tmdb_key:
            st.error("Enter a TMDB API key above first.")
        else:
            import fetch_new_titles
            os.environ["TMDB_API_KEY"] = st.session_state.tmdb_key
            with st.spinner("Pulling from TMDB..."):
                fetch_new_titles.main()
            st.cache_data.clear()
            st.success("Pull complete — reloading data.")
            st.rerun()

# ---------------- Search a specific title ----------------
st.subheader("🔍 Look up a specific title")
st.caption("Search any movie on TMDB and see what this framework recommends for it — evaluated "
           "in the context of your current catalog, so the same calibration and archetypes apply.")

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'spotlight_result' not in st.session_state:
    st.session_state.spotlight_result = None
if 'spotlight_title_id' not in st.session_state:
    st.session_state.spotlight_title_id = None

sq_col1, sq_col2 = st.columns([4, 1])
with sq_col1:
    search_query = st.text_input("Search", placeholder="e.g. Spider-Man: Brand New Day",
                                  label_visibility="collapsed", key="movie_search_box")
with sq_col2:
    do_search = st.button("Search", key="search_btn", width='stretch')

if do_search:
    if not st.session_state.tmdb_key:
        st.error("Enter your TMDB API key in the sidebar first.")
    elif not search_query.strip():
        st.error("Type a movie title to search.")
    else:
        import tmdb_client
        with st.spinner(f"Searching TMDB for '{search_query}'..."):
            try:
                st.session_state.search_results = tmdb_client.search_movies(st.session_state.tmdb_key, search_query)
                st.session_state.spotlight_result = None  # clear any previous spotlight
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.search_results = None

if st.session_state.search_results:
    if len(st.session_state.search_results) == 0:
        st.info("No matches found on TMDB for that title.")
    else:
        options = [f"{r['title']} ({r['year']})" if r['year'] else r['title']
                   for r in st.session_state.search_results]
        chosen_idx = st.radio("Select the correct title:", options=list(range(len(options))),
                               format_func=lambda i: options[i], key="search_choice_radio")
        chosen = st.session_state.search_results[chosen_idx]

        pcol, dcol = st.columns([1, 3])
        with pcol:
            if chosen.get('poster_path'):
                st.image(f"https://image.tmdb.org/t/p/w200{chosen['poster_path']}", width='stretch')
        with dcol:
            st.write(chosen.get('overview', '')[:300])
            if st.button("Analyze this title", type="primary"):
                import tmdb_client
                with st.spinner("Fetching details and running the analysis..."):
                    try:
                        row = tmdb_client.fetch_movie_row(st.session_state.tmdb_key, chosen['id'])
                    except Exception as e:
                        row = None
                        st.error(f"Couldn't fetch details: {e}")
                if row is None:
                    st.warning("This title is missing budget or runtime data on TMDB, so the "
                               "economics layer can't evaluate it.")
                else:
                    # Always use freshly-fetched data for the searched title, even if an older
                    # (possibly stale) copy already exists in the tracked catalog.
                    merged = catalog_meta[catalog_meta['title_id'] != row['title_id']].copy()
                    merged = pd.concat([merged, pd.DataFrame([row])], ignore_index=True)
                    with st.spinner("Running hypothesis tests, modeling, clustering, and economics..."):
                        st.session_state.spotlight_result = run_full_pipeline(merged, pulled_metadata=pull_meta)
                    st.session_state.spotlight_title_id = row['title_id']

if st.session_state.spotlight_result is not None:
    sres = st.session_state.spotlight_result
    srow_df = sres['catalog'][sres['catalog']['title_id'] == st.session_state.spotlight_title_id]
    if not srow_df.empty:
        srow = srow_df.iloc[0]
        st.markdown("### Result")
        c1, c2 = st.columns([1, 2])
        with c1:
            if isinstance(srow.get('poster_path'), str) and srow['poster_path']:
                st.image(f"https://image.tmdb.org/t/p/w300{srow['poster_path']}", width='stretch')
        with c2:
            st.markdown(f"## {srow['tmdb_title']}")
            st.write(srow.get('overview', ''))
            badge_color = {'Renew / Invest More': '🟢', 'Renew (Monitor)': '🟡',
                           'Renegotiate Terms': '🟠', 'Cancel / Do Not Renew': '🔴',
                           'Insufficient Data': '⚪'}
            st.markdown(f"#### {badge_color.get(srow['recommendation'], '⚪')} {srow['recommendation']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Budget", f"${srow['production_budget_musd']:.0f}M")
            m2.metric("Real box office", f"${srow['real_revenue_musd']:.0f}M" if srow['real_revenue_musd'] > 0
                      else "Not yet known")
            m3.metric("Box office multiple", f"{srow['real_box_office_multiple']:.2f}x"
                      if pd.notna(srow['real_box_office_multiple']) else "N/A")

            m4, m5, m6 = st.columns(3)
            m4.metric("Simulated-only ROI", srow['roi_ratio'])
            m5.metric("Blended ROI (drives verdict)", srow['blended_roi_ratio'])
            m6.metric("Archetype", srow['archetype'])

        st.caption("Simulated engagement, subscriber, and churn figures are illustrative (no "
                   "public API has real streaming data); budget and box office revenue above are "
                   "real TMDB figures. This verdict reflects the current catalog's calibration, "
                   "so it may shift slightly as the catalog above grows.")
    else:
        st.warning("Couldn't find the analyzed title in the pipeline output — try re-running the analysis.")

st.divider()

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

# ---------------- Popular Titles (with poster art + details) ----------------
st.subheader("🔥 Popular titles")
st.caption("Real TMDB titles, sorted by TMDB popularity. Simulated bootstrap titles don't have "
           "real poster art or synopses, so this view only shows titles pulled from TMDB.")

real_titles = catalog_meta[catalog_meta.get('source', pd.Series(dtype=str)) == 'real_tmdb_pull'].copy()

if real_titles.empty:
    st.info("No real TMDB titles yet — this section will populate once the daily pull (or a "
            "manual pull from the sidebar) has run at least once.")
else:
    filt_col1, filt_col2, filt_col3 = st.columns([2, 2, 1])
    with filt_col1:
        genre_options = ["All"] + sorted(real_titles['genre'].dropna().unique().tolist())
        genre_filter = st.selectbox("Filter by genre", genre_options)
    with filt_col2:
        sort_option = st.selectbox("Sort by", ["TMDB popularity", "Critic score", "Budget", "Most recently added"])
    with filt_col3:
        n_show = st.number_input("Show", min_value=4, max_value=48, value=12, step=4)

    view = real_titles if genre_filter == "All" else real_titles[real_titles['genre'] == genre_filter]

    sort_map = {
        "TMDB popularity": ('tmdb_popularity', False),
        "Critic score": ('critic_score', False),
        "Budget": ('production_budget_musd', False),
        "Most recently added": ('pulled_date', False),
    }
    sort_col, ascending = sort_map[sort_option]
    view = view.sort_values(sort_col, ascending=ascending, na_position='last').head(int(n_show))

    # If the full analysis has already been run, pull each title's recommendation to show as a badge
    rec_lookup = {}
    if st.session_state.get('pipeline_result') is not None:
        rec_catalog = st.session_state.pipeline_result['catalog']
        rec_lookup = rec_catalog.set_index('title_id')['recommendation'].to_dict()

    badge_color = {
        'Renew / Invest More': '🟢', 'Renew (Monitor)': '🟡',
        'Renegotiate Terms': '🟠', 'Cancel / Do Not Renew': '🔴'
    }

    cols_per_row = 4
    rows_of_titles = [view.iloc[i:i + cols_per_row] for i in range(0, len(view), cols_per_row)]
    for row_chunk in rows_of_titles:
        cols = st.columns(cols_per_row)
        for col, (_, title) in zip(cols, row_chunk.iterrows()):
            with col:
                poster_path = title.get('poster_path')
                if isinstance(poster_path, str) and poster_path:
                    st.image(f"https://image.tmdb.org/t/p/w300{poster_path}", width='stretch')
                else:
                    st.markdown("🎬 *No poster available*")

                st.markdown(f"**{title.get('tmdb_title', title['title_id'])}**")
                st.caption(f"{title['genre']} · ${title['production_budget_musd']:.1f}M budget · "
                           f"Critic score {title['critic_score']:.0f}/100")
                if pd.notna(title.get('release_date_full')):
                    st.caption(f"Released {title['release_date_full']}")

                overview = title.get('overview', '')
                if isinstance(overview, str) and overview:
                    truncated = overview if len(overview) <= 140 else overview[:140].rsplit(' ', 1)[0] + "…"
                    st.write(truncated)

                rec = rec_lookup.get(title['title_id'])
                if rec:
                    st.markdown(f"{badge_color.get(rec, '⚪')} **{rec}**")

                tmdb_numeric_id = str(title['title_id']).replace('TMDB', '')
                st.markdown(f"[View on TMDB](https://www.themoviedb.org/movie/{tmdb_numeric_id})")

    if not rec_lookup:
        st.caption("Run the full analysis below to see each title's renew/cancel recommendation "
                   "as a badge on its card.")

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
        st.dataframe(result['hypothesis_results'], width='stretch')
        st.caption(f"Cohen's d for star power effect: {result['cohens_d']:.3f}" if not np.isnan(result['cohens_d']) else "")
        st.caption("Benjamini-Hochberg correction applied across all tests — check `significant_after_bh` "
                   "rather than raw p-values.")

    with tabs[1]:
        st.dataframe(result['model_results'], width='stretch')
        fig, ax = plt.subplots(figsize=(6, 4))
        result['feature_importance'].set_index('feature')['importance_mean'].sort_values().plot(
            kind='barh', ax=ax, color='darkorange')
        ax.set_title(f"Permutation importance ({result['best_model_name']})")
        st.pyplot(fig)

    with tabs[2]:
        arch_counts = result['catalog']['archetype'].value_counts()
        st.bar_chart(arch_counts)
        st.dataframe(result['cluster_profile'], width='stretch')

    with tabs[3]:
        st.caption(f"Auto-calibrated churn-reduction constant: {result['calibrated_constant']:.6f} "
                   "(anchored so the median title breaks even)")
        st.info("For titles that have already been released, TMDB's real box office revenue is "
                 "blended into the recommendation (weighted 70% real evidence / 30% simulated "
                 "streaming signal) rather than relying on the simulated engagement number alone. "
                 "This avoids nonsense results like a $2B blockbuster getting flagged 'Cancel' "
                 "purely because its genre's simulated decay rate is high. Titles with no release "
                 "revenue yet (upcoming/unreleased) rely on the simulated signal alone, since "
                 "that's genuinely all that's known about them.")
        rec_counts = result['catalog']['recommendation'].value_counts().reindex(
            ['Renew / Invest More', 'Renew (Monitor)', 'Renegotiate Terms', 'Cancel / Do Not Renew', 'Insufficient Data']
        ).dropna()
        st.bar_chart(rec_counts)

        st.markdown("**Sensitivity: % of titles recommended for renewal**")
        pivot = result['sensitivity'].pivot(index='ltv_multiplier', columns='cost_multiplier',
                                             values='pct_titles_recommended_renew')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, cbar_kws={'label': '% -> renew'})
        st.pyplot(fig)

        st.markdown("**Top opportunities**")
        st.dataframe(result['top_titles'], width='stretch')
        st.markdown("**Titles to sunset**")
        st.dataframe(result['bottom_titles'], width='stretch')

        with_real = result['catalog'][result['catalog']['real_box_office_multiple'].notna()]
        if not with_real.empty:
            st.markdown("**Titles where real box office overrode the simulated signal**")
            st.caption("roi_ratio = simulated-only · blended_roi_ratio = what actually drove the recommendation")
            st.dataframe(
                with_real[['title_id', 'tmdb_title', 'real_box_office_multiple', 'roi_ratio',
                           'blended_roi_ratio', 'recommendation']].sort_values('real_box_office_multiple', ascending=False),
                width='stretch'
            )

    with tabs[4]:
        st.dataframe(result['churn_coef'], width='stretch')
        st.caption(f"Churn model AUC: {result['churn_auc']:.3f}")
        st.markdown("**A/B test power analysis (churn as guardrail metric)**")
        power_df = pd.DataFrame([
            {"MDE (pp)": mde * 100, "Required n per arm": n}
            for mde, n in result['power_analysis'].items()
        ])
        st.dataframe(power_df, width='stretch')

    with tabs[5]:
        st.text(result['memo'])
        st.download_button("Download memo as .txt", result['memo'], file_name="executive_memo.txt")

st.divider()
st.caption("Engagement, subscriber, and churn data are simulated using realistic benchmark ranges "
           "— no public API publishes real streaming viewership. Pre-release metadata (genre, "
           "budget, cast popularity, critic score) for TMDB-sourced titles is real.")