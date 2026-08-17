"""
pipeline.py — core analysis functions for the content portfolio dashboard.

Design principle: engagement simulation is PER-TITLE DETERMINISTIC (seeded from a hash of
title_id), not drawn from one global random stream. This is what makes it safe to append new
titles to a growing historical dataset — recomputing engagement for the whole table never
changes previously-computed rows, and a title's simulated numbers are stable across runs.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

GENRES = ['Drama', 'Comedy', 'Action', 'Thriller', 'Documentary', 'Sci-Fi',
          'Romance', 'Horror', 'Animation', 'Reality']
CONTENT_TYPES = ['Original Series', 'Original Film', 'Licensed Series', 'Licensed Film']

GENRE_COMPLETION_BASE = {
    'Drama': 0.62, 'Comedy': 0.68, 'Action': 0.60, 'Thriller': 0.66,
    'Documentary': 0.48, 'Sci-Fi': 0.58, 'Romance': 0.63, 'Horror': 0.55,
    'Animation': 0.70, 'Reality': 0.52
}
DECAY_RATE_BASE = {
    'Drama': 0.55, 'Comedy': 0.60, 'Action': 0.72, 'Thriller': 0.75,
    'Documentary': 0.40, 'Sci-Fi': 0.65, 'Romance': 0.58, 'Horror': 0.78,
    'Animation': 0.35, 'Reality': 0.62
}
REWATCH_BASE = {
    'Drama': 0.10, 'Comedy': 0.18, 'Action': 0.08, 'Thriller': 0.06,
    'Documentary': 0.09, 'Sci-Fi': 0.12, 'Romance': 0.15, 'Horror': 0.05,
    'Animation': 0.28, 'Reality': 0.11
}

# TMDB genre id -> our analysis buckets (used by fetch_new_titles.py too)
TMDB_GENRE_ID_TO_BUCKET = {
    28: 'Action', 12: 'Action', 16: 'Animation', 35: 'Comedy', 80: 'Thriller',
    99: 'Documentary', 18: 'Drama', 10751: 'Comedy', 14: 'Sci-Fi', 36: 'Drama',
    27: 'Horror', 10402: 'Drama', 9648: 'Thriller', 10749: 'Romance',
    878: 'Sci-Fi', 10770: 'Drama', 53: 'Thriller', 10752: 'Drama', 37: 'Action'
}
TARGET_GENRES_TMDB_IDS = {
    'Action': 28, 'Comedy': 35, 'Drama': 18, 'Documentary': 99, 'Horror': 27,
    'Animation': 16, 'Romance': 10749, 'Sci-Fi': 878, 'Thriller': 53
}

# The full set of columns the app and pipeline expect to exist. data/catalog_history.csv is
# mutated externally by a daily cron job, so its schema can drift out from under the app code
# (e.g. if the CSV was last written by an older version of fetch_new_titles.py before a new
# column was added). ensure_columns() backfills anything missing with a safe default so neither
# app.py nor pipeline.py ever crashes on a stale file — the missing data just reads as "unknown"
# instead of taking the whole dashboard down.
EXPECTED_COLUMNS_DEFAULTS = {
    'title_id': '', 'tmdb_title': '', 'genre': 'Drama', 'content_type': 'Original Film',
    'runtime_min': 90.0, 'num_episodes': 1, 'production_budget_musd': 10.0,
    'lead_star_power': 50.0, 'critic_score': 60.0, 'release_month': 1,
    'release_dayofweek': 'Fri', 'is_series': 0, 'licensing_flag': 0,
    'tmdb_popularity': np.nan, 'overview': '', 'poster_path': None,
    'release_date_full': None, 'source': 'simulated_bootstrap', 'pulled_date': None,
    'real_revenue_musd': 0.0,
}


def ensure_columns(df):
    """Backfills any missing expected columns with safe defaults. Call this on any catalog
    DataFrame loaded from disk before using it, since the CSV's schema can lag behind the code."""
    df = df.copy()
    for col, default in EXPECTED_COLUMNS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df


# --------------------------------------------------------------------------
# Bootstrap / simulated catalog (used to seed the dashboard before any real
# TMDB pulls have happened, and as a fallback demo mode)
# --------------------------------------------------------------------------
def simulate_bootstrap_catalog(n=400, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'title_id': [f"SIM{i:04d}" for i in range(n)],
        'tmdb_title': [f"Simulated Title {i}" for i in range(n)],
        'genre': rng.choice(GENRES, n, p=[0.16, 0.13, 0.14, 0.12, 0.07, 0.09, 0.09, 0.08, 0.07, 0.05]),
        'content_type': rng.choice(CONTENT_TYPES, n, p=[0.35, 0.25, 0.25, 0.15]),
        'runtime_min': rng.normal(50, 25, n).clip(20, 180).round(),
        'num_episodes': np.where(rng.rand(n) < 0.55, rng.randint(4, 24, n), 1),
        'production_budget_musd': rng.lognormal(mean=2.5, sigma=1.0, size=n).clip(0.5, 250),
        'lead_star_power': rng.beta(2, 5, n) * 100,
        'critic_score': rng.normal(62, 18, n).clip(5, 100).round(1),
        'release_month': rng.randint(1, 13, n),
        'release_dayofweek': rng.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n,
                                         p=[0.08, 0.08, 0.1, 0.1, 0.34, 0.2, 0.1]),
        'tmdb_popularity': np.nan,
        'overview': '',
        'poster_path': None,
        'release_date_full': pd.NaT,
        'real_revenue_musd': 0.0,
        'source': 'simulated_bootstrap',
        'pulled_date': pd.NaT,
    })
    df['is_series'] = (df['num_episodes'] > 1).astype(int)
    df['licensing_flag'] = df['content_type'].str.startswith('Licensed').astype(int)
    return df


# --------------------------------------------------------------------------
# Engagement simulation — PER-ROW DETERMINISTIC, safe to append incrementally
# --------------------------------------------------------------------------
def _row_seed(title_id):
    """Stable 32-bit seed derived from title_id, independent of row order or table size."""
    h = hashlib.sha256(str(title_id).encode()).hexdigest()
    return int(h[:8], 16)


def _simulate_engagement_row(row):
    rs = np.random.RandomState(_row_seed(row['title_id']))

    genre_base = GENRE_COMPLETION_BASE.get(row['genre'], 0.55)
    star_effect = (row['lead_star_power'] / 100) * 0.12
    critic_effect = ((row['critic_score'] - 62) / 100) * 0.10
    runtime_penalty = -(max(row['runtime_min'] - 50, 0) / 500)
    completion_rate = np.clip(genre_base + star_effect + critic_effect + runtime_penalty
                               + rs.normal(0, 0.06), 0.05, 0.98)

    base_view = 0.5 + (row['production_budget_musd'] ** 0.4) * 0.4
    star_boost = (row['lead_star_power'] / 100) * 3.0
    weekend_boost = 0.4 if row['release_dayofweek'] in ('Fri', 'Sat') else 0.0
    week1 = (base_view + star_boost + weekend_boost) * rs.lognormal(0, 0.35)

    decay_base = DECAY_RATE_BASE.get(row['genre'], 0.6)
    decay_rate = np.clip(decay_base + rs.normal(0, 0.05), 0.15, 0.95)
    cum12wk = week1 * (1 / (decay_rate + 0.15)) * rs.normal(1, 0.1)

    base_days = 14 - (completion_rate * 8) - (row['is_series'] * 2)
    median_days = np.clip(base_days + rs.normal(0, 1.5), 1, 30)

    rewatch_base = REWATCH_BASE.get(row['genre'], 0.1)
    rewatch = np.clip(rewatch_base + rs.normal(0, 0.03), 0.01, 0.5)

    return pd.Series({
        'completion_rate': round(completion_rate, 3),
        'week1_viewing_hours_musd_equiv': round(week1, 2),
        'decay_rate': round(decay_rate, 3),
        'cum12wk_viewing_hours_musd_equiv': round(cum12wk, 2),
        'median_days_to_complete': round(median_days, 1),
        'rewatch_rate': round(rewatch, 3),
    })


def simulate_engagement(catalog_meta):
    """Adds engagement columns. Safe to call on the full history each time — every title's
    engagement values are a pure function of its own title_id + metadata, never of table order
    or which other rows are present."""
    eng = catalog_meta.apply(_simulate_engagement_row, axis=1)
    return pd.concat([catalog_meta.reset_index(drop=True), eng.reset_index(drop=True)], axis=1)


# --------------------------------------------------------------------------
# Hypothesis testing
# --------------------------------------------------------------------------
def run_hypothesis_tests(catalog):
    results = []
    groups = [catalog.loc[catalog['genre'] == g, 'completion_rate'] for g in catalog['genre'].unique()
              if (catalog['genre'] == g).sum() >= 2]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        results.append(('H1: Completion rate differs by genre (ANOVA)', f_stat, p_val))

    q1, q3 = catalog['lead_star_power'].quantile([0.33, 0.67])
    low = catalog.loc[catalog['lead_star_power'] <= q1, 'week1_viewing_hours_musd_equiv']
    high = catalog.loc[catalog['lead_star_power'] >= q3, 'week1_viewing_hours_musd_equiv']
    cohens_d = np.nan
    if len(low) > 1 and len(high) > 1:
        t_stat, p_val2 = stats.ttest_ind(high, low, equal_var=False)
        cohens_d = (high.mean() - low.mean()) / np.sqrt((high.std() ** 2 + low.std() ** 2) / 2)
        results.append(('H2: High star power -> higher week-1 viewership (Welch t-test)', t_stat, p_val2))

    weekend = catalog.loc[catalog['release_dayofweek'].isin(['Fri', 'Sat']), 'week1_viewing_hours_musd_equiv']
    weekday = catalog.loc[~catalog['release_dayofweek'].isin(['Fri', 'Sat']), 'week1_viewing_hours_musd_equiv']
    if len(weekend) > 1 and len(weekday) > 1:
        t_stat3, p_val3 = stats.ttest_ind(weekend, weekday, equal_var=False)
        results.append(('H3: Weekend release -> higher week-1 viewership (Welch t-test)', t_stat3, p_val3))

    r, p_val4 = stats.pearsonr(catalog['week1_viewing_hours_musd_equiv'], catalog['cum12wk_viewing_hours_musd_equiv'])
    results.append(('H4: Week-1 viewership correlates with 12wk cumulative (Pearson r)', r, p_val4))

    orig = catalog.loc[catalog['licensing_flag'] == 0, 'completion_rate']
    lic = catalog.loc[catalog['licensing_flag'] == 1, 'completion_rate']
    if len(orig) > 1 and len(lic) > 1:
        t5, p5 = stats.ttest_ind(orig, lic, equal_var=False)
        results.append(('H5: Original vs licensed content -> completion rate (Welch t-test)', t5, p5))

    res_df = pd.DataFrame(results, columns=['hypothesis', 'statistic', 'p_value'])
    res_df = res_df.sort_values('p_value').reset_index(drop=True)
    m = len(res_df)
    res_df['bh_threshold'] = [(i + 1) / m * 0.05 for i in range(m)]
    res_df['significant_after_bh'] = res_df['p_value'] <= res_df['bh_threshold']
    return res_df, cohens_d


# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------
def run_modeling(catalog):
    feature_cols_num = ['production_budget_musd', 'lead_star_power', 'runtime_min', 'num_episodes', 'critic_score']
    feature_cols_cat = ['genre', 'content_type', 'release_dayofweek']
    target_col = 'cum12wk_viewing_hours_musd_equiv'

    X = catalog[feature_cols_num + feature_cols_cat]
    y = catalog[target_col]

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', feature_cols_num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols_cat)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
    }
    model_results, fitted_pipelines = [], {}
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=min(5, max(2, len(X_train) // 20)), scoring='r2')
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        model_results.append((name, cv_scores.mean(), r2_score(y_test, preds), mean_absolute_error(y_test, preds)))
        fitted_pipelines[name] = pipe

    model_res_df = pd.DataFrame(model_results, columns=['model', 'cv_r2_mean', 'test_r2', 'test_mae'])
    best_name = model_res_df.sort_values('test_r2', ascending=False).iloc[0]['model']
    best_pipe = fitted_pipelines[best_name]

    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=15, random_state=42, scoring='r2')
    imp_df = pd.DataFrame({
        'feature': X_test.columns, 'importance_mean': perm.importances_mean, 'importance_std': perm.importances_std
    }).sort_values('importance_mean', ascending=False)

    catalog = catalog.copy()
    catalog['predicted_12wk_engagement'] = best_pipe.predict(X)
    return catalog, model_res_df, imp_df, best_name


# --------------------------------------------------------------------------
# Clustering / archetypes
# --------------------------------------------------------------------------
def run_clustering(catalog, n_clusters=4):
    catalog = catalog.copy()
    cluster_features = catalog[['completion_rate', 'week1_viewing_hours_musd_equiv',
                                 'decay_rate', 'rewatch_rate', 'median_days_to_complete']]
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    catalog['archetype_cluster'] = kmeans.fit_predict(cluster_scaled)

    profile = catalog.groupby('archetype_cluster')[
        ['completion_rate', 'week1_viewing_hours_musd_equiv', 'decay_rate', 'rewatch_rate', 'median_days_to_complete']
    ].mean()
    profile_ranked = profile.copy()
    profile_ranked['opening_strength'] = profile_ranked['week1_viewing_hours_musd_equiv'].rank()
    profile_ranked['longevity_score'] = (-profile_ranked['decay_rate'] + profile_ranked['rewatch_rate']).rank()

    def name_from_ranks(row, n):
        high_open = row['opening_strength'] > n / 2
        high_long = row['longevity_score'] > n / 2
        if high_open and high_long:
            return 'Evergreen Library Title'
        elif high_open and not high_long:
            return 'Splashy Opener, Fast Fade'
        elif not high_open and high_long:
            return 'Slow-Burn Sleeper'
        else:
            return 'Quiet Underperformer'

    archetype_names = profile_ranked.apply(lambda r: name_from_ranks(r, profile_ranked.shape[0]), axis=1)
    catalog['archetype'] = catalog['archetype_cluster'].map(archetype_names)
    return catalog, profile


# --------------------------------------------------------------------------
# Business economics (auto-calibrated — see notebook Part 6 for rationale)
# --------------------------------------------------------------------------
def run_economics(catalog, licensing_cost_frac=0.35, subscriber_ltv_usd=140,
                   subscriber_base=5_000_000, target_median_roi=1.0,
                   real_signal_weight=0.7, theatrical_breakeven_multiple=2.5):
    """
    real_signal_weight / theatrical_breakeven_multiple:
    For titles that have already released (TMDB reports real box office revenue), the simulated
    streaming-engagement ROI is blended with a signal derived from ACTUAL revenue, rather than
    trusting the synthetic engagement number alone. This matters because the synthetic engagement
    formula has no way to know a title actually performed well or badly — without this blend, a
    real $2B-grossing blockbuster on a $225M budget could get flagged "Cancel" purely because its
    genre's simulated decay rate is high, which is exactly backwards. Real evidence should dominate
    a synthetic proxy whenever it's available.

    # ASSUMPTION: a theatrical title typically needs ~2-2.5x its budget in box office revenue to
    # break even (covers marketing spend + theater revenue share). This converts real box office
    # into an ROI-like ratio comparable to our streaming roi_ratio (where 1.0 = breakeven), so the
    # two signals can be blended on the same scale.
    # ASSUMPTION: real_signal_weight=0.7 means known real performance gets 70% of the weight
    # against the simulated streaming signal for titles where revenue is actually known. Titles
    # with no revenue data yet (upcoming/unreleased) get 100% simulated, since that's genuinely
    # all we have for them.
    """
    catalog = catalog.copy()
    catalog['renewal_cost_musd'] = (catalog['production_budget_musd'] * licensing_cost_frac).round(2)

    median_engagement = catalog['cum12wk_viewing_hours_musd_equiv'].median()
    median_cost = catalog['renewal_cost_musd'].median()
    churn_reduction_per_unit = (target_median_roi * median_cost /
                                 (median_engagement * subscriber_base * subscriber_ltv_usd / 1_000_000))

    catalog['est_retained_subs'] = (catalog['cum12wk_viewing_hours_musd_equiv'] *
                                     churn_reduction_per_unit * subscriber_base).round(0)
    catalog['est_retention_value_musd'] = (catalog['est_retained_subs'] * subscriber_ltv_usd / 1_000_000).round(2)
    catalog['net_value_musd'] = (catalog['est_retention_value_musd'] - catalog['renewal_cost_musd']).round(2)
    catalog['roi_ratio'] = (catalog['est_retention_value_musd'] /
                             catalog['renewal_cost_musd'].replace(0, np.nan)).round(2)

    # --- Blend in real box office performance where it's known ---
    if 'real_revenue_musd' not in catalog.columns:
        catalog['real_revenue_musd'] = 0.0
    has_real_revenue = catalog['real_revenue_musd'].fillna(0) > 0
    catalog['real_box_office_multiple'] = np.where(
        has_real_revenue, catalog['real_revenue_musd'] / catalog['production_budget_musd'].replace(0, np.nan), np.nan
    ).round(2)
    real_roi_equivalent = catalog['real_box_office_multiple'] / theatrical_breakeven_multiple

    catalog['blended_roi_ratio'] = catalog['roi_ratio']
    catalog.loc[has_real_revenue, 'blended_roi_ratio'] = (
        real_signal_weight * real_roi_equivalent[has_real_revenue] +
        (1 - real_signal_weight) * catalog.loc[has_real_revenue, 'roi_ratio']
    ).round(2)

    def decision(row):
        r = row['blended_roi_ratio']
        if pd.isna(r):
            return 'Insufficient Data'
        elif r >= 1.5:
            return 'Renew / Invest More'
        elif r >= 0.8:
            return 'Renew (Monitor)'
        elif r >= 0.4:
            return 'Renegotiate Terms'
        else:
            return 'Cancel / Do Not Renew'

    catalog['recommendation'] = catalog.apply(decision, axis=1)

    # Dollar-consistent version of the blend: since ROI = value / cost and cost is the same
    # in both the simulated and real-evidence view, blending in ROI-ratio space (above) is
    # mathematically equivalent to blending the underlying dollar values. Deriving it this way
    # keeps the recommendation categories and the dollar figures below always in agreement.
    catalog['blended_est_retention_value_musd'] = (catalog['renewal_cost_musd'] * catalog['blended_roi_ratio']).round(2)
    catalog['blended_net_value_musd'] = (catalog['blended_est_retention_value_musd'] -
                                          catalog['renewal_cost_musd']).round(2)

    def realized_net_value(row):
        if row['recommendation'] in ('Cancel / Do Not Renew', 'Insufficient Data'):
            return 0.0
        elif row['recommendation'] == 'Renegotiate Terms':
            return round(row['blended_est_retention_value_musd'] - row['renewal_cost_musd'] * 0.7, 2)
        else:
            return row['blended_net_value_musd']
    catalog['realized_net_value_musd'] = catalog.apply(realized_net_value, axis=1)

    sensitivity_rows = []
    for ltv_mult in [0.8, 1.0, 1.2]:
        for cost_mult in [0.8, 1.0, 1.2]:
            adj_roi = (catalog['blended_est_retention_value_musd'] * ltv_mult) / (catalog['renewal_cost_musd'] * cost_mult).replace(0, np.nan)
            sensitivity_rows.append((ltv_mult, cost_mult, round((adj_roi >= 0.8).mean() * 100, 1)))
    sens_df = pd.DataFrame(sensitivity_rows, columns=['ltv_multiplier', 'cost_multiplier', 'pct_titles_recommended_renew'])

    return catalog, churn_reduction_per_unit, sens_df


# --------------------------------------------------------------------------
# Content -> churn link
# --------------------------------------------------------------------------
def simulate_subscribers_and_churn(catalog, n_subs=8000, seed=11):
    rng = np.random.RandomState(seed)
    sub_genre_pref = rng.choice(catalog['genre'].unique(), n_subs)
    supply = catalog.groupby('genre').size()
    genre_supply_score = (supply / supply.max()).to_dict()

    subs = pd.DataFrame({
        'sub_id': [f"S{i:05d}" for i in range(n_subs)],
        'preferred_genre': sub_genre_pref,
        'tenure_months': rng.gamma(3, 6, n_subs).clip(1, 96).round(),
        'avg_weekly_viewing_hrs': rng.gamma(2, 1.5, n_subs).clip(0, 30).round(2),
    })
    subs['genre_supply_score'] = subs['preferred_genre'].map(genre_supply_score)

    logit = (-1.2
             - 0.9 * subs['avg_weekly_viewing_hrs'] / subs['avg_weekly_viewing_hrs'].mean()
             - 1.1 * subs['genre_supply_score']
             - 0.06 * (subs['tenure_months'] < 3).astype(int)
             + rng.normal(0, 0.5, n_subs))
    churn_prob = 1 / (1 + np.exp(-logit))
    subs['churned'] = (rng.rand(n_subs) < churn_prob).astype(int)

    X_churn = subs[['tenure_months', 'avg_weekly_viewing_hrs', 'genre_supply_score']]
    y_churn = subs['churned']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_churn, y_churn, test_size=0.25, random_state=42, stratify=y_churn)

    churn_model = LogisticRegression(max_iter=1000)
    churn_model.fit(Xc_train, yc_train)
    auc = roc_auc_score(yc_test, churn_model.predict_proba(Xc_test)[:, 1])

    coef_df = pd.DataFrame({
        'feature': X_churn.columns, 'coefficient': churn_model.coef_[0],
        'odds_ratio': np.exp(churn_model.coef_[0])
    }).sort_values('odds_ratio')

    return subs, coef_df, auc


def power_analysis_sample_sizes(baseline_churn, mdes=(0.005, 0.01, 0.02), alpha=0.05, power=0.8):
    from scipy.stats import norm
    results = {}
    for mde in mdes:
        p1, p2 = baseline_churn, baseline_churn + mde
        z_alpha, z_beta = norm.ppf(1 - alpha / 2), norm.ppf(power)
        p_bar = (p1 + p2) / 2
        n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (mde ** 2)
        results[mde] = int(np.ceil(n))
    return results


# --------------------------------------------------------------------------
# Executive memo
# --------------------------------------------------------------------------
def build_executive_memo(catalog, coef_df, sens_df, pulled_metadata=None):
    total_titles = len(catalog)
    n_renew = catalog['recommendation'].isin(['Renew / Invest More', 'Renew (Monitor)']).sum()
    n_cancel = (catalog['recommendation'] == 'Cancel / Do Not Renew').sum()
    n_renegotiate = (catalog['recommendation'] == 'Renegotiate Terms').sum()
    total_net_value = catalog['realized_net_value_musd'].sum()
    naive_renew_all_value = catalog['blended_net_value_musd'].sum()

    memo_cols = ['title_id', 'genre', 'archetype', 'blended_net_value_musd', 'blended_roi_ratio',
                 'real_box_office_multiple', 'recommendation']
    top_titles = catalog.sort_values('blended_net_value_musd', ascending=False).head(5)[memo_cols].rename(
        columns={'blended_net_value_musd': 'net_value_if_renewed_musd', 'blended_roi_ratio': 'roi_ratio'})
    bottom_titles = catalog.sort_values('blended_net_value_musd', ascending=True).head(5)[memo_cols].rename(
        columns={'blended_net_value_musd': 'net_value_if_renewed_musd', 'blended_roi_ratio': 'roi_ratio'})

    freshness_line = ""
    if pulled_metadata is not None:
        freshness_line = (f"\nData freshness: catalog last updated {pulled_metadata.get('last_pull_date', 'unknown')}, "
                           f"{pulled_metadata.get('n_real_titles', 0)} real TMDB titles + "
                           f"{pulled_metadata.get('n_simulated_titles', 0)} simulated bootstrap titles.\n")

    memo = f"""
CONTENT PORTFOLIO REVIEW — EXECUTIVE SUMMARY
Prepared for: VP, Content Strategy
Scope: {total_titles} titles across {catalog['genre'].nunique()} genres
{freshness_line}
HEADLINE
Of {total_titles} titles reviewed, {n_renew} are recommended for renewal/investment,
{n_renegotiate} for renegotiated terms, and {n_cancel} for non-renewal.
Following these recommendations nets an estimated ${total_net_value:,.1f}M in forward portfolio
value, versus ${naive_renew_all_value:,.1f}M if every title were renewed as-is.

TOP OPPORTUNITIES
{top_titles.to_string(index=False)}

TITLES TO SUNSET
{bottom_titles.to_string(index=False)}

CONTENT-CHURN LINK
{coef_df.to_string(index=False)}
Lower genre_supply_score (a content gap in a subscriber's preferred genre) is associated with
materially higher churn odds, independent of tenure and overall viewing level.

CAVEATS
Engagement and subscriber data are simulated using realistic benchmark ranges because real
platform-level viewership data is proprietary. Cost/LTV assumptions are illustrative but the
churn-reduction constant is auto-calibrated to this catalog's own budget scale. The renewal
recommendation share moves from ~{sens_df['pct_titles_recommended_renew'].min()}% to
~{sens_df['pct_titles_recommended_renew'].max()}% of titles across a plausible ±20% assumption
range — see the sensitivity chart before treating any single number as precise.
"""
    return memo, top_titles, bottom_titles


def run_full_pipeline(catalog_meta, pulled_metadata=None):
    """Orchestrates the entire analysis on whatever catalog (bootstrap + real pulls) is passed in."""
    catalog = simulate_engagement(catalog_meta)
    hyp_df, cohens_d = run_hypothesis_tests(catalog)
    catalog, model_res_df, imp_df, best_name = run_modeling(catalog)
    catalog, cluster_profile = run_clustering(catalog)
    catalog, calibrated_constant, sens_df = run_economics(catalog)
    subs, coef_df, churn_auc = simulate_subscribers_and_churn(catalog)
    power_results = power_analysis_sample_sizes(subs['churned'].mean())
    memo, top_titles, bottom_titles = build_executive_memo(catalog, coef_df, sens_df, pulled_metadata)

    return {
        'catalog': catalog, 'hypothesis_results': hyp_df, 'cohens_d': cohens_d,
        'model_results': model_res_df, 'feature_importance': imp_df, 'best_model_name': best_name,
        'cluster_profile': cluster_profile, 'calibrated_constant': calibrated_constant,
        'sensitivity': sens_df, 'churn_coef': coef_df, 'churn_auc': churn_auc,
        'power_analysis': power_results, 'memo': memo, 'top_titles': top_titles, 'bottom_titles': bottom_titles,
    }