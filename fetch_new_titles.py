"""
fetch_new_titles.py — pulls newly released movies from TMDB and appends them to
data/catalog_history.csv. Designed to be idempotent: re-running it never duplicates
titles already in the history file, and never changes previously-simulated engagement
values for existing titles (see pipeline.py's per-row deterministic engagement simulation).

Run manually:      python fetch_new_titles.py
Run via GitHub Actions: see .github/workflows/daily_tmdb_pull.yml (scheduled daily)

Requires TMDB_API_KEY as an environment variable.
"""
import os
import sys
import time
import json
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np

from pipeline import TMDB_GENRE_ID_TO_BUCKET, ensure_columns

TMDB_BASE = "https://api.themoviedb.org/3"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "catalog_history.csv")
META_PATH = os.path.join(os.path.dirname(__file__), "data", "last_pull.json")


def fetch_recent_titles(api_key, max_new=60, max_pages=6, sleep_sec=0.05, verbose=True):
    """
    Pulls from TMDB's 'now playing' + 'upcoming' endpoints — these are the titles that are
    genuinely NEW in TMDB's catalog on any given day, as opposed to a fixed popularity-sorted
    snapshot. Genre distribution here is NOT balanced by design — it reflects whatever actually
    released that week, which is itself a useful signal.
    """
    rows = []
    seen_ids = set()

    for endpoint in ["movie/now_playing", "movie/upcoming"]:
        page = 1
        while len(rows) < max_new and page <= max_pages:
            resp = requests.get(f"{TMDB_BASE}/{endpoint}", params={
                "api_key": api_key, "page": page, "language": "en-US", "region": "US"
            }, timeout=10).json()

            if 'results' not in resp:
                if verbose:
                    print(f"  WARNING: unexpected response from {endpoint}: {resp}")
                break

            for r in resp['results']:
                if len(rows) >= max_new or r['id'] in seen_ids:
                    continue
                seen_ids.add(r['id'])

                detail = requests.get(f"{TMDB_BASE}/movie/{r['id']}", params={
                    "api_key": api_key, "append_to_response": "credits"
                }, timeout=10).json()
                time.sleep(sleep_sec)

                budget, runtime = detail.get('budget', 0) or 0, detail.get('runtime', 0) or 0
                revenue = detail.get('revenue', 0) or 0
                if budget <= 0 or runtime <= 0:
                    continue

                cast = sorted(detail.get('credits', {}).get('cast', []), key=lambda c: c.get('order', 99))
                lead_pop = cast[0]['popularity'] if cast else 5.0

                genre_ids = r.get('genre_ids', [])
                bucket = next((TMDB_GENRE_ID_TO_BUCKET[g] for g in genre_ids if g in TMDB_GENRE_ID_TO_BUCKET), 'Drama')

                release_date = pd.to_datetime(r.get('release_date'), errors='coerce')
                if pd.isna(release_date):
                    continue

                rows.append({
                    'title_id': f"TMDB{r['id']}", 'tmdb_title': r.get('title', ''),
                    'genre': bucket, 'runtime_min': runtime,
                    'production_budget_musd': budget / 1_000_000,
                    'lead_star_power_raw': lead_pop,
                    'critic_score': r.get('vote_average', 5.0) * 10,
                    'release_month': release_date.month,
                    'release_dayofweek': release_date.day_name()[:3],
                    'tmdb_popularity': r.get('popularity', 0.0),
                    'overview': r.get('overview', ''),
                    'poster_path': r.get('poster_path'),
                    'release_date_full': release_date.strftime('%Y-%m-%d'),
                    'real_revenue_musd': revenue / 1_000_000,  # 0 if unreleased or TMDB has no figure yet
                })
            page += 1
        if verbose:
            print(f"  {endpoint}: {len(rows)}/{max_new} valid titles collected so far")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Rescale star power to a 0-100 percentile WITHIN THIS BATCH. Note this means a title's
    # star-power score can shift slightly between pulls as the comparison batch changes — a
    # known tradeoff of using a relative percentile rather than TMDB's raw unbounded popularity
    # score. Documented here rather than silently accepted.
    df['lead_star_power'] = (df['lead_star_power_raw'].rank(pct=True) * 100).round(1)
    df = df.drop(columns=['lead_star_power_raw'])

    np.random.seed(int(datetime.now(timezone.utc).timestamp()) % (2**31))  # varies run to run, fine — cosmetic only
    df['content_type'] = np.random.choice(['Original Film', 'Licensed Film'], len(df), p=[0.5, 0.5])
    df['num_episodes'] = 1
    df['is_series'] = 0
    df['licensing_flag'] = df['content_type'].str.startswith('Licensed').astype(int)
    df['source'] = 'real_tmdb_pull'
    df['pulled_date'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return df


def refresh_missing_revenue(api_key, history, max_refresh=40, sleep_sec=0.05, verbose=True):
    """
    Titles pulled before revenue-capture existed, or pulled while still too new for TMDB to have
    a reported box office figure, are stuck with real_revenue_musd=0 forever unless something
    goes back and re-checks them. This re-fetches details for exactly those titles (capped per
    run so a backlog doesn't blow through API limits in one go) and updates them IN PLACE, rather
    than only ever appending brand-new titles.
    """
    if 'real_revenue_musd' not in history.columns:
        history['real_revenue_musd'] = 0.0

    is_real = history['source'] == 'real_tmdb_pull'
    is_tmdb_id = history['title_id'].astype(str).str.startswith('TMDB')
    needs_refresh = history[is_real & is_tmdb_id & (history['real_revenue_musd'].fillna(0) <= 0)].copy()

    if needs_refresh.empty:
        if verbose:
            print("No titles need a revenue refresh.")
        return history, 0

    # Prioritize titles that released longest ago — most likely to have a reported figure by now
    if 'release_date_full' in needs_refresh.columns:
        needs_refresh['_release_dt'] = pd.to_datetime(needs_refresh['release_date_full'], errors='coerce')
        needs_refresh = needs_refresh.sort_values('_release_dt', na_position='last')

    refreshed = 0
    for _, row in needs_refresh.head(max_refresh).iterrows():
        tmdb_numeric_id = str(row['title_id']).replace('TMDB', '')
        if not tmdb_numeric_id.isdigit():
            continue
        try:
            detail = requests.get(f"{TMDB_BASE}/movie/{tmdb_numeric_id}",
                                   params={"api_key": api_key}, timeout=10).json()
        except Exception as e:
            if verbose:
                print(f"  Skipping {row['title_id']}: request failed ({e})")
            continue
        time.sleep(sleep_sec)

        revenue = detail.get('revenue', 0) or 0
        if revenue > 0:
            history.loc[history['title_id'] == row['title_id'], 'real_revenue_musd'] = revenue / 1_000_000
            refreshed += 1

    if verbose:
        checked = min(len(needs_refresh), max_refresh)
        print(f"Revenue refresh: checked {checked} candidate titles, updated {refreshed} with a "
              f"newly-reported box office figure. {len(needs_refresh) - checked} still pending "
              f"for a future run." if len(needs_refresh) > max_refresh else
              f"Revenue refresh: checked {checked} candidate titles, updated {refreshed}.")
    return history, refreshed


def main():
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        print("ERROR: TMDB_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    if os.path.exists(DATA_PATH):
        history = pd.read_csv(DATA_PATH)
        history = ensure_columns(history)
    else:
        # Should not normally happen — bootstrap seed should be committed to the repo —
        # but fail gracefully into an empty frame with the right columns if it's missing.
        history = pd.DataFrame(columns=[
            'title_id', 'tmdb_title', 'genre', 'content_type', 'runtime_min', 'num_episodes',
            'production_budget_musd', 'lead_star_power', 'critic_score', 'release_month',
            'release_dayofweek', 'is_series', 'licensing_flag', 'source', 'pulled_date',
            'real_revenue_musd'
        ])

    print(f"Existing history: {len(history)} titles")
    existing_ids = set(history['title_id'])

    history, n_refreshed = refresh_missing_revenue(api_key, history)

    new_titles = fetch_recent_titles(api_key, max_new=60)
    if new_titles.empty:
        print("No new titles fetched this run (API issue or nothing new).")
        added = 0
    else:
        new_titles = new_titles[~new_titles['title_id'].isin(existing_ids)]
        added = len(new_titles)
        if added > 0:
            history = pd.concat([history, new_titles], ignore_index=True)
        print(f"Added {added} genuinely new titles. Total history now: {len(history)}")

    if added > 0 or n_refreshed > 0:
        history.to_csv(DATA_PATH, index=False)

    meta = {
        'last_pull_date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'n_titles_added_this_pull': added,
        'n_titles_revenue_refreshed_this_pull': n_refreshed,
        'total_titles': len(history),
        'n_real_titles': int((history['source'] == 'real_tmdb_pull').sum()) if 'source' in history else 0,
        'n_simulated_titles': int((history['source'] == 'simulated_bootstrap').sum()) if 'source' in history else 0,
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    print("Metadata written:", meta)


if __name__ == "__main__":
    main()