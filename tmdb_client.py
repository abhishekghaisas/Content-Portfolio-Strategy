"""
tmdb_client.py — TMDB helpers for ad hoc single-title lookups (the search feature in app.py).

Kept separate from fetch_new_titles.py's bulk-pull logic on purpose: that code is already tested
and wired into the daily cron job, and a search feature has different needs (fetch exactly one
title, always get its freshest data, no batching/pagination).
"""
import requests
import pandas as pd
from datetime import datetime, timezone

from pipeline import TMDB_GENRE_ID_TO_BUCKET

TMDB_BASE = "https://api.themoviedb.org/3"


def search_movies(api_key, query, max_results=8):
    """Returns a short list of candidate matches for disambiguation (title, year, poster)."""
    resp = requests.get(f"{TMDB_BASE}/search/movie", params={
        "api_key": api_key, "query": query, "include_adult": "false"
    }, timeout=10).json()
    if 'results' not in resp:
        raise RuntimeError(f"TMDB search failed — check your API key. Response: {resp}")

    results = []
    for r in resp['results'][:max_results]:
        results.append({
            'id': r['id'],
            'title': r.get('title', '(untitled)'),
            'year': (r.get('release_date') or '')[:4],
            'poster_path': r.get('poster_path'),
            'overview': r.get('overview', ''),
        })
    return results


def fetch_movie_row(api_key, tmdb_id):
    """
    Fetches full detail for one movie and returns a dict matching the catalog schema used
    throughout the dashboard, always with the freshest data (including current box office
    revenue if it's been released). Returns None if the title is missing budget or runtime,
    since the economics layer can't evaluate a title without those.
    """
    detail = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}", params={
        "api_key": api_key, "append_to_response": "credits"
    }, timeout=10).json()
    if 'id' not in detail:
        raise RuntimeError(f"TMDB detail fetch failed. Response: {detail}")

    budget = detail.get('budget', 0) or 0
    runtime = detail.get('runtime', 0) or 0
    revenue = detail.get('revenue', 0) or 0
    if budget <= 0 or runtime <= 0:
        return None

    cast = sorted(detail.get('credits', {}).get('cast', []), key=lambda c: c.get('order', 99))
    lead_pop_raw = cast[0]['popularity'] if cast else 5.0
    # ASSUMPTION: bulk pulls rescale star power to a 0-100 percentile WITHIN their pull batch.
    # A single ad hoc lookup has no batch to rank against, so this uses a rough absolute-scale
    # approximation instead — flagged here since it's not computed the same way as bulk-pulled
    # titles and the two aren't perfectly comparable.
    lead_star_power = round(min(lead_pop_raw, 100.0), 1)

    genre_ids = [g['id'] for g in detail.get('genres', [])]  # full detail uses 'genres' objects,
    bucket = next((TMDB_GENRE_ID_TO_BUCKET[g] for g in genre_ids if g in TMDB_GENRE_ID_TO_BUCKET), 'Drama')

    release_date = pd.to_datetime(detail.get('release_date'), errors='coerce')
    if pd.isna(release_date):
        return None

    return {
        'title_id': f"TMDB{detail['id']}", 'tmdb_title': detail.get('title', ''),
        'genre': bucket, 'content_type': 'Original Film',  # ASSUMPTION: licensing status unknown
        'runtime_min': runtime, 'num_episodes': 1,
        'production_budget_musd': budget / 1_000_000,
        'lead_star_power': lead_star_power,
        'critic_score': detail.get('vote_average', 5.0) * 10,
        'release_month': release_date.month,
        'release_dayofweek': release_date.day_name()[:3],
        'tmdb_popularity': detail.get('popularity', 0.0),
        'overview': detail.get('overview', ''),
        'poster_path': detail.get('poster_path'),
        'release_date_full': release_date.strftime('%Y-%m-%d'),
        'real_revenue_musd': revenue / 1_000_000,
        'is_series': 0, 'licensing_flag': 0,
        'source': 'real_tmdb_pull', 'pulled_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
    }