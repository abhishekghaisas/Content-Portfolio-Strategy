# Content Portfolio Strategy Dashboard

A renew/cancel/invest recommendation dashboard for a streaming content catalog, seeded with
simulated data and grown daily with real movie releases pulled from TMDB.

## How it's architected (and why)

Streamlit Community Cloud doesn't support background cron jobs inside the app itself — the app
only runs code when someone has it open. So the "daily update" part lives outside Streamlit:

1. **GitHub Actions** (`.github/workflows/daily_tmdb_pull.yml`) runs on a schedule, calls
   `fetch_new_titles.py`, and commits any newly-released titles to `data/catalog_history.csv`.
2. **The Streamlit app** (`app.py`) just reads that CSV. It shows a lightweight overview
   immediately (no heavy compute), and only runs the full analysis — hypothesis tests, modeling,
   clustering, economics, churn link — when you click **Run Full Analysis**, since that's
   compute-heavier and doesn't need to happen on every page load.

This split (cheap scheduled ingestion vs. on-demand heavy analysis) is the same pattern real data
platforms use, and it's the only way to get a "daily-updating" dashboard on Streamlit's free tier.

## One-time setup

### 1. Get a TMDB API key
Free at https://www.themoviedb.org/settings/api

### 2. Create a GitHub repo and push this folder
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```
The bootstrap `data/catalog_history.csv` (400 simulated titles) is already included, so the
dashboard has something to show from the very first deploy.

### 3. Add your TMDB key as a GitHub Actions secret
Repo → **Settings → Secrets and variables → Actions → New repository secret**
Name: `TMDB_API_KEY`, value: your key.

### 4. Trigger the first real pull manually (don't wait for the schedule)
Repo → **Actions → Daily TMDB pull → Run workflow**. This adds real titles to
`data/catalog_history.csv` and commits them, so you don't have to wait until 13:00 UTC tomorrow
to see real data.

### 5. Deploy to Streamlit Community Cloud
Go to https://share.streamlit.io, connect your GitHub account, pick this repo, set the main file
to `app.py`, and deploy. No Streamlit secrets are required for the automatic daily pull (that
happens in GitHub Actions, not in the deployed app) — the API key field in the app's sidebar is
only for manually testing a pull without waiting for the schedule.

## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `pipeline.py` | Core analysis functions (hypothesis tests, modeling, clustering, economics, churn) shared by the app |
| `fetch_new_titles.py` | Pulls newly-released movies from TMDB, dedupes against history, appends |
| `data/catalog_history.csv` | The growing dataset — starts as 400 simulated titles, gains real titles over time |
| `data/last_pull.json` | Metadata about the most recent automated pull |
| `.github/workflows/daily_tmdb_pull.yml` | The daily schedule |

## Design notes worth knowing
- **Engagement simulation is per-title deterministic** (seeded from a hash of `title_id`), not
  from one global random sequence. This means adding new titles never changes previously-computed
  engagement values for existing titles — an important property for a dataset meant to grow
  indefinitely. See `pipeline.py::_row_seed`.
- **`fetch_new_titles.py` is idempotent** — re-running it (e.g., if a workflow run gets retried)
  never adds duplicate titles, since it checks `title_id` against the existing history first.
- **Real vs. simulated data is tracked explicitly** via the `source` column
  (`simulated_bootstrap` vs. `real_tmdb_pull`), so the dashboard is always honest about what's
  real TMDB metadata vs. what's illustrative.
- **The economics layer auto-calibrates** its churn-reduction constant to the median title in
  whatever catalog is currently loaded, rather than using a fixed number tuned to one budget
  regime — see `pipeline.py::run_economics`.
- **New-title genre distribution is intentionally unbalanced** — `fetch_new_titles.py` pulls from
  `now_playing`/`upcoming`, which reflects whatever actually released, not a designed sample.
