# Airbnb NYC Price Prediction — End-to-End ML Project

**Dataset:** Kaggle "New York City Airbnb Open Data" (`AB_NYC_2019.csv`)

🚀 **Live app:** https://202618006avdhishahds605-cx4f3mu3y9j4cavh9yqxmz.streamlit.app/

## Files

| File | Purpose |
|---|---|
| `Airbnb_Price_Prediction.ipynb` | Task 1 (EDA, cleaning, feature engineering) + Task 2 (model comparison, tuning, evaluation) + Task 4 (final summary, filled in with actual results). Saves `airbnb_price_pipeline.pkl` at the end. |
| `app.py` | Task 3 — Streamlit app that loads the saved pipeline and predicts nightly price from user input. |
| `airbnb_price_pipeline.pkl` | Saved scikit-learn pipeline (preprocessing + trained model). |
| `AB_NYC_2019.csv` | The dataset. |
| `requirements.txt` | Deployment dependencies, with exact versions pinned to match the training environment. |

## Setup & How to Run Locally

```bash
pip install -r requirements.txt

# If running the notebook and you're NOT using Anaconda, also install Jupyter
# (Anaconda already bundles it, so this is only needed otherwise):
pip install jupyter

# 1. AB_NYC_2019.csv should already be in this same folder.

# 2. Run the notebook top to bottom — trains/tunes the model and saves
#    airbnb_price_pipeline.pkl
jupyter notebook Airbnb_Price_Prediction.ipynb

# 3. Launch the Streamlit app
streamlit run app.py
```
## Project Summary

### Task 1 — Data Analysis & Preparation
- **Cleaning:** removed `price == 0` rows (not real bookable prices), capped the top 1% of
  prices as extreme outliers, and capped `minimum_nights` at 365 (a handful of listings had
  values like 1000+, clearly not realistic short-term-rental minimums). Started with
  **48,895** listings; **48,396** remained after cleaning (1.0% removed).
- **Missing values:** `reviews_per_month` and `last_review` are missing exactly when a
  listing has zero reviews — resolved logically (0 reviews/month; "days since last review"
  set to the dataset maximum, i.e. treated as long-stale) rather than dropped or naively imputed.
- **Feature engineering:** derived `days_since_last_review` from the raw date, and modeled
  `log1p(price)` instead of raw price, since price is heavily right-skewed.
- **Feature selection:** kept borough (`neighbourhood_group`), `room_type`, coordinates, and
  listing/host activity stats (reviews, availability, host's listing count). Dropped
  identifiers (`id`, `host_id`, `name`, `host_name`) and the high-cardinality `neighbourhood`
  column (~220 categories) in favor of the coarser `neighbourhood_group` + coordinates.

### Task 2 — Model Training & Evaluation
- Compared **Linear Regression, Ridge, Random Forest, and HistGradientBoosting** on
  RMSE (log-price scale), R², and MAE (in dollars), and checked the train/test RMSE gap
  as an overfitting signal for each.
- **Baseline results:** HistGradientBoosting had the lowest test RMSE (0.393, log scale),
  Random Forest close behind (0.395), both clearly beating Linear Regression/Ridge (0.449).
  Random Forest's overfit gap (0.251) was already much larger than HistGradientBoosting's
  (0.023) even before any tuning — an early signal of which model would generalize better.
- **Final model chosen: HistGradientBoosting**, tuned with `RandomizedSearchCV` (3-fold CV)
  over `max_iter`, `max_depth`, and `learning_rate`. Best params: `max_iter=200,
  max_depth=None, learning_rate=0.1`.
- **Final test performance:** RMSE ≈ $75, MAE ≈ $43, R² = 0.637, overfit gap only 0.031
  (log scale) — good generalization, minimal overfitting.
- **Why HistGradientBoosting over Random Forest:** comparable accuracy, but a dramatically
  smaller saved model (~0.6 MB vs. 60+ MB for a similarly-tuned Random Forest) — a deliberate
  trade-off favoring a practical, easily deployable model with no meaningful accuracy cost.
- Final pipeline (preprocessing + tuned model) is saved as a single object with `joblib`,
  so `app.py` calls `.predict()` directly on raw new input without re-implementing any
  preprocessing.

### Task 3 — Streamlit Application
`app.py` takes borough, room type, coordinates, minimum nights, review stats, host listing
count, and availability, and returns a predicted nightly price with a confidence caveat.
It loads `airbnb_price_pipeline.pkl` directly, anchoring the file path to its own folder
location (not the working directory) so it works correctly regardless of where in a repo
it's deployed from. **Live at:**
https://202618006avdhishahds605-cx4f3mu3y9j4cavh9yqxmz.streamlit.app/

### Task 4 — Final Summary & Limitations
Written out in full in the notebook's final markdown cell, with real numbers (not
placeholders). Key limitations: the fine-grained `neighbourhood` feature was dropped for
cardinality reasons (some within-borough variation is lost), the model reflects 2019 market
conditions only, extreme luxury listings (top 1% of price) were excluded from training so
predictions for very high-end listings will be systematically low, and an R² of 0.637 means
roughly 64% of price variation is explained — solid, but factors like listing photos, host
reputation, and seasonal demand aren't captured by these features alone.

## Deployment Notes (Streamlit Community Cloud)

This app is deployed and live at the link above. A few real issues came up during
deployment that are worth documenting:

- **Exact version pinning is required.** `requirements.txt` originally used loose bounds
  (`scikit-learn>=1.3`), which let Streamlit Cloud install a different scikit-learn version
  than the one used to train and pickle the model locally — this caused an `AttributeError`
  when unpickling. Fixed by pinning exact versions matching the local (Anaconda) training
  environment: `scikit-learn==1.5.1`, `numpy==1.26.4`, `joblib==1.4.2`.
- **`jupyter` should NOT be in the deployment `requirements.txt`.** It was originally
  included for convenience, but it's a heavy package with a large dependency tree that
  significantly slows down cloud builds — and the deployed app never imports it. Removing
  it cut the resolved package count from 126 down to 49.
- **Python version matching matters too.** Streamlit Cloud's default Python version may not
  match the version used locally to train the model (e.g. Cloud defaulting to a very new
  Python release without prebuilt wheels yet for some pinned package versions, forcing slow
  or failing source builds). Set the Python version explicitly in the app's Advanced
  Settings to match your local training environment (3.12, matching Anaconda, for this
  project).

