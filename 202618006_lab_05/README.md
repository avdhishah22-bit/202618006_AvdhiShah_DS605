# DS605 Lab 5 — Garment Employee Productivity

**Everything is in one file: [`garment_productivity_ml.ipynb`](./garment_productivity_ml.ipynb).**
Data loading, cleaning, the fixed train/test split, Part A (Scikit-learn),
Part B (from-scratch baseline), Part C (from-scratch optimized), and the
comparison tables all run top-to-bottom in that single notebook — no local
environment, install, or extra files needed.

## How to run (no environment required)

1. Go to **[colab.research.google.com](https://colab.research.google.com)** and sign in with any Google account.
2. `File → Upload notebook` and select `garment_productivity_ml.ipynb`
   (or, once this is on GitHub: `File → Open notebook → GitHub` tab, paste
   your repo URL, and pick the file).
3. `Runtime → Run all`. It takes under a minute — the first cell installs
   `ucimlrepo` (just a tiny data-downloader, not an ML library) and pulls the
   real UCI dataset automatically.
4. Scroll down: every metric, timing result, and the final comparison tables
   print inline in the notebook itself. That's your full deliverable.

## What's inside

- **Part 0:** loads the UCI *Productivity Prediction of Garment Employees*
  dataset, cleans known quirks (`department` typos/whitespace), builds the
  `MeetsTarget` classification label, and creates one fixed 80/20 train/test
  split reused by every part below.
- **Part A (Scikit-learn — the only place it's used):** `ColumnTransformer` +
  `LinearRegression` / `LogisticRegression`, timed and scored.
- **Part B (from scratch, NumPy/Pandas only):** hand-written imputation,
  one-hot encoding, scaling, closed-form Linear Regression, gradient-descent
  Logistic Regression, and every metric (MAE/RMSE/R², accuracy/precision/
  recall/F1) computed manually.
- **Part C (optimized from scratch):** feature selection (drops near-zero-
  variance columns), ridge-tuned Linear Regression, Adam-optimized +
  L2-regularized Logistic Regression — hyperparameters chosen on an internal
  validation slice cut from the training rows only.
- **Comparison + key observations** at the end of the notebook.

## Push to GitHub

```bash
git init
git add garment_productivity_ml.ipynb README.md
git commit -m "Lab 5: sklearn vs from-scratch ML, single notebook"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

GitHub renders `.ipynb` files with their outputs directly in the browser, so
after you run it once in Colab and push it (with outputs saved), your
grader can see every result without running anything themselves.

## Citation

Al Imran, A. (2020). *Productivity Prediction of Garment Employees* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C51S6D

