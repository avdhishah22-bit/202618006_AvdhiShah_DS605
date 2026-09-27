# Lab 5 — Garment Employee Productivity

**Everything is in [`202618006_lab5.ipynb`](./garment_productivity_ml.ipynb).**
Data loading, cleaning, the fixed train/test split, Part A (Scikit-learn),
Part B (from-scratch baseline), Part C (from-scratch optimized), and the
comparison tables all run top-to-bottom in that single notebook.

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



