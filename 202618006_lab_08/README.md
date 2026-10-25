#  Decision Trees and Support Vector Machines

Decision Tree and Linear SVM implemented **from scratch (NumPy/Pandas only)**, compared with Scikit-learn, plus kernel SVMs on nonlinear data.
Dataset: [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) — 1,372 samples, 4 features (variance, skewness, curtosis, entropy), 2 classes.

## Repository contents

| File | Description |
|---|---|
| `202618006_lab_08.ipynb` | Complete runnable notebook (all parts, plots, tables, discussion) |
| `data_banknote_authentication.txt` | Dataset (comma-separated, no header) |
| `README.md` |  |

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook 202618006_lab_08.ipynb      # then Run All
```
The notebook reads `data_banknote_authentication.txt` from the same folder. Random seed is fixed (`42`), so results are reproducible.

## What is implemented

**Setup.** One stratified 80/20 train/test split (own NumPy implementation) is reused by every model. Features are standardised with a from-scratch scaler fitted on the training set only. 2-D plots use *variance + skewness*; final evaluation uses all four features.

**Part A — Decision Tree (`DecisionTreeScratch`).** Gini impurity and entropy; best feature/threshold search using midpoints of sorted values and cumulative class counts; recursive growth with stopping rules (`max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, pure node); vectorised `predict`. Outputs: decision boundaries (manual vs Scikit-learn), printed split rules, a drawn manual tree, and a criterion × depth study with train/test accuracy and prediction agreement against `DecisionTreeClassifier`.

**Part B — Linear SVM (`LinearSVMScratch`).** Labels mapped to −1/+1; soft-margin hinge-loss objective `λ/2‖w‖² + mean(max(0, 1 − y(w·x+b)))` with `λ = 1/(C·n)` (equivalent to Scikit-learn's `C`), optimised by full-batch subgradient descent with decaying learning rate. Plots show the separating line, ±1 margins and margin violators (manual) next to Scikit-learn's SVM with support vectors. Studies: effect of `C`, learning rate and number of iterations (loss curves, convergence to Scikit-learn's optimum, cosine similarity of `w`).

**Part C — Kernel SVM.** `make_moons` and `make_circles`; Scikit-learn SVMs with linear, polynomial (degree 3) and RBF kernels; side-by-side boundaries; RBF `gamma`/`C` sweep.

**Part D — Final comparison (4 features).** Manual tree, Scikit-learn tree, manual linear SVM, Scikit-learn linear SVM, Scikit-learn RBF SVM. Metrics: accuracy, precision, recall, F1, training and prediction time, train-vs-test gap, confusion matrices, model complexity, plus mean ± std over 10 repeated splits.


## Observations

- **Correctness of the from-scratch models.** The manual tree and manual SVM closely reproduce their Scikit-learn counterparts (near-identical predictions, same objective value / weight direction for the SVM). Small tree differences come from tie-breaking between equally good splits.
- **Decision tree.** Very shallow trees underfit; as depth grows, training accuracy approaches 100 % while test accuracy stops improving or drops (overfitting, jagged boundaries). Gini and entropy give very similar trees.
- **Linear SVM.** Small `C` → wide margin, many margin violators (more bias); large `C` → narrow margin, fewer violators (more variance). Very small learning rates converge slowly; large constant rates oscillate, so a decaying step size helps. More iterations move the solution toward Scikit-learn's.
- **Kernels.** A linear SVM cannot separate moons/circles (circles ≈ chance level); polynomial and RBF kernels create curved boundaries by fitting a linear separator in an implicit higher-dimensional space. Large `gamma` or `C` makes the RBF boundary overfit.
- **Final comparison.** Banknote data is almost linearly separable, so all models perform well; the RBF SVM is typically best, with the tree and linear SVM close behind.
- **Interpretability.** Tree: readable `feature <= threshold` rules. Linear SVM: interpretable weight vector. RBF SVM: black box (many support vectors).
- **Scaling.** Trees are scale-invariant; SVMs need standardised features (fitted on training data only to avoid leakage).
- **Speed.** Scikit-learn is faster to train than the pure-Python/NumPy versions; linear models and trees predict very cheaply, while RBF prediction cost grows with the number of support vectors.


