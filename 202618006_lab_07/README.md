# DS605 Lab 7: Image Representation, PCA, Eigenfaces and Face Recognition

Face recognition on the AT&T (ORL) Database of Faces (40 subjects x 10 images, 112 x 92 grayscale).
All PCA / eigenface computations are implemented with NumPy linear algebra. Scikit-learn is used only
in Part D for the traditional classifiers.

## Repository contents

| Path | What it is |
|---|---|
| `lab7_face_recognition.ipynb` | Runnable notebook (Parts A-D) |
| `figures/` | Explained variance, eigenfaces, reconstructions, matches, classifier comparison |
| `results/` | `ml_comparison.csv/.md`, `reconstruction_storage.csv`, `summary.json`, reduced representation `.npz` |


## How to run

1. Download the AT&T/ORL faces and place them as `data/att_faces/s1/1.pgm ... s40/10.pgm`.
2. Run the notebook, or `python lab7_face_recognition.py` (set `ORL_DIR` to use another data path).

Everything is seeded (`SEED = 42`), so results are reproducible.

## Approach

**Part A - Representation.** Each image is read as a grayscale matrix (112 x 92, values 0-255), scaled to [0, 1]
and flattened into a 10,304-dimensional vector, giving a 400 x 10304 matrix with subject labels. The split is
per subject: 8 random images for training and 2 for testing (320 / 80), so every subject appears in both sets.

**Part B - PCA from scratch.**
- Mean face from the training set only; training data is mean-centered.
- *Covariance / eigendecomposition:* the 10304 x 10304 covariance has rank at most 319, so the m x m Gram matrix
  `Xc Xc^T / (m-1)` is decomposed with `np.linalg.eigh` and lifted back to pixel space (`v = Xc^T u`, normalised).
  The code checks that these vectors satisfy `C v = lambda v` for the true covariance.
- *SVD:* `np.linalg.svd(Xc)`, with eigenvalues `S^2/(m-1)`.
- The two are compared on eigenvalues and on the |cosine| of matching directions (signs are arbitrary).
- Cumulative explained variance is plotted and the smallest k reaching 90% is retained.
- Eigenfaces are the principal directions reshaped to 112 x 92.
- Faces are projected, the reduced representation is saved, and test faces are reconstructed for several k,
  reporting MSE, PSNR, per-image compression (`10304/k`) and total storage including the shared mean and components.

**Part C - Recognition.** Each test image is projected, and its nearest training face (Euclidean distance) gives the
predicted subject. This is done on raw pixels and on PCA features, and accuracy is also traced against k.
Correct and incorrect matches are shown in `figures/08_matches.png`.

**Part D - Traditional ML.** KNN, Logistic Regression, linear SVM and RBF SVM are trained on PCA features and
compared to the nearest-neighbour baseline. For each model, k and the hyper-parameters are chosen by 4-fold
stratified cross-validation on the **training set only**, so the test set is never used for tuning.

## Results

Run the notebook to populate these (the files below are written automatically):

- Variance / components: `figures/03_cumulative_variance.png`, `results/summary.json` (`k90`, `k95`)
- Eigenfaces and mean face: `figures/01_mean_face.png`, `figures/04_eigenfaces.png`
- Reconstructions and storage: `figures/05_reconstructions.png`, `results/reconstruction_storage.csv`
- Recognition: `figures/07_nn_accuracy_vs_k.png`, `figures/08_matches.png`
- ML comparison table: `results/ml_comparison.md`, `figures/09_classifier_comparison.png`

> After running on the real data, paste the final numbers and your observations here
> (k for 90% variance, raw-pixel vs PCA accuracy, best classifier, reconstruction/storage trade-off).

## Observations to discuss

- Eigendecomposition and SVD should agree to machine precision; SVD is the more numerically stable route.
- PCA matching should be roughly as accurate as raw-pixel matching while using far fewer dimensions and being much faster.
- Reconstruction quality rises quickly for small k and then flattens; the 90% point is a good compact operating point.
- Accuracy vs k typically peaks and then plateaus or drops slightly as noisy low-variance directions are added.
- Errors tend to come from pose, expression, or lighting differences that dominate the leading components.
