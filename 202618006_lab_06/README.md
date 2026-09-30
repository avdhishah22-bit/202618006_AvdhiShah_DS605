# Lab 6: Feature Extraction and Machine Learning with Image and Text Data

Classical machine learning on hand-crafted image features (asphalt cracks) and vectorised text
(email spam). **No CNNs, deep learning, or pretrained embeddings are used anywhere** - image
features are computed from raw pixels using NumPy/OpenCV only; scikit-learn is used only for
vectorizers (CountVectorizer, TF-IDF) and traditional ML models.

## Repository structure
```
202618006_lab_06/
├── README.md
├── 202618006_lab6.ipynb
├── figures/
├── results/

```

## How to run
1. Open `202618006_lab6.ipynb` in Jupyter or VS Code.
2. Run the cell at the top first to setup, it installs every required package
   (`numpy<2`, `pandas`, `opencv-python`, `matplotlib`, `scikit-learn`) into the current kernel.
3. Run all cells top to bottom (Kernel → Restart & Run All).

## Datasets
| Task | Source |
|---|---|
| Asphalt cracks (400 images) | Mendeley Data - Asphalt Crack Dataset | 
| Email spam (5,172 emails) | Kaggle - Email Spam Classification Dataset | 

**Note on the email file:** emails spam is supplied as a pre-computed 3,000-word count matrix
per email (no raw text), plus a `Prediction` label. To still use CountVectorizer/TF-IDF as
required, each email is reconstructed as a bag-of-words pseudo-document (each word repeated by
its count) before vectorising. Word order is lost in this reconstruction, so bigram features are
only evaluated if a raw-text CSV is used instead.

## Method summary

**Part A - Image features and classification.** Images are read with OpenCV, their
dimensions/channels inspected, resized to 256x256, and converted to grayscale. 14 baseline
features are computed per image with NumPy/OpenCV: mean brightness, contrast (std), median,
min/max, 5th/95th percentiles, skewness, kurtosis, dark-pixel ratio (<50), bright-pixel ratio
(>200), entropy, and Canny edge count/density. Four classifiers are trained and compared -
Logistic Regression, SVM (RBF), KNN, Random Forest - on an 80/20 stratified split, plus 5-fold
cross-validation F1 (useful given only 400 images). Accuracy, precision, recall, F1, confusion
matrix, training time, and prediction time are reported for each.

**Part B - Text vectorisation and spam classification.** Basic cleaning (lowercasing, stripping
"Subject:", URLs, HTML tags, non-letters) is applied, then email text is vectorised with both
CountVectorizer and TF-IDF. Each representation is used to train Multinomial Naive Bayes,
Logistic Regression, and Linear SVC. Accuracy, precision, recall, F1, number of generated
features, vectorisation time, training time, and prediction time are compared across the two
representations.

**Part C - Improved representations.**
- *Images:* 41 features, adding multi-threshold and Otsu-based Canny edge density, Sobel
  gradient-magnitude statistics, a black-hat morphological filter (targets thin dark
  crack-like lines), CLAHE-normalised intensity statistics, and a 4x4 spatial grid of edge
  density (to retain some spatial information lost by global statistics).
- *Text:* stop-word removal, `min_df=3`, and a capped vocabulary (2,000 and 1,000 words),
  compared against the full-vocabulary baseline.

Each improvement is directly compared against its baseline on accuracy/F1, feature
dimensionality, and computation time.

## Limitations
- Only 400 images means test-set metrics are noisy; 5-fold CV F1 is reported alongside the
  held-out test score as a sanity check.
- The email dataset (Enron-derived) contains duplicate reconstructed documents (~541 of 5,172),
  so a small amount of train/test leakage is possible.
- Reconstructing text from a word-count matrix loses word order, so n-gram features could not be
  meaningfully evaluated on this file.
