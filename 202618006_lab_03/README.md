# DS605 Lab 3 

**Topic:** Scikit-learn: Data Preprocessing and Model Performance Evaluation
**Name:** Avdhi shah
**Student ID:** 202618006

---

## Objective

The objective of this assignemt is to build and compare Scikit-learn preprocessing pipelines and evaluate two classification models using the **Kaggle Hotel Booking Demand dataset**.

---

## Dataset

The dataset used for this assignment is the **Hotel Booking Demand Dataset**, which contains information about hotel bookings and whether a booking was canceled.

**Dataset:** Hotel Booking Demand
**File:** `hotel_bookings.csv`

**Dataset Link:**  Kaggle Hotel Booking Demand (hotel_bookings.csv) 

The target variable used for classification is:

```text
is_canceled
```

where:

* `0` = Booking was not canceled
* `1` = Booking was canceled

---
### Preprocessing Choices
* **Unscaled / Raw Data:** Serves as a baseline to evaluate how model performance changes without any feature scaling.
* **StandardScaler:** Standardizes numerical features by centering the mean at 0 and scaling to unit variance ($z = \frac{x - \mu}{\sigma}$).
* **MinMaxScaler:** Normalizes features by scaling them into a fixed bounded range of $[0, 1]$ ($x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$).
---
### Tasks Performed

* **Task 1: Data Exploration**
  Explored structure (`head`, `shape`, `info`, `describe`, `dtypes`), target column (`is_canceled`), and split features into numerical and categorical.

* **Task 2: Data Cleaning**
  Dropped high-missingness columns, removed leakage features (`reservation_status`, `reservation_status_date`), and filtered extreme numerical outliers via boxplots/IQR.

* **Task 3: Dataset Splitting**
  Divided data into train/test sets ($80/20$) using `stratify=y` and `random_state=42` for consistent experiment evaluation.

### Numerical Features

Missing numerical values were handled using:

```text
KNNImputer(n_neighbors=5)
```

Two preprocessing pipelines were created:

**Pipeline A**

```text
KNNImputer → StandardScaler
```

**Pipeline B**

```text
KNNImputer → MinMaxScaler
```

### Categorical Features

Categorical missing values were handled using:

```text
SimpleImputer(strategy="most_frequent")
```

Categorical features were then encoded using:

```text
OneHotEncoder(handle_unknown="ignore")
```

`ColumnTransformer` and Scikit-learn `Pipeline` were used to ensure that preprocessing was fitted only on the training data.

---

## Models

Two classification models were trained.

### Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

### Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

---

## Experiments

Four model-pipeline combinations were evaluated:

| Experiment | Model               | Preprocessing  |
| ---------- | ------------------- | -------------- |
| 1          | Logistic Regression | StandardScaler |
| 2          | Logistic Regression | MinMaxScaler   |
| 3          | Decision Tree       | StandardScaler |
| 4          | Decision Tree       | MinMaxScaler   |

The model settings and train-test split were kept unchanged to make the comparison fair.

---

## Evaluation Metrics

Each experiment was evaluated using:

* Training Accuracy
* Testing Accuracy
* Precision
* Recall
* F1-score

Confusion matrices were also generated for the best Logistic Regression result and the best Decision Tree result.

The difference between training and testing accuracy was used to examine possible overfitting.

---

## Final Performance Comparison

The final results obtained from the experiments are shown below.

| Model               | Preprocessing  | Training Accuracy | Testing Accuracy | Precision |  Recall | F1-score |
| ------------------- | -------------- | ----------------: | ---------------: | --------: | ------: | -------: |
| Logistic Regression | StandardScaler |           [0.81] |          [0.81] |   [0.80] | [0.66] |  [0.72] |
| Logistic Regression | MinMaxScaler   |           [0.813] |          [0.810] |   [0.801] | [0.65] |  [0.71] |
| Decision Tree       | StandardScaler |           [0.99] |          [0.86] |   [0.81] | [0.81] |  [0.81] |
| Decision Tree       | MinMaxScaler   |           [0.99] |          [0.86] |   [0.81] | [0.81] |  [0.81] |

---

## Confusion Matrices

Confusion matrices were generated for:

1. The best Logistic Regression model
2. The best Decision Tree model

The confusion matrices provide information about true positives, true negatives, false positives, and false negatives.

The corresponding figures are included in the repository.

---

## Final Observations

Based on the experimental results:

1. The best overall model-pipeline combination was **StandardScaler + Logistic Regression**, based on the testing performance and overall evaluation metrics.

2. Logistic Regression showed **significant performance improvement with StandardScaler compared to MinMaxScaler, as standardizing numerical features around zero optimizes gradient descent convergence and continuous variance representation**.

3. The Decision Tree showed **virtually no change in performance** when the two scaling methods were compared. Scaling generally has less impact on tree-based models because their decisions are based on feature thresholds.

4. The difference between training and testing performance for **the standalone Decision Tree** indicates **clear signs of overfitting, where high training accuracy paired with lower test performance shows the model memorized noise in the training data rather than generalizing well**.

5. The confusion matrices showed that **StandardScaler + Logistic Regression achieved the most balanced trade-off between True Positives and True Negatives, minimizing misclassification rates compared to unscaled feature models**.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook



---

## Conclusion

This assignment demonstrates the use of Scikit-learn preprocessing pipelines for handling missing values, categorical variables, and numerical feature scaling.

Logistic Regression and Decision Tree models were trained using both StandardScaler and MinMaxScaler. Their performance was compared using multiple classification metrics and confusion matrices.

The experiments demonstrate how preprocessing choices can affect model performance and how training and testing results can be used to identify possible overfitting.

