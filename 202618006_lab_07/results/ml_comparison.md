| Model | Features | k | Best params (CV) | CV acc | Test acc |
|---|---|---|---|---|---|
| Nearest neighbour (NumPy, Euclidean) | raw pixels (10304) | 10304 | - | - | 0.988 |
| Nearest neighbour (NumPy, Euclidean) | PCA k=99 | 99 | - | - | 0.988 |
| KNN | PCA k=30 | 30 | {'n_neighbors': 1, 'weights': 'uniform'} | 0.969 | 0.988 |
| Logistic Regression | PCA k=50 | 50 | {'logisticregression__C': 0.01} | 0.959 | 0.950 |
| SVM (linear) | PCA k=30 | 30 | {'svc__C': 0.1} | 0.959 | 0.963 |
| SVM (RBF) | PCA k=75 | 75 | {'C': 10, 'gamma': 'scale'} | 0.981 | 0.988 |
