# Lab 2 - NumPy and Pandas
## Vectorized Programming and Titanic Dataset Data Wrangling


## Student Information
 **Name**  Avdhi Shah 
 **Student ID**  202618006 
 **Dataset**  Kaggle Titanic Dataset (`train.csv`) 
 **Tools**  Python, NumPy, Pandas, Matplotlib, Seaborn 
 **Notebook**  Jupyter Notebook 

---

## Objective

The objective of this lab is to practice **vectorized programming using NumPy** and **basic data wrangling, analysis, and visualization using Pandas**.

The lab is divided into two parts:

- **Part A:** Vectorized Programming with NumPy
- **Part B:** Data Wrangling with Pandas using the Titanic dataset

The analysis demonstrates array manipulation, statistical calculations, linear algebra, random data generation, filtering, grouping, missing-value handling, feature engineering, outlier detection, and visualization.

---

# Part A - Vectorized Programming with NumPy

## Task 1 - Arrays, Statistics, and Indexing

This task demonstrates fundamental NumPy array operations.

The following operations were performed:

- Generated Array A with 100 random integers.
- Used a random seed for reproducibility.
- Calculated minimum, maximum, median, mean, and standard deviation.
- Generated Array B with exactly 100 values using `np.arange()`.
- Created arrays using `np.zeros()` and `np.ones()`.
- Displayed array shape and data type.
- Used `np.linspace()` to generate evenly spaced values.
- Compared `np.linspace()` with `np.arange()`.
- Created 2D and 3D arrays.
- Demonstrated shape, dimensions, indexing, rows, columns, and slicing.
- Used `reshape()` to create a matrix.
- Used `flatten()` to convert the matrix back into a 1D array.

### Key Concept

`np.arange()` generates values based on a step size, while `np.linspace()` generates a specified number of evenly spaced values between two endpoints.

---

## Task 2 - Vectorized Arithmetic and Linear Algebra

Two matrices were created and NumPy vectorized operations were performed.

The following operations were implemented:

- Matrix addition
- Element-wise multiplication
- Matrix multiplication using `@`
- Matrix multiplication using `np.matmul()`
- Matrix transpose
- Determinant
- Matrix inverse
- Verification using `np.allclose()`

Explicit Python loops were avoided.

### Key Concept

The `*` operator performs element-wise multiplication, while `@` performs matrix multiplication.

For an invertible matrix:

```text
A × A⁻¹ = I
