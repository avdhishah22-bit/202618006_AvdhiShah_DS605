# Lab 2 - NumPy and Pandas
## Vectorized Programming and Titanic Dataset Data Wrangling


## Student Information


 **Name**  Avdhi Shah 
 **Student ID** 202618006
 **Lab**  Lab 2 - NumPy and Pandas 
 **Dataset**  Kaggle Titanic Dataset (`train.csv`) 


## Objective

The objective of this lab is to practice **vectorized programming using NumPy** and **data wrangling and visualization using Pandas**.

The lab focuses on creating and manipulating NumPy arrays, performing statistical and linear algebra operations, generating data from a normal distribution, and analyzing the Kaggle Titanic dataset using Pandas.


# Part A - Vectorized Programming with NumPy

## Task 1 - Arrays, Statistics, and Indexing

This task demonstrates fundamental NumPy array operations.

The following operations were performed:

- Generated an array of 100 random integers using `np.random.randint()`.
- Used `np.random.seed()` to make the random values reproducible.
- Calculated the minimum, maximum, median, mean, and standard deviation.
- Generated an array of exactly 100 values using `np.arange()`.
- Created arrays using `np.zeros()` and `np.ones()`.
- Displayed array shape and data type.
- Used `np.linspace()` to generate evenly spaced values.
- Compared `np.linspace()` with `np.arange()`.
- Created and explored 2D and 3D arrays.
- Demonstrated indexing, rows, columns, and slicing.
- Used `reshape()` to convert an array into a matrix.
- Used `flatten()` to convert the matrix back into a 1D array.

### Key Concept

`np.arange()` generates values using a specified step size, while `np.linspace()` generates a specified number of evenly spaced values between two endpoints.

---

# Task 2 - Vectorized Arithmetic and Linear Algebra

Two matrices were created and NumPy vectorized operations were used to perform:

- Matrix addition
- Element-wise multiplication
- Matrix multiplication using `@` and `np.matmul()`
- Matrix transpose
- Determinant calculation
- Matrix inverse
- Verification of the inverse using `np.allclose()`



### Key Concept

The `*` operator performs **element-wise multiplication**, while the `@` operator performs **matrix multiplication**.

The inverse of a matrix exists only when its determinant is non-zero.


