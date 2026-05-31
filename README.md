# Abalone Dataset Descriptive Analysis

## Project Overview

This project performs a descriptive and statistical analysis of the **Abalone dataset** from the UCI Machine Learning Repository. The main focus of the analysis is to explore the physical characteristics of abalones, especially the relationship between **Length** and **Whole Weight**.

The project was completed as part of the **Business Intelligence Workplace Project 1** assignment. The notebook includes data exploration, descriptive statistics, visualizations, outlier detection, data transformation, regression modeling, and residual analysis.

## Assignment Objective

The main objective of the assignment is to analyze the Abalone dataset and explain the behavior of the data using graphs, statistical measures, and conclusions.

The required analysis includes:

* Index plots for `Length` and `Whole_weight`
* Histograms and boxplots
* Descriptive statistics
* Moment-based statistics
* Scatterplot analysis
* Outlier identification and removal
* Data transformation
* Linear regression modeling
* Residual analysis
* Normality checks
* Autocorrelation analysis
* Group segmentation by sex
* Correlation heatmap

## Dataset

The dataset used is the **Abalone dataset**.

The dataset contains physical measurements of abalones. In this project, the columns are named as follows:

| Column         | Description                                  |
| -------------- | -------------------------------------------- |
| Sex            | Category of abalone: Male, Female, or Infant |
| Length         | Longest shell measurement                    |
| Diameter       | Measurement perpendicular to length          |
| Height         | Height of the shell                          |
| Whole_weight   | Total weight of the abalone                  |
| Shucked_weight | Weight of meat                               |
| Viscera_weight | Gut weight                                   |
| Shell_weight   | Shell weight                                 |
| Rings          | Number of rings                              |

The notebook reads the dataset from a local file named:

```bash
abalone.data
```

## Technologies Used

The project was implemented in Python using the following libraries:

```python
pandas
numpy
matplotlib
scipy
scikit-learn
```

## Project Workflow

### 1. Data Loading

The dataset is loaded using `pandas`:

```python
df = pd.read_csv("abalone.data", header=None)
```

After loading the dataset, column names are added manually to make the data easier to understand and analyze.

### 2. Initial Data Exploration

The notebook prints:

* Original dataset shape
* First five rows of the dataset

This gives a quick overview of the dataset size and structure.

### 3. Index Plots

Index plots are created for:

* `Length`
* `Whole_weight`

These plots show how the values change across the dataset index and help identify unusual observations.

### 4. Histograms and Boxplots

Histograms are used to show the distribution of `Length` and `Whole_weight`.

Boxplots are used to identify the spread of the variables and detect possible outliers.

### 5. Descriptive Statistics

The notebook calculates the five-number summary for both variables:

* Minimum
* First quartile, Q1
* Median
* Third quartile, Q3
* Maximum
* Interquartile range, IQR

These statistics help describe the central tendency and spread of the data.

### 6. Moment-Based Statistics

The following statistics are calculated:

* Mean
* Standard deviation
* Variance
* Skewness
* Kurtosis

These values help describe the shape, variability, and distribution of the selected variables.

### 7. Scatterplot Analysis

A scatterplot of `Whole_weight` against `Length` is created.

This plot shows the relationship between the size of the abalone and its total weight. The relationship is positive, meaning that abalones with larger length usually have higher whole weight.

### 8. Outlier Detection and Removal

Outliers are identified using the IQR method.

The outlier limits are calculated using:

```python
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

Outliers are detected for both `Length` and `Whole_weight`.

The notebook then:

* Prints the number of outliers
* Displays the outlier rows
* Marks outliers on index plots
* Marks outliers on the scatterplot
* Removes the outliers from the dataset
* Creates a second statistics table after removing outliers

### 9. Data Transformation

A logarithmic transformation is applied to both variables:

```python
df_clean["log_length"] = np.log(df_clean["Length"])
df_clean["log_weight"] = np.log(df_clean["Whole_weight"])
```

The goal of this transformation is to make the relationship between length and weight more linear.

### 10. Linear Regression Modeling

A linear regression model is fitted using:

* Independent variable: `log_length`
* Dependent variable: `log_weight`

The model calculates:

* Intercept
* Coefficient
* R² score
* Mean squared error, MSE

The regression line is plotted on the transformed data.

### 11. Residual Analysis

Residuals are calculated as:

```python
residuals = y - y_pred
```

A residual plot is created to check whether the variance is approximately constant.

The notebook also calculates:

* Mean of residuals
* Standard deviation of residuals
* Variance of residuals

### 12. Normality Analysis

The notebook checks the residual distribution using:

* Histogram of residuals with a theoretical normal curve
* QQ-plot of residuals

These plots help evaluate whether the residuals are approximately normally distributed.

### 13. Independence Test

An autocorrelation function, ACF, plot is created for the residuals.

This helps check whether residuals are independent from each other across different lags.

### 14. Group Segmentation by Sex

The notebook compares `Whole_weight` across the categorical variable `Sex`.

It includes:

* Boxplots of `Whole_weight` by sex
* Scatterplot of `Whole_weight` vs. `Length` colored by sex

This helps compare differences between male, female, and infant abalones.

### 15. Correlation Heatmap

A correlation heatmap is created for the full numeric feature set.

The categorical `Sex` variable is encoded as:

```python
M = 0
F = 1
I = 2
```

The heatmap shows the strength and direction of relationships between the variables.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Open the Project Folder

```bash
cd your-repository-name
```

### 3. Install the Required Libraries

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

### 4. Add the Dataset

Make sure the dataset file is placed in the same folder as the notebook:

```bash
abalone.data
```

### 5. Run the Notebook

Open the notebook:

```bash
jupyter notebook P1bi.ipynb
```

Then run the cells from top to bottom.

## Repository Structure

```bash
.
├── P1bi.ipynb
├── abalone.data
└── README.md
```

## Main Results

The analysis shows that `Length` and `Whole_weight` have a strong positive relationship. Larger abalones usually have higher whole weight.

After removing outliers, the dataset becomes cleaner and the relationship between the variables becomes easier to model.

The log-log transformation improves the linear pattern between length and weight, making it suitable for linear regression. The regression model is then evaluated using R² and MSE.

Residual plots and normality checks are used to evaluate the quality of the regression model.

## Conclusion

This project demonstrates a complete descriptive analysis workflow using Python. It starts with basic exploration and visualization, then continues with statistical summaries, outlier handling, transformation, regression modeling and diagnostic checks.

The project shows that abalone length is an important predictor of whole weight. The analysis also shows the importance of cleaning the data and checking model assumptions before making conclusions.
