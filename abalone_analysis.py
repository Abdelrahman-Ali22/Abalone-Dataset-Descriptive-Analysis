"""
Abalone Dataset Descriptive Analysis
Business Intelligence Workplace Project 1

This script performs descriptive statistics, visualizations, outlier detection,
log transformation, linear regression, residual diagnostics, segmentation by sex,
and correlation analysis for the Abalone dataset.

How to run:
    python abalone_analysis.py

Required libraries:
    pandas, numpy, matplotlib, scipy, scikit-learn
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------------------

DATA_FILE = "abalone.data"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
    "Rings",
]


def save_show(filename: str) -> None:
    """Save the current plot and display it."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()


try:
    df = pd.read_csv(DATA_FILE, header=None, names=COLUMNS)
    print(f"Dataset loaded from local file: {DATA_FILE}")
except FileNotFoundError:
    print("Local file 'abalone.data' was not found. Loading dataset from UCI URL instead...")
    df = pd.read_csv(DATA_URL, header=None, names=COLUMNS)

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# -------------------------------------------------------------------
# 2. Index plots for Length and Whole Weight
# -------------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(df.index, df["Length"])
plt.title("Index Plot of Length")
plt.xlabel("Index")
plt.ylabel("Length")

plt.subplot(1, 2, 2)
plt.plot(df.index, df["Whole_weight"])
plt.title("Index Plot of Whole Weight")
plt.xlabel("Index")
plt.ylabel("Whole Weight")

save_show("01_index_plots.png")


# -------------------------------------------------------------------
# 3. Histograms and boxplots
# -------------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df["Length"], bins=30, edgecolor="black")
plt.title("Histogram of Length")
plt.xlabel("Length")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(df["Whole_weight"], bins=30, edgecolor="black")
plt.title("Histogram of Whole Weight")
plt.xlabel("Whole Weight")
plt.ylabel("Frequency")

save_show("02_histograms.png")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.boxplot(df["Length"])
plt.title("Boxplot of Length")
plt.ylabel("Length")

plt.subplot(1, 2, 2)
plt.boxplot(df["Whole_weight"])
plt.title("Boxplot of Whole Weight")
plt.ylabel("Whole Weight")

save_show("03_boxplots.png")


# -------------------------------------------------------------------
# 4. Descriptive statistics and moment-based statistics
# -------------------------------------------------------------------

def descriptive_stats(data: pd.Series) -> pd.Series:
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    return pd.Series(
        {
            "Min": data.min(),
            "Q1": q1,
            "Median": data.median(),
            "Q3": q3,
            "Max": data.max(),
            "IQR": iqr,
        }
    )


def moment_stats(data: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "Mean": data.mean(),
            "Standard Deviation": data.std(),
            "Variance": data.var(),
            "Skewness": data.skew(),
            "Kurtosis": data.kurt(),
        }
    )


five_number_table = pd.DataFrame(
    {
        "Length": descriptive_stats(df["Length"]),
        "Whole_weight": descriptive_stats(df["Whole_weight"]),
    }
)

moment_table = pd.DataFrame(
    {
        "Length": moment_stats(df["Length"]),
        "Whole_weight": moment_stats(df["Whole_weight"]),
    }
)

print("\n--- Five-Number Summary and IQR ---")
print(five_number_table)

print("\n--- Moment-Based Statistics ---")
print(moment_table)


# -------------------------------------------------------------------
# 5. Scatterplot of Whole Weight vs Length
# -------------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.scatter(df["Length"], df["Whole_weight"], alpha=0.5)
plt.xlabel("Length")
plt.ylabel("Whole Weight")
plt.title("Whole Weight vs Length")
save_show("04_scatter_weight_vs_length.png")


# -------------------------------------------------------------------
# 6. Outlier identification and removal using IQR method
# -------------------------------------------------------------------

Q1_L = df["Length"].quantile(0.25)
Q3_L = df["Length"].quantile(0.75)
IQR_L = Q3_L - Q1_L
lower_L = Q1_L - 1.5 * IQR_L
upper_L = Q3_L + 1.5 * IQR_L

Q1_W = df["Whole_weight"].quantile(0.25)
Q3_W = df["Whole_weight"].quantile(0.75)
IQR_W = Q3_W - Q1_W
lower_W = Q1_W - 1.5 * IQR_W
upper_W = Q3_W + 1.5 * IQR_W

outliers = df[
    (df["Length"] < lower_L)
    | (df["Length"] > upper_L)
    | (df["Whole_weight"] < lower_W)
    | (df["Whole_weight"] > upper_W)
]

print("\nNumber of Outliers:")
print(len(outliers))

print("\nOutlier Rows:")
print(outliers)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(df.index, df["Length"], label="Length")
plt.scatter(outliers.index, outliers["Length"], color="red", label="Outliers")
plt.title("Index Plot of Length with Outliers")
plt.xlabel("Index")
plt.ylabel("Length")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df.index, df["Whole_weight"], label="Whole Weight")
plt.scatter(outliers.index, outliers["Whole_weight"], color="red", label="Outliers")
plt.title("Index Plot of Whole Weight with Outliers")
plt.xlabel("Index")
plt.ylabel("Whole Weight")
plt.legend()

save_show("05_index_plots_with_outliers.png")

plt.figure(figsize=(7, 5))
plt.scatter(df["Length"], df["Whole_weight"], label="Normal Data", alpha=0.3)
plt.scatter(outliers["Length"], outliers["Whole_weight"], color="red", label="Outliers")
plt.xlabel("Length")
plt.ylabel("Whole Weight")
plt.title("Outliers on Scatterplot")
plt.legend()
save_show("06_scatter_with_outliers.png")

df_clean = df.drop(outliers.index).copy()

print("\nClean Dataset Shape After Removing Outliers:")
print(df_clean.shape)

clean_five_number_table = pd.DataFrame(
    {
        "Length": descriptive_stats(df_clean["Length"]),
        "Whole_weight": descriptive_stats(df_clean["Whole_weight"]),
    }
)

clean_moment_table = pd.DataFrame(
    {
        "Length": moment_stats(df_clean["Length"]),
        "Whole_weight": moment_stats(df_clean["Whole_weight"]),
    }
)

print("\n--- Five-Number Summary After Removing Outliers ---")
print(clean_five_number_table)

print("\n--- Moment-Based Statistics After Removing Outliers ---")
print(clean_moment_table)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.boxplot(df_clean["Length"])
plt.title("Boxplot of Length After Outlier Removal")
plt.ylabel("Length")

plt.subplot(1, 2, 2)
plt.boxplot(df_clean["Whole_weight"])
plt.title("Boxplot of Whole Weight After Outlier Removal")
plt.ylabel("Whole Weight")

save_show("07_boxplots_after_outlier_removal.png")

plt.figure(figsize=(7, 5))
plt.scatter(df_clean["Length"], df_clean["Whole_weight"], alpha=0.5)
plt.xlabel("Length")
plt.ylabel("Whole Weight")
plt.title("Whole Weight vs Length After Outlier Removal")
save_show("08_scatter_after_outlier_removal.png")


# -------------------------------------------------------------------
# 7. Data transformation
# -------------------------------------------------------------------

# Log transformation requires positive values.
df_clean = df_clean[(df_clean["Length"] > 0) & (df_clean["Whole_weight"] > 0)].copy()

df_clean["log_length"] = np.log(df_clean["Length"])
df_clean["log_weight"] = np.log(df_clean["Whole_weight"])

plt.figure(figsize=(7, 5))
plt.scatter(df_clean["log_length"], df_clean["log_weight"], alpha=0.5)
plt.xlabel("Log Length")
plt.ylabel("Log Whole Weight")
plt.title("Log-Log Transformation")
save_show("09_log_log_transformation.png")


# -------------------------------------------------------------------
# 8. Regression modeling on transformed data
# -------------------------------------------------------------------

X = df_clean[["log_length"]]
y = df_clean["log_weight"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("\n--- Regression Results ---")
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("R2:", r2)
print("MSE:", mse)

plt.figure(figsize=(7, 5))
plt.scatter(df_clean["log_length"], df_clean["log_weight"], alpha=0.4, label="Data")
plt.plot(df_clean["log_length"], y_pred, color="red", label="Regression Line")
plt.xlabel("Log Length")
plt.ylabel("Log Whole Weight")
plt.title("Linear Regression on Log-Log Transformed Data")
plt.legend()
save_show("10_regression_line.png")


# -------------------------------------------------------------------
# 9. Residual analysis
# -------------------------------------------------------------------

residuals = y - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Log Whole Weight")
plt.ylabel("Residuals")
plt.title("Residual Plot")
save_show("11_residual_plot.png")

print("\n--- Residual Statistics ---")
print("Residual Mean:", residuals.mean())
print("Residual Standard Deviation:", residuals.std())
print("Residual Variance:", residuals.var())


# -------------------------------------------------------------------
# 10. Normality analysis of residuals
# -------------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=30, density=True, alpha=0.6, edgecolor="black")

mu = residuals.mean()
sigma = residuals.std()
x_range = np.linspace(residuals.min(), residuals.max(), 100)
normal_curve = stats.norm.pdf(x_range, mu, sigma)

plt.plot(x_range, normal_curve, color="red")
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.title("Histogram of Residuals with Normal Curve")
save_show("12_residual_histogram_normal_curve.png")

plt.figure(figsize=(7, 5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals")
save_show("13_residual_qq_plot.png")


# -------------------------------------------------------------------
# 11. Independence test using ACF
# -------------------------------------------------------------------

def simple_acf(x: np.ndarray, max_lag: int) -> list[float]:
    """Calculate simple autocorrelation values from lag 0 to max_lag."""
    x = np.array(x)
    x_centered = x - np.mean(x)
    denominator = np.sum(x_centered**2)

    acf_values = []
    for lag in range(max_lag + 1):
        numerator = np.sum(x_centered[: len(x_centered) - lag] * x_centered[lag:])
        acf_values.append(numerator / denominator)

    return acf_values


acf_values = simple_acf(residuals.values, 30)

plt.figure(figsize=(8, 5))
plt.stem(range(31), acf_values)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("ACF of Residuals")
save_show("14_residual_acf.png")


# -------------------------------------------------------------------
# 12. Group segmentation by sex
# -------------------------------------------------------------------

df_clean.boxplot(column="Whole_weight", by="Sex")
plt.title("Whole Weight by Sex")
plt.suptitle("")
plt.xlabel("Sex")
plt.ylabel("Whole Weight")
save_show("15_weight_by_sex_boxplot.png")

plt.figure(figsize=(7, 5))

for sex_value in df_clean["Sex"].unique():
    subset = df_clean[df_clean["Sex"] == sex_value]
    plt.scatter(subset["Length"], subset["Whole_weight"], label=sex_value, alpha=0.5)

plt.xlabel("Length")
plt.ylabel("Whole Weight")
plt.title("Whole Weight vs Length by Sex")
plt.legend(title="Sex")
save_show("16_scatter_by_sex.png")


# -------------------------------------------------------------------
# 13. Correlation heatmap
# -------------------------------------------------------------------

df_corr = df_clean.copy()
df_corr["Sex_encoded"] = df_corr["Sex"].map({"M": 0, "F": 1, "I": 2})

corr = df_corr.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            round(corr.iloc[i, j], 2),
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )

plt.title("Correlation Heatmap of Full Feature Set")
save_show("17_correlation_heatmap.png")

print("\nAnalysis completed successfully.")
print(f"All figures were saved in the folder: {FIGURES_DIR}")
