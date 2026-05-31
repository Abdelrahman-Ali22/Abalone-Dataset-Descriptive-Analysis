import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv("abalone.data", header=None)

df.columns = ["Sex", "Length", "Diameter", "Height",
              "Whole_weight", "Shucked_weight",
              "Viscera_weight", "Shell_weight", "Rings"]

print("Original Dataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


plt.figure(figsize=(10, 4))

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

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))

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

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.boxplot(df["Length"])
plt.title("Boxplot of Length")
plt.ylabel("Length")

plt.subplot(1, 2, 2)
plt.boxplot(df["Whole_weight"])
plt.title("Boxplot of Whole Weight")
plt.ylabel("Whole Weight")

plt.tight_layout()
plt.show()

def get_descriptive_stats(data, name):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    print(f"\n--- {name} Descriptive Statistics ---")
    print(f"Min: {data.min()}")
    print(f"Q1: {q1}")
    print(f"Median: {data.median()}")
    print(f"Q3: {q3}")
    print(f"Max: {data.max()}")
    print(f"IQR: {iqr}")


get_descriptive_stats(df["Length"], "Length")
get_descriptive_stats(df["Whole_weight"], "Whole Weight")


def get_moments(data, name):
    print(f"\n--- {name} Moment-Based Statistics ---")
    print(f"Mean: {data.mean()}")
    print(f"Standard Deviation: {data.std()}")
    print(f"Variance: {data.var()}")
    print(f"Skewness: {data.skew()}")
    print(f"Kurtosis: {data.kurt()}")


get_moments(df["Length"], "Length")
get_moments(df["Whole_weight"], "Whole Weight")


plt.figure()
plt.scatter(df["Length"], df["Whole_weight"], alpha=0.5)
plt.xlabel("Length")
plt.ylabel("Whole Weight")
plt.title("Whole Weight vs Length")
plt.show()


# Outlier limits for Length
Q1_L = df["Length"].quantile(0.25)
Q3_L = df["Length"].quantile(0.75)
IQR_L = Q3_L - Q1_L

lower_L = Q1_L - 1.5 * IQR_L
upper_L = Q3_L + 1.5 * IQR_L

# Outlier limits for Whole_weight
Q1_W = df["Whole_weight"].quantile(0.25)
Q3_W = df["Whole_weight"].quantile(0.75)
IQR_W = Q3_W - Q1_W

lower_W = Q1_W - 1.5 * IQR_W
upper_W = Q3_W + 1.5 * IQR_W

# Find outliers in either Length or Whole_weight
outliers = df[
    (df["Length"] < lower_L) | (df["Length"] > upper_L) |
    (df["Whole_weight"] < lower_W) | (df["Whole_weight"] > upper_W)
]

print("\nNumber of Outliers:")
print(len(outliers))

print("\nOutlier Rows:")
print(outliers)

plt.figure(figsize=(10, 4))

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

plt.tight_layout()
plt.show()


plt.figure()
plt.scatter(df["Length"], df["Whole_weight"], label="Normal Data", alpha=0.3)
plt.scatter(outliers["Length"], outliers["Whole_weight"],
            color="red", label="Outliers")
plt.xlabel("Length")
plt.ylabel("Whole Weight")
plt.title("Outliers on Scatterplot")
plt.legend()
plt.show()

df_clean = df.drop(outliers.index)

print("\nClean Dataset Shape After Removing Outliers:")
print(df_clean.shape)


clean_stats = df_clean[["Length", "Whole_weight"]].agg([
    "min",
    "max",
    "median",
    "mean",
    "std",
    "var",
    "skew",
    "kurt"
])

print("\n--- Second Statistics Table After Removing Outliers ---")
print(clean_stats)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.boxplot(df_clean["Length"])
plt.title("Boxplot of Length After Outlier Removal")
plt.ylabel("Length")

plt.subplot(1, 2, 2)
plt.boxplot(df_clean["Whole_weight"])
plt.title("Boxplot of Whole Weight After Outlier Removal")
plt.ylabel("Whole Weight")

plt.tight_layout()
plt.show()


plt.figure()
plt.scatter(df_clean["Length"], df_clean["Whole_weight"], alpha=0.5)
plt.xlabel("Length")
plt.ylabel("Whole Weight")
plt.title("Whole Weight vs Length After Outlier Removal")
plt.show()


df_clean = df_clean[
    (df_clean["Length"] > 0) &
    (df_clean["Whole_weight"] > 0)
]

df_clean["log_length"] = np.log(df_clean["Length"])
df_clean["log_weight"] = np.log(df_clean["Whole_weight"])

plt.figure()
plt.scatter(df_clean["log_length"], df_clean["log_weight"], alpha=0.5)
plt.xlabel("Log Length")
plt.ylabel("Log Whole Weight")
plt.title("Log-Log Transformation")
plt.show()


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


# Regression line plot
plt.figure()
plt.scatter(df_clean["log_length"], df_clean["log_weight"], alpha=0.4, label="Data")
plt.plot(df_clean["log_length"], y_pred, color="red", label="Regression Line")
plt.xlabel("Log Length")
plt.ylabel("Log Whole Weight")
plt.title("Linear Regression on Log-Log Transformed Data")
plt.legend()
plt.show()


residuals = y - y_pred

plt.figure()
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Log Whole Weight")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


print("\n--- Residual Statistics ---")
print("Residual Mean:", residuals.mean())
print("Residual Standard Deviation:", residuals.std())
print("Residual Variance:", residuals.var())


plt.figure()

plt.hist(residuals, bins=30, density=True, alpha=0.6, edgecolor="black")

mu = residuals.mean()
sigma = residuals.std()

x_range = np.linspace(residuals.min(), residuals.max(), 100)

normal_curve = (
    1 / (sigma * np.sqrt(2 * np.pi))
) * np.exp(
    -0.5 * ((x_range - mu) / sigma) ** 2
)

plt.plot(x_range, normal_curve, color="red")
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.title("Histogram of Residuals with Normal Curve")
plt.show()


# QQ plot
plt.figure()
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals")
plt.show()


def simple_acf(x, max_lag):
    x = np.array(x)
    x_centered = x - np.mean(x)

    acf_values = []

    for lag in range(max_lag + 1):
        numerator = np.sum(
            x_centered[:len(x_centered) - lag] *
            x_centered[lag:]
        )

        denominator = np.sum(x_centered ** 2)

        acf_values.append(numerator / denominator)

    return acf_values


acf_values = simple_acf(residuals.values, 30)

plt.figure()
plt.stem(range(31), acf_values)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("ACF of Residuals")
plt.show()


df_clean.boxplot(column="Whole_weight", by="Sex")
plt.title("Whole Weight by Sex")
plt.suptitle("")
plt.xlabel("Sex")
plt.ylabel("Whole Weight")
plt.show()


# Scatterplot colored by sex
plt.figure()

for sex_value in df_clean["Sex"].unique():
    subset = df_clean[df_clean["Sex"] == sex_value]
    plt.scatter(subset["Length"], subset["Whole_weight"],
                label=sex_value, alpha=0.5)

plt.xlabel("Length")
plt.ylabel("Whole Weight")
plt.title("Whole Weight vs Length by Sex")
plt.legend(title="Sex")
plt.show()

df_corr = df_clean.copy()

df_corr["Sex_encoded"] = df_corr["Sex"].map({
    "M": 0,
    "F": 1,
    "I": 2
})

corr = df_corr.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

# Add correlation numbers inside the heatmap boxes
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            round(corr.iloc[i, j], 2),
            ha="center",
            va="center",
            color="black",
            fontsize=8
        )

plt.title("Correlation Heatmap of Full Feature Set")
plt.tight_layout()
plt.show()
