import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data

df = pd.read_csv("data/diamonds.csv")

print("\nFirst 5 roes:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# basic statics

print("\nStatical Summary")
print(df.describe())

# visualization

# price distribution
plt.figure()
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Price Distribution")

# Carat vs Price
plt.figure()
sns.scatterplot(x="carat", y="price", data=df)
plt.title("Carat vs Price")
plt.show()

# Cut vs Price
plt.figure()
sns.boxplot(x="cut", y="price", data=df)
plt.title("Cut vs Price")
plt.show()

# Color vs Price
plt.figure()
sns.boxplot(x="color", y="price", data=df)
plt.title("Color vs Price")
plt.show()

# Correlation heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------
# 4. FEATURE INSIGHT
# -------------------------
print("\nInsights:")
print("1. Carat has strongest impact on price")
print("2. Cut has moderate impact")
print("3. Color has weak-medium impact")
print("4. Price increases as carat increases")
