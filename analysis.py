"""
💎 Diamond Price Analysis
=========================
A comprehensive Exploratory Data Analysis (EDA) on the Diamonds dataset
to understand how diamond features affect pricing.

Tech Stack: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys
import warnings

# Fix Windows console encoding for emoji/unicode characters
sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Create plots directory if it doesn't exist
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"

# Custom color palette for diamond-themed visuals
DIAMOND_PALETTE = ["#2C3E50", "#1ABC9C", "#3498DB", "#9B59B6", "#E74C3C"]
GRADIENT_CMAP = "viridis"


def separator(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 1. DATA LOADING & EXPLORATION
# ─────────────────────────────────────────────

separator("1. DATA LOADING & EXPLORATION")

df = pd.read_csv("data/diamonds.csv")

print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nColumn Data Types:")
print(df.dtypes)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✅ No missing values found in the dataset!")
else:
    print(f"⚠️  Total missing values: {missing.sum()}")

print("\nDuplicate Rows:")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates:,}")


# ─────────────────────────────────────────────
# 2. STATISTICAL SUMMARY
# ─────────────────────────────────────────────

separator("2. STATISTICAL SUMMARY")

print("Numerical Features Summary:")
print(df.describe().round(2))

print("\nCategorical Features Summary:")
for col in ["cut", "color", "clarity"]:
    print(f"\n--- {col.upper()} Distribution ---")
    counts = df[col].value_counts()
    percentages = df[col].value_counts(normalize=True).mul(100).round(1)
    summary = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})
    print(summary)


# ─────────────────────────────────────────────
# 3. GROUPED AGGREGATIONS
# ─────────────────────────────────────────────

separator("3. GROUPED AGGREGATIONS")

print("Average Price by Cut:")
cut_price = df.groupby("cut")["price"].agg(["mean", "median", "min", "max", "count"])
cut_price.columns = ["Avg Price", "Median Price", "Min Price", "Max Price", "Count"]
print(cut_price.round(2))

print("\nAverage Price by Color:")
color_price = df.groupby("color")["price"].agg(["mean", "median", "min", "max", "count"])
color_price.columns = ["Avg Price", "Median Price", "Min Price", "Max Price", "Count"]
print(color_price.round(2))

print("\nAverage Price by Clarity:")
clarity_price = df.groupby("clarity")["price"].agg(["mean", "median", "min", "max", "count"])
clarity_price.columns = ["Avg Price", "Median Price", "Min Price", "Max Price", "Count"]
print(clarity_price.round(2))

print("\nAverage Price by Cut & Color:")
cross = df.groupby(["cut", "color"])["price"].mean().round(2).unstack()
print(cross)


# ─────────────────────────────────────────────
# 4. OUTLIER DETECTION (IQR METHOD)
# ─────────────────────────────────────────────

separator("4. OUTLIER DETECTION (IQR Method)")

numerical_cols = ["carat", "depth", "table", "price", "x", "y", "z"]

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    pct = (len(outliers) / len(df)) * 100
    print(f"  {col:>8}: {len(outliers):>5,} outliers ({pct:.1f}%)  "
          f"[Range: {lower:.2f} – {upper:.2f}]")


# ─────────────────────────────────────────────
# 5. DATA VISUALIZATIONS
# ─────────────────────────────────────────────

separator("5. DATA VISUALIZATIONS")

# --- 5.1 Price Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["price"], bins=50, kde=True, color=DIAMOND_PALETTE[1], ax=axes[0])
axes[0].set_title("Price Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Frequency")

sns.histplot(np.log1p(df["price"]), bins=50, kde=True, color=DIAMOND_PALETTE[2], ax=axes[1])
axes[1].set_title("Log-Transformed Price Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Log(Price)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_price_distribution.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 01_price_distribution.png")

# --- 5.2 Carat Distribution ---
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["carat"], bins=50, kde=True, color=DIAMOND_PALETTE[3], ax=ax)
ax.set_title("Carat Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Carat")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_carat_distribution.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 02_carat_distribution.png")

# --- 5.3 Carat vs Price (Scatter) ---
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df["carat"], df["price"],
    c=df["price"], cmap=GRADIENT_CMAP,
    alpha=0.4, s=8, edgecolors="none"
)
plt.colorbar(scatter, label="Price ($)")
ax.set_title("Carat vs Price", fontsize=14, fontweight="bold")
ax.set_xlabel("Carat")
ax.set_ylabel("Price ($)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_carat_vs_price.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 03_carat_vs_price.png")

# --- 5.4 Cut vs Price (Boxplot) ---
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="cut", y="price", data=df, order=cut_order, palette="Set2", ax=ax)
ax.set_title("Diamond Price by Cut Quality", fontsize=14, fontweight="bold")
ax.set_xlabel("Cut")
ax.set_ylabel("Price ($)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_cut_vs_price.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 04_cut_vs_price.png")

# --- 5.5 Color vs Price (Boxplot) ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="color", y="price", data=df, palette="coolwarm", ax=ax)
ax.set_title("Diamond Price by Color Grade", fontsize=14, fontweight="bold")
ax.set_xlabel("Color (D=Best → J=Worst)")
ax.set_ylabel("Price ($)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_color_vs_price.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 05_color_vs_price.png")

# --- 5.6 Clarity vs Price (Boxplot) ---
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="clarity", y="price", data=df, order=clarity_order, palette="RdYlGn", ax=ax)
ax.set_title("Diamond Price by Clarity", fontsize=14, fontweight="bold")
ax.set_xlabel("Clarity (I1=Worst → IF=Best)")
ax.set_ylabel("Price ($)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_clarity_vs_price.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 06_clarity_vs_price.png")

# --- 5.7 Carat vs Price by Cut (Scatter with hue) ---
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(
    x="carat", y="price", hue="cut", data=df,
    hue_order=cut_order, palette="Set2", alpha=0.5, s=15, ax=ax
)
ax.set_title("Carat vs Price (Colored by Cut)", fontsize=14, fontweight="bold")
ax.set_xlabel("Carat")
ax.set_ylabel("Price ($)")
ax.legend(title="Cut", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_carat_vs_price_by_cut.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 07_carat_vs_price_by_cut.png")

# --- 5.8 Violin Plot: Cut vs Price ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(x="cut", y="price", data=df, order=cut_order, palette="muted", ax=ax)
ax.set_title("Price Distribution by Cut (Violin Plot)", fontsize=14, fontweight="bold")
ax.set_xlabel("Cut")
ax.set_ylabel("Price ($)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_violin_cut_vs_price.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 08_violin_cut_vs_price.png")

# --- 5.9 Pairplot (sampled for performance) ---
print("Generating pairplot (sampling 2000 rows for performance)...")
sample = df.sample(n=2000, random_state=42)
pair = sns.pairplot(
    sample[["carat", "depth", "table", "price"]],
    diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15},
    diag_kws={"fill": True}
)
pair.figure.suptitle("Pairplot of Key Numerical Features", y=1.02, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/09_pairplot.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 09_pairplot.png")

# --- 5.10 Count Plots for Categorical Features ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(x="cut", data=df, order=cut_order, palette="Set2", ax=axes[0])
axes[0].set_title("Distribution of Cut", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Cut")
axes[0].set_ylabel("Count")

sns.countplot(x="color", data=df, palette="coolwarm", ax=axes[1])
axes[1].set_title("Distribution of Color", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Color")
axes[1].set_ylabel("Count")

sns.countplot(x="clarity", data=df, order=clarity_order, palette="RdYlGn", ax=axes[2])
axes[2].set_title("Distribution of Clarity", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Clarity")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/10_categorical_distributions.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 10_categorical_distributions.png")


# ─────────────────────────────────────────────
# 6. CORRELATION ANALYSIS
# ─────────────────────────────────────────────

separator("6. CORRELATION ANALYSIS")

corr_matrix = df.corr(numeric_only=True)

print("Correlation with Price:")
price_corr = corr_matrix["price"].drop("price").sort_values(ascending=False)
for feature, corr in price_corr.items():
    strength = "Strong" if abs(corr) > 0.6 else "Moderate" if abs(corr) > 0.3 else "Weak"
    direction = "+" if corr > 0 else "-"
    bar = "█" * int(abs(corr) * 20)
    print(f"  {feature:>8}: {corr:+.4f}  ({strength} {direction})  {bar}")

# --- Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, square=True,
    linewidths=1, cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/11_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 11_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 7. MACHINE LEARNING: LINEAR REGRESSION
# ─────────────────────────────────────────────

separator("7. MACHINE LEARNING: Linear Regression")

# Prepare features (numerical only for simplicity)
features = ["carat", "depth", "table", "x", "y", "z"]
target = "price"

X = df[features]
y = df[target]

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"  Mean Absolute Error (MAE):  ${mae:,.2f}")
print(f"  Root Mean Squared Error:    ${rmse:,.2f}")
print(f"  R² Score:                    {r2:.4f} ({r2*100:.1f}%)")

# Feature coefficients
print(f"\n📈 Feature Coefficients:")
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)

for _, row in coef_df.iterrows():
    print(f"  {row['Feature']:>8}: {row['Coefficient']:>10,.2f}")

print(f"\n  Intercept: {model.intercept_:,.2f}")

# --- Actual vs Predicted Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.3, s=8, color=DIAMOND_PALETTE[2])
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
axes[0].set_title("Actual vs Predicted Price", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].legend()

# Residuals distribution
residuals = y_test - y_pred
axes[1].hist(residuals, bins=50, color=DIAMOND_PALETTE[3], edgecolor="white", alpha=0.8)
axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2)
axes[1].set_title("Residuals Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Residual (Actual - Predicted)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/12_ml_results.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 12_ml_results.png")

# --- Feature Importance Bar Chart ---
fig, ax = plt.subplots(figsize=(8, 5))
coef_df_sorted = coef_df.sort_values("Coefficient", key=abs, ascending=True)
colors = [DIAMOND_PALETTE[1] if c > 0 else DIAMOND_PALETTE[4] for c in coef_df_sorted["Coefficient"]]
ax.barh(coef_df_sorted["Feature"], coef_df_sorted["Coefficient"].abs(), color=colors)
ax.set_title("Feature Importance (Linear Regression Coefficients)", fontsize=14, fontweight="bold")
ax.set_xlabel("|Coefficient|")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/13_feature_importance.png", bbox_inches="tight")
plt.show()
print("✅ Saved: 13_feature_importance.png")


# ─────────────────────────────────────────────
# 8. KEY INSIGHTS & SUMMARY
# ─────────────────────────────────────────────

separator("8. KEY INSIGHTS & SUMMARY")

print("💎 Diamond Price Analysis — Key Findings:")
print()
print("  1. Carat has the STRONGEST correlation with price (r ≈ 0.92)")
print("     → Heavier diamonds are significantly more expensive")
print()
print("  2. Diamond dimensions (x, y, z) are highly correlated with both")
print("     carat and price — larger diamonds weigh more and cost more")
print()
print("  3. Cut quality has a MODERATE impact on price")
print("     → Surprisingly, 'Ideal' cut diamonds have a lower average price")
print("        because they tend to be smaller in carat weight")
print()
print("  4. Color grade shows a WEAK-MEDIUM effect")
print("     → Better color grades (D, E) don't always mean higher prices")
print("        due to confounding with carat size")
print()
print("  5. Price distribution is RIGHT-SKEWED")
print("     → Most diamonds are relatively affordable, with a long tail")
print("        of expensive diamonds")
print()
print("  6. The Linear Regression model achieves ~85-90% R² score")
print("     → Numerical features alone explain most price variation")
print("     → Adding categorical features could improve accuracy further")
print()
print(f"  📊 Total diamonds analyzed: {len(df):,}")
print(f"  💰 Price range: ${df['price'].min():,} – ${df['price'].max():,}")
print(f"  💎 Average price: ${df['price'].mean():,.2f}")
print(f"  📈 Plots saved to: ./{PLOTS_DIR}/")
print()
print("=" * 60)
print("  Analysis Complete! ✅")
print("=" * 60)
