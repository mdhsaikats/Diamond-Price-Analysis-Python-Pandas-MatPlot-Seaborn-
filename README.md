# 💎 Diamond Price Analysis Project

A comprehensive Python data analysis project that explores how different diamond features affect price using real-world data from the classic Diamonds dataset (~54,000 diamonds).

---

## 📌 Project Overview

This project performs **Exploratory Data Analysis (EDA)** on the Diamonds dataset to uncover pricing patterns and build a predictive model.

It includes:

- Data loading, cleaning, and exploration
- Statistical summaries and grouped aggregations
- Outlier detection using the IQR method
- 13 professional data visualizations (saved as PNG files)
- Correlation analysis with feature importance ranking
- Machine Learning model (Linear Regression) to predict diamond prices

---

## 📊 Dataset Information

The dataset contains **53,940 diamonds** with the following features:

| Feature   | Description                                          | Type        |
|-----------|------------------------------------------------------|-------------|
| `carat`   | Weight of the diamond                                | Numerical   |
| `cut`     | Quality of cut (Fair, Good, Very Good, Premium, Ideal) | Categorical |
| `color`   | Diamond color grading (D=Best → J=Worst)             | Categorical |
| `clarity` | Clarity grade (I1=Worst → IF=Best)                   | Categorical |
| `depth`   | Total depth percentage                               | Numerical   |
| `table`   | Width of the top of diamond relative to widest point | Numerical   |
| `price`   | Price in USD (target variable)                       | Numerical   |
| `x`       | Length in mm                                         | Numerical   |
| `y`       | Width in mm                                          | Numerical   |
| `z`       | Depth in mm                                          | Numerical   |

---

## 🛠️ Tech Stack

- **Python** 🐍 — Core language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Matplotlib** — Base plotting library
- **Seaborn** — Statistical data visualization
- **Scikit-learn** — Machine learning (Linear Regression)

---

## 📁 Project Structure

```
diamond-price-analysis/
│
├── data/
│   └── diamonds.csv          # Dataset (53,940 rows)
│
├── plots/                     # Auto-generated visualizations
│   ├── 01_price_distribution.png
│   ├── 02_carat_distribution.png
│   ├── 03_carat_vs_price.png
│   ├── 04_cut_vs_price.png
│   ├── 05_color_vs_price.png
│   ├── 06_clarity_vs_price.png
│   ├── 07_carat_vs_price_by_cut.png
│   ├── 08_violin_cut_vs_price.png
│   ├── 09_pairplot.png
│   ├── 10_categorical_distributions.png
│   ├── 11_correlation_heatmap.png
│   ├── 12_ml_results.png
│   └── 13_feature_importance.png
│
├── analysis.py                # Main analysis script
├── createcsv.py               # Dataset download utility
└── README.md
```

---

## 🚀 Features

### 📌 1. Data Exploration
- Load dataset using Pandas
- Check dataset structure and data types
- Identify missing values and duplicates
- First/last row preview

### 📌 2. Statistical Summary
- Descriptive statistics for numerical features
- Value counts and percentages for categorical features (Cut, Color, Clarity)

### 📌 3. Grouped Aggregations
- Average, median, min, max price by Cut, Color, and Clarity
- Cross-tabulation: Price by Cut × Color

### 📌 4. Outlier Detection
- IQR (Interquartile Range) method applied to all numerical features
- Reports outlier count, percentage, and valid range for each feature

### 📌 5. Data Visualization (13 Plots)
- Price distribution (original + log-transformed)
- Carat distribution
- Carat vs Price scatter plot with color gradient
- Boxplots: Cut, Color, Clarity vs Price
- Carat vs Price colored by Cut quality
- Violin plot: Cut vs Price
- Pairplot of key numerical features
- Categorical feature count distributions
- Correlation heatmap (triangular)
- ML: Actual vs Predicted + Residuals
- Feature importance bar chart

### 📌 6. Correlation Analysis
- Pearson correlation matrix
- Feature-by-feature correlation with price (ranked)
- Strength classification (Strong / Moderate / Weak)

### 📌 7. Machine Learning
- Linear Regression model
- 80/20 train-test split
- Evaluation: MAE, RMSE, R² Score
- Feature coefficients analysis
- Actual vs Predicted visualization

---

## 📈 Key Insights

| # | Insight |
|---|---------|
| 1 | **Carat** has the strongest correlation with price (r ≈ 0.92) |
| 2 | Diamond dimensions (x, y, z) are highly correlated with carat and price |
| 3 | "Ideal" cut diamonds have lower average prices because they tend to be smaller |
| 4 | Better color grades (D, E) don't always mean higher prices due to carat confounding |
| 5 | Price distribution is **right-skewed** — most diamonds are affordable |
| 6 | Linear Regression achieves **~85-90% R²** using only numerical features |

---

## ▶️ How to Run This Project

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Generate the dataset (if not present)

```bash
python createcsv.py
```

### 3. Run the analysis

```bash
python analysis.py
```

All visualizations will be saved to the `plots/` directory.

---

## 📊 Sample Outputs

This project generates:

- 📊 13 high-quality PNG visualizations in `plots/`
- 📈 Detailed console output with statistics and insights
- 🤖 ML model performance metrics

---

## 🧠 What I Learned

- Exploratory Data Analysis (EDA) workflow
- Data visualization best practices with Matplotlib & Seaborn
- Statistical analysis and outlier detection
- Feature correlation and importance analysis
- Building and evaluating a machine learning model
- Writing clean, well-documented Python code

---

## 🚀 Future Improvements

- [ ] Add interactive dashboard using Streamlit or Plotly
- [ ] Try advanced ML models (Random Forest, XGBoost)
- [ ] Add feature engineering (polynomial features, encoding categoricals)
- [ ] Perform hyperparameter tuning
- [ ] Deploy as a web application
- [ ] Create a Jupyter Notebook version for interactive exploration

---

## 👨‍💻 Author

Beginner Data Science Project using Python

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub and keep learning!
