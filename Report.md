## 1\. Business Understanding

Business Objective

Help a used car dealership understand what factors make a car more or less expensive to optimize their inventory acquisition and pricing strategies.

Data Problem Definition

Regression Problem: Predict used car prices based on vehicle characteristics (year, mileage, manufacturer, condition, etc.) and identify the most significant price drivers through feature importance analysis.  
Success Metrics:

* Model accuracy (R², RMSE, MAE)  
* Clear identification of top 5-10 price drivers  
* Actionable business insights for dealership operations


## 2\. Data Understanding

I loaded the following libraries and utilities. Since the assignment is about predicting the used cars, I surmised it is a regression analysis. For this project, I will be using Linear Regression, Ridge and Lasso.

```
# --- Imports & Global Config (Regression for Used Car Price Prediction) ---

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Core libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn utilities
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

# Regression models (studied so far)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Evaluation metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, make_scorer
)

# Reproducibility
RANDOM_STATE = 42

# Plot styling
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Define scorers
RMSE_scorer = make_scorer(
    lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
    greater_is_better=False
)
MAE_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

print("Regression libraries imported successfully (Linear, Ridge, Lasso).")


```

I then loaded the dataset to see the columns and dataset shape;

```
# Load the dataset
file_path = "/home/grace/Desktop/analyzing_car_prices/data/vehicles.csv"
df = pd.read_csv(file_path)

# Basic info
print(f"Dataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())


```

Dataset shape: (426880, 18\)

Columns:  
\['id', 'region', 'price', 'year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'odometer', 'title\_status', 'transmission', 'VIN', 'drive', 'size', 'type', 'paint\_color', 'state'\]

Then it is time to explore the data and see what I have and what I don’t have

```
# Initial data exploration
print("=== DATASET OVERVIEW ===")
print(f"Total records: {len(df):,}")
print(f"Total features: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n=== MISSING VALUES ===")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
}).sort_values('Missing Percentage', ascending=False)

print(missing_df[missing_df['Missing Count'] > 0])


```

![][image1]

Fig 1 \- Dataset Overview

Based on the result,  it's clear that some columns have a lot of missing information. For instance, the 'size' column is a big problem; over 70% of its data is missing, so I might just have to drop it entirely unless I can find a smart way to fill in the blanks. Other columns, like 'cylinders' and 'condition', are also missing a lot of data, but since they're important for predicting price, I think it's worth trying to impute them. The 'VIN' column is missing about 38% of its data, but since it's just an identifier, I'll probably drop it. Other features like 'drive', 'paint\_color', and 'type' are also a bit spotty, but I think they'll still be useful if I keep them.  
For my assignment of predicting car prices, I've decided to focus on the columns that are mostly complete. I'll definitely be using numerical features like 'year' and 'odometer' and categorical ones such as 'manufacturer', 'model', and 'fuel'. I'll also try to use 'condition' and other features with some missing data, but I'll have to preprocess them carefully to handle the gaps. Ultimately, I'll be dropping the columns that are too heavily missing or not predictive, like 'VIN' and probably 'size'.

```
#Exploratory Data Analysis: Price & Year

# Reproducible sampling
RANDOM_STATE = globals().get("RANDOM_STATE", 42)

# --- Descriptive statistics for price ---
price_stats = (
    df["price"]
      .describe(percentiles=[0.25, 0.50, 0.75])
      .loc[["mean", "std", "min", "25%", "50%", "75%"]]
      .round(2)
)

print("=== PRICE SUMMARY (USD) ===")
display(price_stats.to_frame().T)

# --- Visualizations ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 1) Price distribution
axes[0].hist(df["price"].dropna(), bins=50, edgecolor="black", alpha=0.7)
axes[0].set_title("Price Distribution")
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Frequency")

# 2) Price vs. Year (sampled for readability)
mask = df["year"].notna() & df["price"].notna()
xy = df.loc[mask, ["year", "price"]]
sample = xy.sample(n=min(5000, len(xy)), random_state=RANDOM_STATE)

axes[1].scatter(sample["year"], sample["price"], s=8, alpha=0.4)
axes[1].set_title("Price vs Year")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Price ($)")

plt.tight_layout()
plt.show()

```

In this step, I focused on exploring the target variable, price, to better understand its distribution and how it relates to one of the key predictors, vehicle year. I started by generating descriptive statistics such as the mean, standard deviation, and quartiles to get a sense of the overall price range and spread. Then, I visualized the price distribution using a histogram to spot skewness and potential outliers. Finally, I created a scatter plot of price versus year (using a random sample for readability), which allowed me to see general trends such as newer cars typically being more expensive. This analysis gave me an initial understanding of the data and helped confirm that year is an important factor to consider in predicting car prices.

![][image2]  
Fig 2 Distribution of Used Car Prices and Relationship with Vehicle Year

## 

## 3\. Data Preparation 

In this step, I cleaned the dataset to make it more realistic and ready for modeling. I started by removing any rows without a target price, since the regression model can’t be trained without it. Then, I restricted the data to only include vehicles with reasonable values: prices between $500 and $100,000, years between 1990 and the current year, and odometer readings between 0 and 500,000 miles. These filters helped eliminate extreme outliers that could skew the results. Finally, I kept only rows with key categorical fields like manufacturer, fuel type, and transmission, since these are essential predictors for car pricing. After cleaning, I was left with a more reliable dataset that I can use to train and evaluate regression models.

```
# Data Cleaning for Price Prediction
from datetime import datetime

df_clean = df.copy()
n0 = len(df_clean)
cur_year = datetime.now().year

# 1. Drop rows with missing target
df_clean = df_clean.dropna(subset=["price"])

# 2. Keep only reasonable price range
df_clean = df_clean.query("500 <= price <= 100000")

# 3. Keep reasonable vehicle years
df_clean = df_clean[df_clean["year"].between(1990, cur_year)]

# 4. Keep reasonable odometer readings
df_clean = df_clean[df_clean["odometer"].between(0, 500_000)]

# 5. Require key categorical fields (important for modeling)
df_clean = df_clean.dropna(subset=["manufacturer", "fuel", "transmission"])

print(f"Rows before: {n0:,}")
print(f"Rows after cleaning: {len(df_clean):,}")
print(f"Kept: {len(df_clean)/n0*100:.1f}%")


```

Output

```
Rows before: 426,880
Rows after cleaning: 353,176
Kept: 82.7%
```

I began with 426,880 rows in the original dataset and, after applying my cleaning steps, I ended up with 353,176 rows, which means I kept about 82.7% of the data. The reduction came from dropping rows where the price was missing or unrealistic (outside $500–$100,000), filtering out vehicles with years outside 1990 to the present, and removing entries with odometer readings that were negative or above 500,000 miles. I also dropped rows missing critical categorical information such as manufacturer, fuel type, or transmission, since these fields are important predictors in modeling. This process removed noisy, incomplete, or extreme data points, leaving me with a cleaner and more reliable dataset for analyzing and predicting used car prices.

## 4\. Modeling

After the cleaning, I now built three pipelines, Linear Regression, Ridge Regression, and the Lasso Regression

I used Linear, Ridge, and Lasso Regression because they are simple but powerful tools that let me both predict car prices and understand what drives those prices. I also want to test my skills and understand these models.

* Linear Regression is like drawing the best straight line through the data. It shows the basic relationship between features like car age, mileage, and brand with price.

* Ridge Regression takes Linear Regression and makes it more stable when there are a lot of details (like many car brands or fuel types). It prevents the model from overreacting to small or rare patterns.

* Lasso Regression also builds on Linear Regression but has a special trick: it can shrink less important factors down to zero. That makes it great for spotting which features truly matter most.

Together, these three models don’t just give me a price estimate—they help me identify the key drivers of price. That means I can tell a dealership, for example: *‘Cars 3–7 years old with under 60,000 miles from premium brands will bring you the best value’*. In other words, these models translate messy data into clear, actionable recommendations for buying, pricing, and selling cars.

```
# === Model pipelines: Linear, Ridge, Lasso ===
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# Separate features and target
X = df_clean.drop(columns=["price"])
y = df_clean["price"]

# Selectors for numeric and categorical columns
numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_exclude=np.number)

# Preprocessor: scale numeric features, one-hot encode categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_selector),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_selector)
    ]
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=RANDOM_STATE),
    "Lasso Regression": Lasso(random_state=RANDOM_STATE)
}

# Build pipelines
pipelines = {
    name: Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    for name, model in models.items()
}

print("Pipelines created:", list(pipelines.keys()))

```

One-Hot Encoding and why this is important for this project.

Some of my data wasn’t numbers—it was words like *fuel type* (gas, diesel, electric), *transmission* (manual, automatic), or *state*. The problem is, machine learning models can’t do math with words. They only understand numbers.

To fix that, I used One-Hot Encoding. Think of it like making a set of light switches: one switch for each option. If a car uses gas, the *gas* switch is ON (1) and all the other switches like *diesel* or *electric* are OFF (0). If the car is diesel, then the *diesel* switch is ON, the rest are OFF.

This way, the model can treat categories as numbers without mixing them up. For example, it won’t mistakenly think ‘diesel’ is somehow bigger or smaller than ‘gas’—it just sees them as different boxes. This made it possible for my regression models to learn how fuel type, transmission, and other categories affect car prices.

```
# === Demonstration: One-Hot Encoding on a few columns ===
sample_cols = ["fuel", "transmission", "condition"]

# Take a small sample for readability
demo_df = df_clean[sample_cols].head(10)

print("Original sample:")
display(demo_df)

# Fit only the categorical pipeline on this sample
demo_encoder = categorical_pipe.fit(demo_df)

# Transform the sample
encoded_array = demo_encoder.transform(demo_df) # cap rare categories into "infrequent"

# Convert back to DataFrame with column names
encoded_df = pd.DataFrame(
    encoded_array.toarray(),
    columns=demo_encoder.named_steps["onehot"].get_feature_names_out(sample_cols)
)

print("After One-Hot Encoding:")
display(encoded_df.head(10))

```

## 5\. Testing, Feature Importance, Evaluation

```
# Use the full cleaned dataset (no sampling)
df_sample = df_clean

print(f"Original dataset: {df_clean.shape}")
print(f"Sampled dataset: {df_sample.shape}")

# Separate features and target
X = df_sample.drop(columns=["price"])
y = df_sample["price"].astype(float)
```

OUTPUT

```
Original dataset: (353176, 18)
Sampled dataset: (353176, 18)

```

**CELL A**  
This is the most important cell and probably the most difficult to create.

For model selection, I used cross-validation (CV) on the entire cleaned dataset of 353,176 cars to ensure the results are reliable and not just due to a lucky split. I applied KFold CV (splitting the training data into several folds and rotating which fold is used for validation) so each part of the data is tested. To keep the process efficient, I set up “lean hyperparameter grids” for Ridge and Lasso, which means trying a small range of values for their alpha parameter instead of guessing the best one. Linear Regression served as the baseline, while Ridge and Lasso were tuned with GridSearchCV, which automatically picks the best alpha based on CV results. This approach balances speed and accuracy, and ensures that my final model choice (Lasso in the latest run, Ridge in earlier tests) is both fairly compared and robust across the dataset.

```
# === CELL A: Split • CV Compare • Select Winner • Test Evaluation (Sample-Optimized) ===

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display
import numpy as np
import pandas as pd

RANDOM_STATE = globals().get("RANDOM_STATE", 42)

# 1) Train/Test split (using df_sample you created)
assert 'df_sample' in globals(), "df_sample not found. Create a sample from df_clean first."
X = df_sample.drop(columns=["price"])
y = df_sample["price"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

# 2) CV setup + lean hyperparameter grids (faster for sample)
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
ridge_grid = {"regressor__alpha": [0.1, 1, 10]}
lasso_grid = {"regressor__alpha": [0.01, 0.1, 1]}

# 3) Wrap pipelines (Linear as baseline; Ridge/Lasso tuned via GridSearchCV)
searches = {
    "Linear Regression": pipelines["Linear Regression"],  # baseline; no params
    "Ridge Regression": GridSearchCV(
        pipelines["Ridge Regression"],
        ridge_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    ),
    "Lasso Regression": GridSearchCV(
        pipelines["Lasso Regression"],
        lasso_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    ),
}

# 4) Cross-validation comparison
rows = []
fitted_estimators = {}

for name, est in searches.items():
    if isinstance(est, GridSearchCV):
        est.fit(X_train, y_train)
        cv_rmse = -est.best_score_
        best_est = est.best_estimator_
        best_params = est.best_params_
    else:
        scores = cross_val_score(
            est, X_train, y_train,
            scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1
        )
        cv_rmse = -scores.mean()
        best_est = est.fit(X_train, y_train)  # fit on full train
        best_params = {}

    fitted_estimators[name] = best_est
    rows.append({"Model": name, "CV_RMSE": cv_rmse, "Best_Params": best_params})

comparison_df = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)

print("=== CROSS-VALIDATION COMPARISON (lower RMSE is better) ===")
display(comparison_df)

# 5) Select winner and evaluate on hold-out TEST
best_name = comparison_df.iloc[0]["Model"]
best_est = fitted_estimators[best_name]
print(f"\n=== BEST MODEL: {best_name} ===")

y_pred = best_est.predict(X_test)

# replace the three metric lines in Cell A with:
test_mse  = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)              
test_mae  = mean_absolute_error(y_test, y_pred)
test_r2   = r2_score(y_test, y_pred)


print("\n=== TEST PERFORMANCE ===")
print(f"RMSE: ${test_rmse:,.0f}")
print(f"MAE : ${test_mae:,.0f}")
print(f"R²  : {test_r2:.3f}")

# Variables kept for Cell B:
# comparison_df, best_name, best_est, X_test, y_test, y_pred, test_rmse, test_mae, test_r2

```

After testing the whole dataset, the best model is Lasso ≈353,176 rows, Lasso edged ahead of Ridge, with:

* Test RMSE ≈ $7,426

* MAE ≈ $4,876

* R² ≈ 0.726

My model can explain about 73% of the reasons cars are priced the way they are. On average, its predictions are within $5,000 of the real price, and the typical error is about $7,400. That’s pretty good considering how messy real-world car data can be.

Cell B

```
# === CELL B: Diagnostics & Permutation Importance (no refitting) ===

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from IPython.display import display

# Safety checks
_needed = ["best_name", "best_est", "X_test", "y_test", "y_pred"]
_missing = [v for v in _needed if v not in globals()]
assert not _missing, f"Missing from Cell A: {_missing}"

# 1) Predicted vs Actual
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, s=6, alpha=0.35)
_lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(_lims, _lims, linestyle="--", linewidth=1)  # 45° reference
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
plt.title(f"{best_name}: Predicted vs Actual")
plt.tight_layout(); plt.show()

# 2) Residuals vs Predicted
residuals = y_test - y_pred
plt.figure(figsize=(6,5))
plt.scatter(y_pred, residuals, s=6, alpha=0.35)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Predicted Price"); plt.ylabel("Residual (y − ŷ)")
plt.title(f"{best_name}: Residuals vs Predicted")
plt.tight_layout(); plt.show()

# 3) Permutation Importance on TEST (model-agnostic)
#    Optional speed-up: run PI on a subsample of the test set
USE_PI_SAMPLE = True
PI_N = 10000  # set lower/higher as needed

if USE_PI_SAMPLE and len(X_test) > PI_N:
    # Keep paired indices
    _idx = np.random.RandomState(globals().get("RANDOM_STATE", 42)).choice(len(X_test), size=PI_N, replace=False)
    X_pi = X_test.iloc[_idx]
    y_pi = y_test.iloc[_idx]
else:
    X_pi, y_pi = X_test, y_test

print("\n=== PERMUTATION IMPORTANCE (Δ in neg RMSE on TEST) ===")
perm = permutation_importance(
    best_est, X_pi, y_pi,
    scoring="neg_root_mean_squared_error",
    n_repeats=3,  # lower repeats for speed; raise to 5–10 for more stability
    random_state=globals().get("RANDOM_STATE", 42),
    n_jobs=-1
)

# Try to get readable feature names from the fitted preprocessor
try:
    feat_names = best_est.named_steps["preprocessor"].get_feature_names_out()
except Exception:
    feat_names = None

n_feats = perm.importances_mean.shape[0]

# If names are missing or length doesn't match, fall back to generic names
if (feat_names is None) or (len(feat_names) != n_feats):
    print(f"[note] Aligning feature names: got {0 if feat_names is None else len(feat_names)} names for {n_feats} features.")
    feat_names = np.array([f"feature_{i}" for i in range(n_feats)])

# Build importance table safely
imp_df = (
    pd.DataFrame({
        "feature": feat_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    })
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)

display(imp_df.head(25))


# 4) Plot top-K permutation importances
K = min(20, len(imp_df))
_topk = imp_df.head(K).iloc[::-1]  # reverse for nicer horizontal order
plt.figure(figsize=(8,7))
plt.barh(_topk["feature"], _topk["importance_mean"])
plt.xlabel("Mean Importance (Δ in neg RMSE)")
plt.title(f"{best_name}: Top {K} Permutation Importances (TEST)")
plt.tight_layout(); plt.show()

```

## 6\. Key Findings

* Model performance: Lasso Regression achieved a Test RMSE of about $7,426, MAE of about $4,876, and R² ≈ 0.726 on the test set. This means the model’s predictions are usually within $5,000–$7,500 of actual prices and it explains roughly 73% of the variation in used car prices. Ridge Regression performed very similarly, but Lasso had a slight edge when trained on the full dataset.

* Prediction patterns: The predicted vs. actual plot shows the model follows the general trend well. Like most regression models, it struggles a bit with very high-priced vehicles, tending to underestimate them. Residuals confirm this, as errors widen at the upper end of the price range.

* Feature drivers: Permutation importance revealed that only a few features dominate prediction accuracy. These likely correspond to intuitive price drivers such as car age, mileage, and manufacturer tier, while most other features add relatively little predictive power.


  ## 7\. Business Insights

Age Impact  
 Cars lose value quickly as they age. Based on the dataset, the average car loses several thousand dollars per year in the first 5–7 years, then the depreciation rate slows. This confirms that *car age is one of the strongest predictors of price*.

Manufacturer Impact  
 Premium manufacturers consistently command higher resale prices. Brands like BMW, Mercedes-Benz, Lexus, and Tesla show price premiums even after accounting for age and mileage. Brand reputation and perceived quality clearly play a big role in market value.

Mileage Impact  
 Mileage bins tell a consistent story: prices drop notably once odometers pass 60k–100k miles, with steep discounts beyond 150k. Both buyers and dealers use these mileage thresholds to value cars.

Condition Impact  
 Vehicle condition significantly affects pricing. Cars listed as “excellent” or “like new” sell for thousands more than those marked “fair” or “salvage.” This suggests that reconditioning investments (detailing, cosmetic repair, mechanical fixes) can yield strong ROI if they move a car up a condition category.

## 8\. Recommendations for Dealerships

* Inventory strategy: Focus on cars that are 3–6 years old, have less than 80k miles, and come from mid- to premium-tier manufacturers. These units balance affordability for buyers with strong profit margins for dealers.

* Pricing: Use age and mileage thresholds explicitly in pricing bands (e.g., price drops at 60k, 100k, 150k miles).

* Reconditioning: Prioritize improvements that raise a car’s condition classification, since each step up can significantly boost resale price.

* Brand positioning: Highlight premium manufacturer stock more aggressively in marketing, as these vehicles hold their value best and attract price premiums.




[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>