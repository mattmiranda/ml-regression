import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import tree
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set(color_codes=True)

# Load dataset into a pandas dataframe
df = pd.read_excel("./data/kimchi_dataset.xlsx")
print("DataFrame: ", df)

# Checking data types in dataset
print("Data types: ", df.dtypes)

# Rename columns to make it easier to access
df = df.rename(
    columns={
        "Total Volume": "Volume",
        "Total Boxes": "Boxes_T",
        "Small Boxes": "Boxes_S",
        "Large Boxes": "Boxes_L",
        "XLarge Boxes": "Boxes_XL",
    }
)

# Checking for duplicate rows
duplicate_rows_df = df[df.duplicated()]
print("number of duplicated rows: ", duplicate_rows_df.shape[0])

# Checking that number of items match for all columns
print("Checking item count: ", df.count())

# Handling missing values (null or na)
print("number of missing values: ", df.isnull().sum())

# There are 4 missing values for price and 1 for volume. Since 5 is a small portion of the data I will simply drop the rows with missing values.
df = df.dropna()
print("new count: ", df.count())

print(
    "Now that the dataset is clean (no missing values) we can continue with EDA by looking for outliers and skewed data."
)

# Plot distribution
plt.figure(figsize=[16, 8])
plt.subplot(2, 2, 1)
sns.distplot(df["Price"])

plt.subplot(2, 2, 2)
sns.distplot(df["Volume"])

plt.subplot(2, 2, 3)
sns.distplot(df["Boxes_S"])

plt.subplot(2, 2, 4)
sns.distplot(df["Boxes_L"])
plt.title("Distribution Plots")

plt.show()

# Convert date column into separate features, "month" and "season"
df.Date = pd.to_datetime(df.Date)
df["month"] = df.Date.dt.month
df["season"] = df.apply(
    lambda x: 0
    if x["month"] in [1, 2, 12]
    else (1 if x["month"] in [3, 4, 5] else (2 if x["month"] in [6, 7, 8] else 4)),
    axis=1,
)
df = df.drop(["Date", "Boxes_T"], axis=1)

X = df.drop(["Price"], axis=1)
y = df["Price"]

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43
)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)


def run_models(models, X_train, y_train, X_test, y_test):
    mae = []
    rmse = []
    names = []
    for item in models:
        pipeline = item["pipeline"]
        names.append(item["name"])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mae.append(mean_absolute_error(y_true=y_test, y_pred=predictions))
        rmse.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions)))
    col = {"MAE": mae, "RMSE": rmse}
    ret_df = pd.DataFrame(data=col, index=names)
    return ret_df


# One hot encoding to encode nominal variabe "Region"
one_hot_enc = OneHotEncoder(sparse=False)

# Normalize other numerical features with QuantileTransformer
quant_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal")

# Make column transformer to make it easier to create scikit-learn pipeline later
col_trans = make_column_transformer(
    (one_hot_enc, ["Region"]),
    (quant_trans, ["Volume", "Boxes_S", "Boxes_L", "Boxes_XL"]),
)

# Create pipelines from all models

models = []

# Instantiate pipeline with linear regression
lm = LinearRegression()
lm_pipeline = make_pipeline(col_trans, lm)
models.append({"name": "LinearReg", "pipeline": lm_pipeline})

# Instantiate Gradient Boosting Regressor with default loss = 'squared_error' and default criterion = 'friedman_mse'
gbm = GradientBoostingRegressor()
gbm_pipeline = make_pipeline(col_trans, gbm)
models.append({"name": "GradBoost", "pipeline": gbm_pipeline})

# Deault criterion for Random Forrest is 'squared_error'
rfm = RandomForestRegressor()
rfm_pipeline = make_pipeline(col_trans, rfm)
models.append({"name": "RandFor", "pipeline": rfm_pipeline})

# Decision Tree
dtm = tree.DecisionTreeRegressor(max_depth=1)
dtm_pipeline = make_pipeline(col_trans, dtm)
models.append({"name": "DecTree", "pipeline": dtm_pipeline})

# KNN
knn = neighbors.KNeighborsRegressor(n_neighbors=5, weights="uniform")
knn_pipeline = make_pipeline(col_trans, knn)
models.append({"name": "KNN", "pipeline": knn_pipeline})

# SVM
svm = SVR()
svm_pipeline = make_pipeline(col_trans, svm)
models.append({"name": "SVM", "pipeline": svm_pipeline})

results = run_models(models, X_train, y_train, X_test, y_test)
print("Result comparison between models using Quantile Transform")
print(results)
