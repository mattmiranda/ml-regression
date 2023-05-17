# ml-regression
Goal: Train and validate regression model to predict kimchi prices.

## Instructions
1. Create a python virtual environment and activate it
```
python -m virtualenv .venv
source .venv/bin/activate
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run python script or jupyter notebook
```
python main.py
```


## Summary
* I decided to use scikit-learn because of its easy and consistent interface. sklearn also has many very handy preprocessing functions. I also particularly like using sklearn pipelines. 
* After loading the data into a pandas DataFrame, it was possible to see that it had null or na values.
* Since there are only 5 missing values, which is a small portion of the data, I decided to drop the rows with missing values.
* Then I wanted to see the distribution of the data, so I plotted distribution and box plots. 
* It's possible to see from the plots that the data is right skewed.
* I investigated methods of detecting and removing outliers such as IQR with trimming or capping. 
* However since the dataset is quite small I decided not to trim it. 
* A transformer will be used to deal with the skewed data.
* Comparing QuantileTransformer and PowerTransformer, QuantileTransformer does a much better job normalizing the data into a normal distribution. Which should help in the training.
* Then I do some feature engineering, I extract the year, month, dayofweek and season from the "Date" column. I did it because there could be a correlation between price and time of the year (due to availability/supply).
* I found out that every date corresponds to a Sunday and the year is fixed as 2018. So I dropped those columns as they are not useful. 
* By plotting a correlation heatmap I could notice correlation between volume and Boxes_S, probably because most of the volume comes from small boxes. I also found that the Price is slightly correlated to Boxes_XL, probably because larger boxes are more expensive.
* There is also some correlation between region and price. Kimchi is sold at different prices in different regions.
* Because there are not too many features I decided not to use PCA for this dataset.
* The dataset has a categorical variables (Region) that needs to be encoded. I use one hot encoding for that.
* In order to scale/normalize the numerical features (such as Volume, Boxes_S, Boxes_L and Boxes_XL):
    * Instantiate OneHotEncoder and the normalizer (I tested StandardNormalizer and QuantileTransformer)
    * Make column transformer specifying the transform and target columns.
    * Then I could simply use make_pipeline passing the column transformer and the model to be trained.
* run_models is a helper function I created to fit (train) and evaluate the performance of several models, by calculating and returning the mean absolute error and root mean squared error.
* As you can see from the output table below, the result indicates that a Support Vector Machine (SVM) seems to perform best with this dataset. Even though SVM with RBF kernel has a fit time complexity more than quadratic, for a small dataset such as this it's not a problem. However if the dataset was bigger, I'd consider switching to a linear model such as SVMLinear or SGDRegressor.

|           | MAE       | RMSE       |
|-----------|-----------|------------|
| LinearReg | 16.655392 | 43.831407  |
| GradBoost | 46.800856 | 373.293755 |
| RandFor   | 36.677619 | 263.342472 |
| DecTree   | 7.421035  | 45.788963  |
| KNN       | 14.366465 | 91.609868  |
| SVM       | 0.121549  | 0.165607   |
| SGD       | 13.178861 | 26.666739  |

* Next steps are SVM hyperparameter tuning. 