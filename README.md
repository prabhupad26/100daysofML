# Welcome to my 100days of Machine learning

### I will be randomly picking up datasets from the web and will be building a machine learning model using different types of Algorithms and build something cool.



## _DAY1_ : _Boston house price prediction model_
> Data obtained from [kaggle](https://www.kaggle.com/samratp/boston-housing-prices-evaluation-validation/data)
> Kaggle Notebook can be viewed [here](https://www.kaggle.com/prabhupad26/boston-house-prices-prediction)

* EDA (Exploratory data analysis) Observations and Processing data for better results  :
1.  Correlation between the features (13 features in the dataset) suggests that there's a mix of positive and negative correlated features are present in the dataset, 
    * In order to avoid the multicollinearity we need to remove either of the feature from a set of highly correlated feature. I tried removing TAX column as `TAX` and `RAD` gave         the highest correlation of 0.91 , simlarly removed `DIS` column as `DIS` and `AGE` gave highly negatively correlation of -0.75.
    * Correlation of `CHAS` is close to 0 i.e. `0.18` which means there is no correlation of `CHAS` to the target variable.
3.  Correlation between the remaining features to the target variable `MEDV`  suggests that (+)`RM`, (+)`LSTAT`, (-)`PTRATIO` are highly (positively or negatively) correlated with the target variable, so these features should be sufficient for predicting the target variable.
4.  Boxplot suggests that there is a lot of outliers in columns : `CRIM, ZN, RM, DIS, B, LSTAT`, since linear regression could be very sensitive towards the outliers so having too much outliers might result in poor prediction. I have tried diagnosing that problem in following ways :
    * `CRIM,ZN,RM,B,PTRATIO,LSTAT,MEDV` found to have some outliers.
    * Removed the entire record which is having extreme outlier, since removing all the outliers is not a good idea as it may cause biased results due to smaller dataset (we only have ~500 records for our experiment).
    * Replace the rest of the outliers with the mean of that feature.
    * _Need to tryout other approaches like : Z-score, Trimming the outliers, MW U-Test, Robust statistics, bootstraping to find out if its a useful outlier or not._

* Creating train and test data :
Since I was only able to obtain ~500 records with 10 relevant features so I'm splitting the data in 80/20 ratio and before doing that I have shuffled the data so that is doesn't create any bias (which will lead to bias problem or under fitting problem). 

* Training and evaluating the ML model with different learning algorithms :
1. Ordinary Least Square (OLS) : This is a good algorithm to start with for getting an understanding about how well a linear model is able to fit the data available, this is also know for it speed and performance with less data. So with the shuffled training set I got an r^2 score of ~ 0.65, from the learning curve it seems the the model converges very early (at training size 100 - 150). Other metrics like MSE, RMSE, MAE, R-squared scores are not good using the testing data.
2. Support vectors Regression (SVR) : Using the linear kernel basis function this model performed very poorly, reason might be due to the large number of features used and less number of records. This might improve on using PCA or using more training data. Other metrics like MSE, RMSE, MAE, R-squared scores are not good using the testing data.
3. Random Forest Regression : Using 100 as the number of estimators this model preformed best with an r^2 score of ~0.96, so this model outperformed other learning models as well. Other metrics like MSE, RMSE, MAE, R-squared scores are better than the other models using the testing data.
4. K-Nearest-Neighbor : Using 10 Number of neighbours gave ~ 0.64 as r^2 score on the training data, even on increasing the number of neightbors it worse results. Other metrics like MSE, RMSE, MAE, R-squared scores are not good using the testing data.

**Conclusion :**
While there are other algorithms which I would like to try out, but for now I'll conclude that the Random forest regressor is by far the best model which perfectly fits the linear curve on the data and predicts accurate MEDV for the given set of features. 
