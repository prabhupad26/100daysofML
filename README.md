# Welcome to my 100days of Machine learning

### I will be randomly picking up datasets from the web and will be building a machine learning model using different types of Algorithms and build something cool.



## _DAY1_ : _Boston house price prediction model_
> Data obtained from [kaggle](https://www.kaggle.com/samratp/boston-housing-prices-evaluation-validation/data)

> Kaggle Notebook can be viewed [here](https://www.kaggle.com/prabhupad26/boston-house-prices-prediction)

> [Link](https://github.com/prabhupad26/100daysofML/blob/main/boston_house_prices_prediction.ipynb) to python notebook .

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


## _DAY2_ : _Titanic survival probability prediction_
> Problem statement and Data obtained from [Kaggle](https://www.kaggle.com/c/titanic/data)

> [Link](https://github.com/prabhupad26/100daysofML/blob/main/titanic_classification_problem.ipynb) to python notebook.

* EDA (Exploratory data analysis) Observations and Processing data for better results  :
1. Null values found in :
      * 687 rows with no Cabin data : Column will be removed as it doesn't much help in predicting `Survival`. 
      * 2 rows with no Embarked data : Missing data will be replaced with 0 while converting the text data to numeric data.
      * 177 rows with no Age data : Missing data will be replaced with the median of Age for every possible combination of `Pclass` and `Gender`.
2. Most passenger who survived are Female.
3. Most passenger who didn't survived lies in the age between 15 and 45, creating Age band will give more insight.
4. Missing Age data is replace with median of `Age` data for 6 possible combinations of `Pclass` and `Gender`
5. Box plots for `Embarked`/`Pclass`, `Age` and `Survived` shows that there are few outliers for Age param in 3 class for Q-embarked passengers , this will get rectified once Age Band are created.
6. New feature `IsAlone` is crafted with the help of `Parch` and `SibSp` with the assumption that the passenger has boarded alone if he/she is not having any family or child or parent(`Parch`, `SibSp` is `0`).
8. Fare band is also created same way as Age Band.


* Creating train and test data :
Since only 891 records are available with 9 relevant (1 handcrafted feature) features so I'm splitting the data in 80/20 ratio and before doing that I have shuffled the data so that is doesn't create any bias (which will lead to bias problem or under fitting problem). 

* Training and evaluating the ML model with different learning algorithms :
Here is the summary of all the algorithm trained and tested :
![image](https://user-images.githubusercontent.com/11462012/128045540-7cbd4640-d1e3-41d3-ad4d-758e63882b2f.png)

> * K-Nearest Neighbour seems to perform better on the test data.


## _DAY3_ : _NewsGroup data classification_
> Data obtained from [SKlearn inbuilt dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

> [Link](https://github.com/prabhupad26/100daysofML/blob/main/titanic_classification_problem.ipynb) to python notebok

In this execise the attempt is to classify a story into one of 20 different news categories, the dataset consist of 18000 newsgroups posts on 20 topics

* EDA (Exploratory data analysis) Observations and Processing data for better results  :
1. Created a data frame which will include all the data obtained from `sklearn.datasets.fetch_20newsgroups`.
2. Each data is a email thread with a mail subject line.
3. Average number of email threads for every news category is ~500-600.
4. I have performed the below preprocessing steps :
      * Removed stopwords ([nltk](https://www.nltk.org/_modules/nltk/corpus.html)).
      * Removed email addresses and special characters using regex.
      * Trimmed each mail thread and lowercased.
5. Created a wordcloud to see the density distribution of words in the dataset.
6. Created a vector representation using the TF-IDF scores  of the entire dataset.


* Training and evaluating the ML model with Multinomial NaiveBayes algorithm :

For training the model the vector representation is used which was created earlier.  It gives an accuracy of 0.81 on the training data, and same accuracy score for test data.

* Model Evaluation :
1. Confusion matrix shows a very good results , shows very few False Positives and False Negatives.
2. ROC Curve shows for every news category the AUC is around 0.90 - 1.0 
3. From the PR curve it looks there are chances of class imbalance for those categories which has lower Area Under the Curve, so the model could be a bit biased towards the other categories which has greater AUC. This could be avoided by including more data for those categories with less AUC.


## _DAY4_ : _Predicting handwritten digits using MNIST dataset with Pytorch framework_
In this exercise I'll be using the pytorch framework to train 2 fully connected neural network (Linear layers) to learn to predict the handwritten digit given a 28 * 28 dimension greyscale image obtained from torchvision inbuild API.

* Data creation for the ML model
   1. Downloaded dataset from `torchvision.dataset.MNIST`.
   2. Applied `torchvision.transforms` to convert the downloaded data to its tensor form and normalized using mean = 0.1307 and standard deviation =0.3081
   3. Created a training set and validation set (testing set gets downloaded using `torchvision.datasets.MNIST`) using an utility function - `torch.utils.data.sampler.SubsetRandomSampler`.
   4. Visualized the data to check if the data and the labels are correct.

* Neural network definition :
   1. Defined 2 Fully connnected layers with dimensions : `28*28 x 512 and 512 x 10` with dropouts of 0.2
   2. The forward pass function is defined in the following manner :
         Fully connected Layer 1 ((batch_size) x 28*28) --> ReLU Activation + Dropouts--> Fully connected Layer 2 (512 x 10) --> ReLU Activation + Dropouts --> output((batch size x 10)) [The final output are the predicted handwritten digits].
   3. Error function used : CrossEntropy Error  (`torch.nn.CrossEntropyLoss`) , Optimizer / Objective function used : Stocastic gradient descent (`torch.optim.SGD`).
   4. Below are the hyperparameter used and experimented:
         - learning rate = 0.001
         - epochs = 20
         - probability of dropping the neuron values per layer = 0.2
         - hidden unit in each fully connected layer = 512


* Model Evaluation :
Cross entropy loss is used for calculating the output error :
1. Training error , Validation error reported as : (0.217226, 0.213741) :

   ![image](https://user-images.githubusercontent.com/11462012/128593014-1a5bc7c0-62ae-4fee-972e-0304c9342c5d.png)
2. Training error reported as `0.208873`
3. Accuracy % (Number of correct prediction/ Total predictions made) per digits 0-9 :

   ![image](https://user-images.githubusercontent.com/11462012/128592955-5e3dbe4c-4d80-44a4-aeb3-c285aa4fa9c3.png)
4. Finally the visualization of the predictions made :
   ![image](https://user-images.githubusercontent.com/11462012/128592974-ae069268-286e-4a2a-b596-40a00a3e1815.png)


## _DAY5_ : _Car Price Prediction (Kaggle competition)_

> Kaggle Notebook can be viewed [here](https://www.kaggle.com/mohaiminul101/car-price-prediction)

* Given a dataset of car prices for old cars along with features like year, engine, mileage, seater, etc. the trained model should make prediction of car prices using these features.
* EDA (Exploratory data analysis) Observations and Processing data for better results  :
   1. Converted categorical text data to numeric data for `mileage, engine, max_power, fuel, seller_type, transmission, owner` columns.
   2. Created new feature `years_old` which tells us about how old the car is.
   3. Removed `name, torque, year` as it doesn't have any useful numeric (there might be a possibility to convert torque to power need to check this) data.
   4. There are ~200 rows missing data for few columns, dropping those data.
   5. `seats` has very less correlation (almost 0) with the selling price, removing that column.
 
* Training and evaluating the ML model with different learning algorithms :
   1. Random forest regression shows the best accuracy score of ~0.97 on the testing data and ~0.98 on the training data (`number of estimators used : 100`).
   2. Linear regression didn't perform well on this data. (Accuracy score : ~0.67)
   3. On removing the skewness in the data using its log values (for `selling_price`, `km_driven`, `years_old`) the linear regression performance improved by 0.84 on train data and 0.85 on testing data.


## _DAY6_ : _Transfer Learning (Part 1) : Classify image of flowers using VGG16NET pretrained model_

In this exercise I'll be retraining a neural network on a pytorch framework. Ths trained pytorch model used is VGG16 NET  having a total number of parameters = `138357544`
out of which I'll be training only ~85k parameters after replacing  the last layer with a fully connected layer with 5 outputs(for training 5 different classes of flowers)  and freezing rest of the params.
The optimizer I have used is Stocastic gradient descent optimizer and has the below configuration :

   `SGD (
Parameter Group 0
    dampening: 0
    lr: 0.001
    momentum: 0
    nesterov: False
    weight_decay: 0
)`

With just 2 epochs the Neural network was train with ~3k images with an CrossEntropy error of `0.87` on training data and shows an accuracy of 77% on testing data. 

![image](https://user-images.githubusercontent.com/11462012/128677834-363f4162-f49b-44b0-8f9d-f6b91dede12e.png)


## _DAY7_ : _Transfer Learning (Part 2) : Neural Style Transfer using VGG19_

In this excercise I have experimented with the neural network to understand the concept of transfer learning using pytorch VGG 19 pretrained model to create and an artistic image using a target image and a style image. On an high level I have written a loss function (reference : 1. [Research Paper](https://arxiv.org/abs/1508.06576) 2. [code](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/neuralstyle)) which calculated :
   + loss (when forward pass is made) between the target image, generated image
   + loss (when forward pass is made) between the gram marix of style and generated image - This loss captured the texture of the image.
* To control these two losses two hyper parameters are defined : 
        1. alpha : Controls the target image visibility in the generated image.
        2. berta : Controls the style texture in the generated image.

![image](https://user-images.githubusercontent.com/11462012/128676826-bee21568-38d2-4a08-aa17-b28d70e05f8a.png)


## _DAY8_ : __
