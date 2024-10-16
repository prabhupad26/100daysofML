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

> [Link](https://github.com/prabhupad26/100daysofML/blob/main/DAY4_MNIST_DATA_pytorch.ipynb) to python notebook .

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

> [Link](https://github.com/prabhupad26/100daysofML/blob/main/DAY5_Car_Prices_Prediction.ipynb) to python notebook.

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

> [Link](https://github.com/prabhupad26/100daysofML/blob/main/DAY6_transfer-learning_part1/transfer_learning.ipynb) to python notebook.

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

> [Link](https://github.com/prabhupad26/100daysofML/blob/main/DAY7_transfer-learning_part2/transfer_learning.ipynb) to python notebook.

In this excercise I have experimented with the neural network to understand the concept of transfer learning using pytorch VGG 19 pretrained model to create and an artistic image using a target image and a style image. On an high level I have written a loss function (reference : 1. [Research Paper](https://arxiv.org/abs/1508.06576) 2. [code](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/neuralstyle)) which calculated :
   + loss (when forward pass is made) between the target image, generated image
   + loss (when forward pass is made) between the gram marix of style and generated image - This loss captured the texture of the image.
* To control these two losses two hyper parameters are defined :
        1. alpha : Controls the target image visibility in the generated image.
        2. berta : Controls the style texture in the generated image.

![image](https://user-images.githubusercontent.com/11462012/128676826-bee21568-38d2-4a08-aa17-b28d70e05f8a.png)


## _DAY8_ : _Pytorch_Tutorials_Saving and loading a trained model_

> [Link](https://github.com/prabhupad26/100daysofML/tree/main/DAY8_pytorch_part_1) to notebooks folder.

* I'll train a neural network and save its model parameters and optimizers to a `checkpoint.pth.tar` file then load those parameters from the file again and test the model with a sample input.

* Install and use tensorboard to see the training results.

![image](https://user-images.githubusercontent.com/11462012/128732621-5cc2235f-1fa6-4db3-a313-abba52504481.png)

* Use Learning rate schedule in pytorch (`torch.optim.lr_scheduler`) .


## _DAY9_ : _Pytorch_RNN LSTM GRU on MNIST Dataset_

> [Link](https://github.com/prabhupad26/100daysofML/blob/main/DAY9_pytorch_part2/pytorch_lstm_rnn_gru.ipynb) to notebooks folder.

* I'll train RNN, LSTM, GRU and compare the results on the MNIST dataset.

* Here is the model accuracy and training loss comparison between the 3 neural nets I trained on the MNIST dataset :

![image](https://user-images.githubusercontent.com/11462012/128881657-09459d84-72ee-4617-a2d1-f7d2033b9bba.png)

* References:
  - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
  - https://colah.github.io/posts/2015-08-Understanding-LSTMs/


## _DAY10_ : _Seq2Seq (Eng2German) Translation transformer implementation (pytorch)_


**References :**
   - [Tensor2tensor notebook to understand self attention](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb#scrollTo=OJKU36QAfqOC)
   - Research paper - [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
   - [Blog post exmplaining transformer](http://jalammar.github.io/illustrated-transformer/)
   - [Pytorch implementation guide](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
   - [Visualize positional encoding code](https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb)
   - 


## _DAY11_ : _Bi directional LSTMs (pytorch)_


I train a Bidirectional LSTM , an LSTM with 2 input which takes original sequence as one input and a reverse of that input which help the neural network to capture the future data as well and the output will consist of those future context as well.
Here I have used a hand crafted dataset for training the model 


## __DAY12__ : __Generating sine waves using LSTMs__

* Training Data :
   - Data is created using `numpy` library, created 100 data points (sine waves y - axis) with 1000 time steps (sine waves x-axis).
 * Learning :
   - The optimizer used here is the `LBFGS` algorithm with a learning rate of `0.8` 


I'll be  training a LSTM network to generate sine waves : 

![image](https://user-images.githubusercontent.com/11462012/129445031-a4f5a294-5701-4df1-bb98-cdeb135be49a.png)

* References
   - [Pytorch example](https://github.com/pytorch/examples/tree/master/time_sequence_prediction)


## __DAY13__ : __Simple GAN to train train Generator to produce fake images from MNIST dataset__


![image](https://user-images.githubusercontent.com/11462012/129450853-3f447f49-327b-4027-a530-a5b0633eca77.png)

* References
   - [Article on GANs](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)

> * Code to run tensorboard on google colab :
>     `%load_ext tensorboard`
>     `%tensorboard --logdir logs`


## __DAY14 - 18__ : __Effective LSTMs for Target-Dependent Sentiment Classification (Research Paper Implementation)__
<a name="AspectbasedLSTM"></a>
* Overview:
   In [this](https://arxiv.org/pdf/1512.01100.pdf) paper it has been shown that by providing target information to an LSTM model can significantly boost the performance of the model in classifying the sentiment for the sentence. Sentiment analysis is a classic problem in NLP in which the polarity of the input (sentence) is to be predicted (polarity like : Good review ,Neutral review, Bad review, Worse review etc.). In this paper there are two LSTMs model has been proposed in which both the models are trained with the context words as well as target words.
* Problem statement example : 
      - Input sentence : "I bought a camera, its picture quality is awesome but the battery life is too short" , here if the target is "picture quality" then the sentiment should be "positive" , but if the target would have been "battery life", then the sentiment would have been "negative".
  
* Dataset:
   Pre trained Word vectors: 
      - Pre trained word vectors representations were obtained from [GloVe Twitter Dataset] (https://nlp.stanford.edu/projects/glove/) which is having around 2Billion word tokens with a dimensions of 50 , 100 , 200.
   - Data : The data is obtained from the [SemEval2014 task4](http://alt.qcri.org/semeval2014/task4/) for Restaurant and Laptop review comments, its a labeled dataset with the targets consists of 3 classes : `Positive`, `Negative` and `Neutral`
   
 * The main focus of this exercise is to understand how an LSTM network behaves when the target information is fed into its input during a sentiment classification task. So basically the idea is to train a LSTM network with target information along with the input and the output would be the polarity of the input based on that provided target information, also there can be cases wherein the same input sentence can have different polarity depending upon the target information. For example:
  If the input sentence is : "I really liked the laptop but not because of its Windows 8 Operating System" and the target information is : "Windows 8" then the polarity of this sentence should be "negative". 
  
 * Model training :
   - The tokenizer is created by capturing all the words in the train, test dataset (using the xml parser in `data_utils.py`) . Now after this a vocabulary is created out of all these words and is indexed.
   - The word vector is loaded from the file and using this the dataset(each word in every example) is converted to the vector form (I have used the 200d vector in this exercise).
   
  
 
 * Model Evaluation : 
   
   ![image](https://user-images.githubusercontent.com/11462012/129765477-48853407-690a-4c3b-92bf-4b21f6f45a81.png)
   
   ![image](https://user-images.githubusercontent.com/11462012/129767071-967149c6-eebc-4461-8350-826f157ab574.png)


 * Sample Examples :
   * Aspect Embedded LSTM Network Result (Target Dependent LSTM) : 
   
      ![image](https://user-images.githubusercontent.com/11462012/129766080-4aeeef07-fe30-469e-be55-013d931cadf0.png)
   
   * Normal LSTM output :
   
      ![image](https://user-images.githubusercontent.com/11462012/129766379-a1c6b05f-4da6-46bc-897a-8c0ecf0b3af5.png)


 * Code References :
   - [Pytorch implementation](https://github.com/hiyouga/PBAN-PyTorch)


## __DAY19-25__ : __CLIP by OpenAI (Research Paper Implementation)__
* Title of the paper - Learning Transferable visual models from Natural Language Supervision.
* This paper has focussed on the idea of learning the image representation with the supervision of text representation. The resultant model has the capability to perform classification tasks without any training data (AKA Zero shot learning classification). This model was able to preform with a significant accuracy on different image data sets like ImageNet.
* Model training :
   - There is a pretraining step which is also called as contrastive learnining step in which the model is trained on the Image representation (created by a transformer as an encoder) and Text representation (created by another encoder) from scratch, the objective of this training is to maximize the cosine similarity between the `N` real/correct pairs of image representation and the text representation and minimizing the `N^2 - N` incorrect set of pairs, optimized using a cross entropy. This training creates a  multimodal embedded  representation which is further used for zero shot classification. _The temperature parameter which estimates the range of logits in the softmax function output is trained as log parameterized multiplicative scalar._
   - Modification in ResNET-50 (Base model architecture) the global average pooling layer is replaced with an attention pooling layer, this attention layer is the transformer style QKV attention, the query is conditioned on global average pooled representation. 
   - The text encoder is also a transformer layer which operated on byte pair encoded representation of text, with a sequence limit of 76 token each sequence is appended / padded with the \[SOS] \[EOS] tokens 


* I tried this model with some of my sample images here are the results :
   
   ![image](https://user-images.githubusercontent.com/11462012/130329167-907fcbaf-f39d-41d1-8b99-0a1a86042da3.png)
   

* Prompt engineering :
   - There are multiple cases in which the text ender of CLIP isn't provided with enough context for the model to make accurate prediction, most of the cases in which single word is provided as the target label. But since the model is pretrained with the test data which somewhat describes the images so this creates a distribution gap which is resolved by doing prompt engineering.
   - Few templates are create like : `A photo of a bug {label}` etc. and these templates are configures with those single word labels. This helps the model to get a significant improvement in identifying the input images.

* A frontend which shows the application of CLIP :
  - A simple GUI in which any of the dataset can be selected (even custom dataset) and any pre trained model can be selected.
  - There are two ways to demonstrate the application of CLIP:
      1. (Obtain image from text description) Enter a text which describe the image and the image will be presented after getting the prediction from the model.
      ![image](https://user-images.githubusercontent.com/11462012/130812087-acd27ef8-9e7b-4c6e-9f55-dbc8d73e142a.png)

      2. (Obtain class from image as input) An image should be uploaded to the GUI and top 5 predicted classes will be displayed. 
      ![image](https://user-images.githubusercontent.com/11462012/130812184-3cf915bf-d53a-4ac2-8e38-2795b303c88d.png)



* References :
  - https://openai.com/blog/clip/#rf2
  - https://github.com/openai/CLIP
  - paper - https://arxiv.org/pdf/2103.00020.pdf


* To explore : 
   - https://github.com/tensorflow/compression
   - https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/



## __DAY26-31__ : __Speech2Text: Fine Tuning Wav2Vec2 for English Auto Speech Recognition__

**Problem Statement/Task : Generate a transcription of a voice recording which is stored in a sound file**

**Approach** :  
We will be using the base check point of a pretrained `wav2vec 2.0` ASR model which is trained on 50 hrs of unlabeled speech recordings and predicts the speaker of input speech recording. We will be adding a linear layer which will be mapping the contextual representation generated by this pretrained model to vocabulary that we have build from the dataset, this linear layer will be trained to do this mapping. 
   1. Obtain sound file to sentence data from `time_asr` dataset
   2. Remove special characters from the labels (sentences)
   3. Build a vocabulary out of all the characters in all the labels. ([UNK] , [PAD] tokens are added to the vocabs for unknown character and padding for identifying the end of words).
   4. Convert the raw sound data to sampled data which will further be used for training the model:
      - [Wav2vec2CTC tokenizer](https://huggingface.co/transformers/master/model_doc/wav2vec2.html#wav2vec2ctctokenizer) is used for tokenizing the inputs which maps the context representation created by wav2vec to the transcription based on the vocab defined in step 3.
      - Feature extractor is used with sampling rate of 16 kHz, input is also padded so that shorter input should be of same size , input is also normalized.(All the data points should have the same sampling rate).
   6. After loading the pretrained model , `require_grad` is set to False using :
      > model.freeze_feature_extractor()
   7. Model is evaluated using the [word error rate](https://huggingface.co/metrics/wer)

**Dataset used for fine-tuning:** `timit_asr` corpus containing 5300 labeled (both test-1680 and train-4620 dataset) speech of sentences recorded by [630 speakers](https://huggingface.co/datasets/timit_asr). The wav2vec2.0 model has performed the [best](https://paperswithcode.com/sota/speech-recognition-on-timit) on this dataset for a automatic speech recognition task. In this exercise we will be using this learning model to get the Text out of Speech.

**References**
1. [Fine tune Wave2vec2 for English ASR with 🤗 transformer](https://huggingface.co/blog/fine-tune-wav2vec2-english)
2. [Sequence Modeling with CTC](https://distill.pub/2017/ctc/)


## __DAY32-36__ : __Target Based Sentiment Analysis UI__
* Problem statement : Build a aspect based sentiment analysis model which will be able to predict the sentiment of a review comment from pre defined categories : `positive`, `negative` and `neutral` *
* Solution : The approach to solve this problem is mentioned [here](#AspectbasedLSTM)

   ![Snapshot of the UI](https://github.com/prabhupad26/100daysofML/blob/main/DAY32-36%20Target%20Based%20Sentiment%20Analysis%20UI/ui_demo_gif.gif)
   
   
## __DAY37-43__ : __Stack Overflow Tags generation__
 * Tasklist for this week's excercise:

   - [x] Load data to DB
   - [x] Read data from DB in iteration
   - [x] Check how much duplicate rows are present and remove the duplicates
   - [x] Get tags count and plot to find out mode frequent tags
   - [x] Get # of tags per question count 
   - [ ] Preprocessing Body: Remove html tags, spl cahrs, , lowercase all, stemming and lemmitization
   - [ ] Define vector
   - [ ] Define labels
   - [ ] Splitting the data
   - [ ] Define ML models
   - [ ] Train, Test

 * References : 
   - https://github.com/chauhanakash23/StackOverflow-Tag-prediction/blob/master/SO_Tag_Predictor.ipynb
