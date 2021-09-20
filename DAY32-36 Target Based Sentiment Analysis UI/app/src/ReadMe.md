## __DAY32-36__ : __Target Based Sentiment Analysis UI__
* Problem statement : Build a aspect based sentiment analysis model which will be able to predict the sentiment of a review comment from pre defined categories : `positive`, `negative` and `neutral` *
**Solution**
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


   ![Snapshot of the UI](https://github.com/prabhupad26/100daysofML/blob/main/DAY32-36%20Target%20Based%20Sentiment%20Analysis%20UI/ui_demo_gif.gif)
