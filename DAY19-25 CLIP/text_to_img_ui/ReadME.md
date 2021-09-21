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

