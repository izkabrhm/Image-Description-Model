# Image-Description-Model
The Image Description model in this project consists of a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) network where the outputs from each neural network are aligned in a multimodal space embedding to generate sentence descriptions from a given image. The model is trained using multiple open-source image data sets and images from RTA Twitter handle. The performance of the model will be assessed using automatic evaluation metrics based on n-grams similarity, recall rate, and word order.
## Methodology & Experimental Setup
  The Image Description model has been primarily implemented with TensorFlow, Keras and other various libraries. The list of the libraries and modules used are specified below.
  * Tensorflow
  * Keras
  * Matplotlib
  * Scikit-Learn
  * NumPy
  * Python Imaging Library (PIL)
  * nlg-eval (Automatic Evaluation Metric Software)
  * Tweepy
  
For the implementation of the Image Description model using multimodal space embedding, we consider the work done by [Mao et al](https://arxiv.org/pdf/1411.4555.pdf). The Image Description model consists mainly; a CNN pre-trained on the ImageNet database and multimodal RNN (LSTM). Although the implementation is inspired by the work of Mao et al, it has variations in areas like the layers of the multimodal RNN, the loss function used, and the optimizers applied.

To test the performance of the Image Description model, 4 architectures of CNN are used. These CNNs are the DenseNet, VggNet, ResNet, and InceptionV3 (GoogleNet). Each CNN architecture has a different number and arrangement of convolution and pooling layers with each architecture giving an image feature vector of different dimensions.

The image feature vector output from CNN is fed into multimodal RNN where it is aligned with the reference captions in a multimodal space embedding to generate a caption describing the contents of the image. 

![alt text](https://miro.medium.com/max/1034/0*_rsnLEdfV7vrAXHi.png)

(a) represents a simple RNN where the output word from the previous time-step is used as input for the next time-step. (b) represents the complete layout of the multimodal RNN. (c) represents the unrolled version of the recurrent layer which passes the hidden state information to the next time-step. The Multimodal RNN is made up of 5 layers namely Embedding Layer 1, Embedding Layer 2, Recurrent Layer, Multimodal layer, and Vocabulary (SoftMax) Layer. 

Once the training for the Image Description model is complete, we test the performance of the model with the validation image data set. The model takes 20 random images from the validation image data set and generates captions corresponding to each image. Once the captioning process is done, we compare the predicted captions with the reference captions using the nlg-eval software API which calculates the automatic evaluation metric score such as BLEU, ROUGE, METEOR, and CIDEr. It should be noted that the automatic evaluation metrics scores are calculated for the entire corpus of 20 predicted captions instead of comparing each caption for an image so that it will give the overall score for 20 images.

After the training and testing stages of our Image Description model, its performance is tested on images collected from the RTA, Dubai twitter handle. The images are collected using the Tweepy API which allows images to be downloaded and stored in a local directory.
