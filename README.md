# Chararcter-Regonition

## Implementation Details
EMMIST characters dataset has been used to train a neural network. The dataset is divided into files (training and testing). The shape of the data set is 785 columns. The first column is the label of the training example. The rest of the columns are the pixels of the picture that contain the alphabet. The pictures are 28*28 pixels and the pixels are represented in one row in the dataset 784 features. The pixels are in greyscale, each pixel has a value between 0 and 255. the questions of how many layers should be used, how many neurons in these layers, and which activation functions should be used, have no clear answers, and all done in the project is trying different compositions to reach the highest accuracy. 

## Conclusion
Model 1 outperforms Model 2 across precision, recall, and F1 score. Model 1 is more balanced in terms of precision and recall, resulting in a higher F1 score. If precision is a crucial factor, Model 1 may be preferred; otherwise, both models exhibit reasonably good performance. Although the two models have good accuracy, it turns out that adding HOG was not very beneficial for enhancing the performance of the model. One of the reasons that some details were not captured by local gradient by HOG 

