
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.model_selection import train_test_split
from skimage import feature,color,io,exposure






letter_images=pd.read_csv("emnist-letters-train.csv")

letter_images.iloc[:,0]=letter_images.iloc[:,0]-1
letter_images.iloc[:,0].min(),letter_images.iloc[:,0].max()


X=letter_images.iloc[:,1:].values
y=letter_images.iloc[:,0].values


X.shape,y.shape




X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y,shuffle=True)


X_train.shape,y_train.shape,X_train[0].reshape((28,28))



train_feature=[]
test_feature=[]

for train_img in X_train:
    train_img=train_img.reshape((28,28))
    
    hog_train_img,_=feature.hog(train_img,visualize=True)
    train_feature.append(hog_train_img)
    
for test_img in X_test:
    test_img=test_img.reshape((28,28))
    
    hog_test_img,_=feature.hog(test_img,visualize=True)
    test_feature.append(hog_test_img)
    




gray_image=X_train[0].reshape((28,28))

hog_feature,hog_img= feature.hog(gray_image,visualize=True)

plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(gray_image,cmap="gray")
plt.title("original image")

plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(hog_img,cmap="gray")
plt.title("hog image")


X_train_img=np.array(train_feature)
X_test_img=np.array(test_feature)


X_train_img.shape,X_test_img.shape,y_train.shape,y_test.shape,X_train_img[0]



model= Sequential([
    tf.keras.Input(shape=(81,)),
    Dense(units=128,activation="relu",name="layer1"),
    Dense(units=128,activation="relu",name="layer2"),
    Dense(units=26,activation="linear",name="layer3")
],name="Mymodel"
)




model.summary()



[layer1,layer2,layer3]=model.layers


w1,b1=layer1.get_weights()
w2,b2=layer2.get_weights()

w3,b3=layer3.get_weights()

print(f"w1 shape:{w1.shape} , b1 shape: {b1.shape}")
print(f"w2 shape:{w2.shape} , b2 shape: {b2.shape}")
print(f"w3 shape:{w3.shape} , b3 shape: {b3.shape}")



model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)


Letter_recognition=model.fit(
    X_train_img,y_train,
    epochs=100
)




prediction_model=model.predict(X_test_img)


prediction_model[:10]


prediction_p=tf.nn.softmax(prediction_model)
prediction_p




loss,accurecy=model.evaluate(X_test_img,y_test)


accurecy





