import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError

X_train = np.array([ [1,100],[2,200],[3,300],[4,400],[5,500],[6,600],[7,700],[8,800],[10,100],[9,300],[2,1000]])
Y_train = np.array([ [1000],[2000],[3000],[4000],[5000],[6000],[7000],[8000],[5500],[6000],[6000] ])

mymodel = Sequential(
    [
    Dense(10,activation = "relu",name = "L1",kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dense(3,activation = "relu", name = "L2",kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dense(1,activation = "linear", name = "L3"),
    ]
)

mymodel.compile(

        loss= MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(0.01),
)

mymodel.fit(
    X_train,Y_train,
    epochs = 10000
)

predictModel = np.array([[12,1200]])
predictModel2 = np.array([[12,600]])
prediction = mymodel.predict(predictModel)
print(f"here is your prediction {prediction[0]}")
print(f"here is your second prediction{mymodel.predict(predictModel2)[0]}")


