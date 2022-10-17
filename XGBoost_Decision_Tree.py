import numpy as np
from xgboost import XGBClassifier
import sklearn 


#Data input = FACE SHAPE(3)--> IS ROUND? | IS OVAL? | IS SQUARE | HAS WHİSKERS? | EAR SHAPE-> IS POINTY? | IS FLAT?

X_matrix = np.array([ [1,0,0,1,1,0],[0,0,1,1,1,0],[1,0,0,1,0,1],[1,0,0,0,0,1],[0,1,0,1,1,0],[1,0,0,0,1,0] ])
Y = np.array([ [1],[0],[0],[1],[1],[0] ])

model = XGBClassifier()


model.fit(X_matrix,Y)

X_test = np.array([[1,0,0,0,1,0]])
p = model.predict(X_test)
if(p == 1):
    print("İt is a cat")

else:
    print("İt is not a cat")
