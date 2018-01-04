import numpy as np
x=np.zeros((1,2,3))
X=np.array([[1,1,1],[1,1,1]])
y=np.array([[1],[1],[1]])

np.reshape(y,(1,3))
Y=np.array([[[1,0,0,0,0,0,0,0,0,0] ,[0,1,0,0,0,0,0,0,0,0]]])
x[0][0]=X[0]
x[0][1]=X[1]

print(x)
print(X)