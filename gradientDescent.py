# Gradient Descent for Linear Regression 
# hypothesis=wx+b;
# loss=(y-hypothesis)**2/N
import numpy as np

# initialse some parameters
x=np.random.randn(10,1)
y=2*x + np.random.rand()
w=0.0
b=0.0
#  hyperparameters
learning_rate=0.01

# create gradient descent function
def descent(x,y,w,b,learning_rate):
    dldw=0.0
    dldb=0.0
    N=x.shape[0]
    for xi,yi in zip(x,y):
        dldw+=-2*xi*(yi-(w*xi+b))
        dldb+=-2*(yi-(w*xi+b))
    w=w-learning_rate*(1/N)*dldw
    b=b-learning_rate*(1/N)*dldb
    return w,b

# iterativly make updates
for epoch in range(400):
    w,b=descent(x,y,w,b,learning_rate)
    hypothesis=w*x+b
    loss=np.divide(np.sum((y-hypothesis)**2,axis=0),x.shape[0])
    print(f'{epoch} loss is{loss},parameters w:{w},b:{b}')
    
