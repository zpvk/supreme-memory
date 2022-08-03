from os import X_OK
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"X train : {X_train}\ny train : {y_train}")
print(f"X shape : {X_train.shape} \ny shape : {y_train.shape}")
m = len(X_train)

plt.scatter(X_train, y_train, marker="X", c='r')   # color
plt.title( "House Predictions" )
plt.xlabel("Size of square feat")
plt.ylabel("House price")
plt.show()

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


tmp_f_wb = compute_model_output(X_train, w, b,)

# Plot our model prediction
plt.plot(X_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(X_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

