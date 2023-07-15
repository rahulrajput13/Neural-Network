import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the SimpleNeuralNetwork class from your original file
from SimpleNeuralNetwork import SimpleNeuralNetwork

##########################################################################################
# TESTING - REGRESSION
##########################################################################################

db = datasets.load_diabetes(return_X_y=True, as_frame=True)
print(db[0].head())
print(db[1].head())

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(db[0], db[1], test_size=0.2, random_state=1) # 80% training and 20% test

# Standardize the features to have mean=0 and variance=1 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
X_test = scaler.transform(X_test.values)
y_train = y_train.values
y_test = y_test.values

# Ensure the targets are in the correct shape
# y_train = np.array(y_train).reshape(-1, 1)
# y_test = np.array(y_test).reshape(-1, 1)

################

model_reg = SimpleNeuralNetwork(learning_rate=0.01, epochs=5, activation='leaky_relu', loss_function='RMSE', layers=[7, 5, 6], classes=1)
model_reg.train(X_train, y_train)

# Predict the test set results
y_pred = model_reg.predict(X_test)

# Compute the MAE, MSE, RMSE, and R2 score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")