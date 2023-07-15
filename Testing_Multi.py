import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# Import the SimpleNeuralNetwork class from your original file
from SimpleNeuralNetwork import SimpleNeuralNetwork

##########################################################################################
# TESTING - MULTI CLASS CLASSIFICATION
##########################################################################################

iris = datasets.load_iris(return_X_y=True, as_frame=True)
print(iris[0].head())
print(iris[1].head())

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris[0], iris[1], test_size=0.2, random_state=1) # 80% training and 20% test

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

model_multi = SimpleNeuralNetwork(learning_rate=0.01, epochs=50, activation='leaky_relu', loss_function='Categorical_Cross_Entropy', layers=[7, 5, 6], classes=3)
model_multi.train(X_train, y_train)

# Predict the test set results
y_pred = model_multi.predict(X_test)
y_pred = np.argmax(model_multi.predict(X_test), axis=1)

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Compute precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")

# Compute ROC AUC (only for binary classification)
if model_multi.classes == 2:
    roc_auc = roc_auc_score(y_test, model_multi.sigmoid(y_pred))
    print(f"ROC AUC: {roc_auc*100:.2f}%")

# Compute confusion matrix (for multi-class classification)
if model_multi.classes > 2:
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)