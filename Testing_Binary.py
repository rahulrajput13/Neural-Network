import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# Import the SimpleNeuralNetwork class from your original file
from SimpleNeuralNetwork import SimpleNeuralNetwork

##########################################################################################
# TESTING - BINARY CLASSIFICATION
##########################################################################################

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

bc = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
print(bc[0].head())
print(bc[1].head())

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(bc[0], bc[1], test_size=0.2, random_state=1) # 80% training and 20% test

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

model_binary = SimpleNeuralNetwork(learning_rate=0.01, epochs=50, activation='leaky_relu', loss_function='Binary_Cross_Entropy', layers=[7, 5, 6], classes=2)
model_binary.train(X_train, y_train)

# Predict the test set results
y_pred = model_binary.predict(X_test)

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
if model_binary.classes == 2:
    roc_auc = roc_auc_score(y_test, model_binary.sigmoid(y_pred))
    print(f"ROC AUC: {roc_auc*100:.2f}%")

# Compute confusion matrix (for multi-class classification)
if model_binary.classes > 2:
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
