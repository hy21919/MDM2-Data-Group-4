import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('recentpreprocessed.csv')

# Split data into train, validation, test sets
train, proto_test = train_test_split(data, train_size=0.6, shuffle=True, random_state=42)
val, test = train_test_split(proto_test, train_size=0.5, shuffle=True, random_state=42)

# train_bin1 = train[train["popularity"] < 10].copy()
# train_bin2 = train[(train["popularity"] >= 10) & (train["popularity"] < 20)].copy()
# train_bin3 = train[(train["popularity"] >= 20) & (train["popularity"] < 30)].copy()
# train_bin4 = train[(train["popularity"] >= 30) & (train["popularity"] < 40)].copy()
# train_bin5 = train[(train["popularity"] >= 40) & (train["popularity"] < 50)].copy()
# train_bin6 = train[(train["popularity"] >= 50) & (train["popularity"] < 60)].copy()
# train_bin7 = train[(train["popularity"] >= 60) & (train["popularity"] < 70)].copy()
# train_bin8 = train[(train["popularity"] >= 70) & (train["popularity"] < 80)].copy()
# train_bin9 = train[(train["popularity"] >= 80) & (train["popularity"] < 90)].copy()
# train_bin10 = train[train["popularity"] >= 90].copy()
#
# val_bin1 = val[val["popularity"] < 8].copy()
# val_bin2 = val[(val["popularity"] >= 8) & (val["popularity"] < 16)].copy()
# val_bin3 = val[(val["popularity"] >= 16) & (val["popularity"] < 31)].copy()
# val_bin4 = val[(val["popularity"] >= 31) & (val["popularity"] < 54)].copy()
# val_bin5 = val[(val["popularity"] >= 54) & (val["popularity"] < 61)].copy()
# val_bin6 = val[(val["popularity"] >= 61) & (val["popularity"] < 66)].copy()
# val_bin7 = val[(val["popularity"] >= 66) & (val["popularity"] < 70)].copy()
# val_bin8 = val[(val["popularity"] >= 70) & (val["popularity"] < 75)].copy()
# val_bin9 = val[(val["popularity"] >= 75) & (val["popularity"] < 79)].copy()
# val_bin10 = val[val["popularity"] >= 79].copy()
#
# test_bin1 = test[test["popularity"] < 8].copy()
# test_bin2 = test[(test["popularity"] >= 8) & (test["popularity"] < 16)].copy()
# test_bin3 = test[(test["popularity"] >= 16) & (test["popularity"] < 31)].copy()
# test_bin4 = test[(test["popularity"] >= 31) & (test["popularity"] < 54)].copy()
# test_bin5 = test[(test["popularity"] >= 54) & (test["popularity"] < 61)].copy()
# test_bin6 = test[(test["popularity"] >= 61) & (test["popularity"] < 66)].copy()
# test_bin7 = test[(test["popularity"] >= 66) & (test["popularity"] < 70)].copy()
# test_bin8 = test[(test["popularity"] >= 70) & (test["popularity"] < 75)].copy()
# test_bin9 = test[(test["popularity"] >= 75) & (test["popularity"] < 79)].copy()
# test_bin10 = test[test["popularity"] >= 79].copy()
#
# # Modifying popularity values to reflect classes
# train_bin1["popularity"], test_bin1["popularity"], val_bin1["popularity"] = 1, 1, 1
# train_bin2["popularity"], test_bin2["popularity"], val_bin2["popularity"] = 2, 2, 2
# train_bin3["popularity"], test_bin3["popularity"], val_bin3["popularity"] = 3, 3, 3
# train_bin4["popularity"], test_bin4["popularity"], val_bin4["popularity"] = 4, 4, 4
# train_bin5["popularity"], test_bin5["popularity"], val_bin5["popularity"] = 5, 5, 5
# train_bin6["popularity"], test_bin6["popularity"], val_bin6["popularity"] = 6, 6, 6
# train_bin7["popularity"], test_bin7["popularity"], val_bin7["popularity"] = 7, 7, 7
# train_bin8["popularity"], test_bin8["popularity"], val_bin8["popularity"] = 8, 8, 8
# train_bin9["popularity"], test_bin9["popularity"], val_bin9["popularity"] = 9, 9, 9
# train_bin10["popularity"], test_bin10["popularity"], val_bin10["popularity"] = 10, 10, 10
#
# # Concatenate classes for each set
# train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5, train_bin6, train_bin7,
# train_bin8, train_bin9, train_bin10])
# test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5, test_bin6, test_bin7, test_bin8,
# test_bin9, test_bin10])
# val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5, val_bin6, val_bin7, val_bin8, val_bin9,
# val_bin10])

'''
EQUAL WIDTHS: Uncomment to run model with 13 classification bins
'''
# Train data
train_bin1 = train[train["popularity"] < 6].copy()
train_bin2 = train[(train["popularity"] >= 6) & (train["popularity"] < 13)].copy()
train_bin3 = train[(train["popularity"] >= 13) & (train["popularity"] < 20)].copy()
train_bin4 = train[(train["popularity"] >= 20) & (train["popularity"] < 28)].copy()
train_bin5 = train[(train["popularity"] >= 28) & (train["popularity"] < 35)].copy()
train_bin6 = train[(train["popularity"] >= 35) & (train["popularity"] < 43)].copy()
train_bin7 = train[(train["popularity"] >= 43) & (train["popularity"] < 50)].copy()
train_bin8 = train[(train["popularity"] >= 50) & (train["popularity"] < 58)].copy()
train_bin9 = train[(train["popularity"] >= 58) & (train["popularity"] < 65)].copy()
train_bin10 = train[(train["popularity"] >= 65) & (train["popularity"] < 73)].copy()
train_bin11 = train[(train["popularity"] >= 73) & (train["popularity"] < 85)].copy()
train_bin12 = train[(train["popularity"] >= 85) & (train["popularity"] < 92)].copy()
train_bin13 = train[(train["popularity"] >= 92) & (train["popularity"] <= 100)].copy()

# Validation data
val_bin1 = val[val["popularity"] < 6].copy()
val_bin2 = val[(val["popularity"] >= 6) & (val["popularity"] < 13)].copy()
val_bin3 = val[(val["popularity"] >= 13) & (val["popularity"] < 20)].copy()
val_bin4 = val[(val["popularity"] >= 20) & (val["popularity"] < 28)].copy()
val_bin5 = val[(val["popularity"] >= 28) & (val["popularity"] < 35)].copy()
val_bin6 = val[(val["popularity"] >= 35) & (val["popularity"] < 43)].copy()
val_bin7 = val[(val["popularity"] >= 43) & (val["popularity"] < 50)].copy()
val_bin8 = val[(val["popularity"] >= 50) & (val["popularity"] < 58)].copy()
val_bin9 = val[(val["popularity"] >= 58) & (val["popularity"] < 65)].copy()
val_bin10 = val[(val["popularity"] >= 65) & (val["popularity"] < 73)].copy()
val_bin11 = val[(val["popularity"] >= 73) & (val["popularity"] < 85)].copy()
val_bin12 = val[(val["popularity"] >= 85) & (val["popularity"] < 92)].copy()
val_bin13 = val[(val["popularity"] >= 92) & (val["popularity"] <= 100)].copy()

# Test data
test_bin1 = test[test["popularity"] < 8].copy()
test_bin2 = test[(test["popularity"] >= 7) & (test["popularity"] < 13)].copy()
test_bin3 = test[(test["popularity"] >= 13) & (test["popularity"] < 20)].copy()
test_bin4 = test[(test["popularity"] >= 20) & (test["popularity"] < 28)].copy()
test_bin5 = test[(test["popularity"] >= 28) & (test["popularity"] < 35)].copy()
test_bin6 = test[(test["popularity"] >= 35) & (test["popularity"] < 43)].copy()
test_bin7 = test[(test["popularity"] >= 43) & (test["popularity"] < 50)].copy()
test_bin8 = test[(test["popularity"] >= 50) & (test["popularity"] < 58)].copy()
test_bin9 = test[(test["popularity"] >= 58) & (test["popularity"] < 65)].copy()
test_bin10 = test[(test["popularity"] >= 65) & (test["popularity"] < 73)].copy()
test_bin11 = test[(test["popularity"] >= 73) & (test["popularity"] < 85)].copy()
test_bin12 = test[(test["popularity"] >= 85) & (test["popularity"] < 92)].copy()
test_bin13 = test[(test["popularity"] >= 92) & (test["popularity"] <= 100)].copy()

# Modifying popularity values to reflect classes
train_bin1["popularity"], test_bin1["popularity"], val_bin1["popularity"] = 1, 1, 1
train_bin2["popularity"], test_bin2["popularity"], val_bin2["popularity"] = 2, 2, 2
train_bin3["popularity"], test_bin3["popularity"], val_bin3["popularity"] = 3, 3, 3
train_bin4["popularity"], test_bin4["popularity"], val_bin4["popularity"] = 4, 4, 4
train_bin5["popularity"], test_bin5["popularity"], val_bin5["popularity"] = 5, 5, 5
train_bin6["popularity"], test_bin6["popularity"], val_bin6["popularity"] = 6, 6, 6
train_bin7["popularity"], test_bin7["popularity"], val_bin7["popularity"] = 7, 7, 7
train_bin8["popularity"], test_bin8["popularity"], val_bin8["popularity"] = 8, 8, 8
train_bin9["popularity"], test_bin9["popularity"], val_bin9["popularity"] = 9, 9, 9
train_bin10["popularity"], test_bin10["popularity"], val_bin10["popularity"] = 10, 10, 10
train_bin11["popularity"], test_bin11["popularity"], val_bin11["popularity"] = 11, 11, 11
train_bin12["popularity"], test_bin12["popularity"], val_bin12["popularity"] = 12, 12, 12
train_bin13["popularity"], test_bin13["popularity"], val_bin13["popularity"] = 13, 13, 13

train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5, train_bin6, train_bin7, train_bin8,
                         train_bin9, train_bin10, train_bin11, train_bin12, train_bin13])
test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5, test_bin6, test_bin7, test_bin8,
                        test_bin9, test_bin10, test_bin11, test_bin12, test_bin13])
val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5, val_bin6, val_bin7, val_bin8, val_bin9,
                       val_bin10, val_bin11, val_bin12, val_bin13])




# Split into features and labels
X_train, y_train = train_ready.drop('popularity', axis=1), train_ready['popularity']
X_val, y_val = val_ready.drop('popularity', axis=1), val_ready['popularity']
X_test, y_test = test_ready.drop('popularity', axis=1), test_ready['popularity']

# Train a linear SVM model
svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovr'))
svm_model.fit(X_train, y_train)

# Predict on validation and test data
y_val_predicted = svm_model.predict(X_val)
y_test_predicted = svm_model.predict(X_test)

# Evaluate the model with accuracy and confusion matrix
accuracy_val = accuracy_score(y_val, y_val_predicted)
accuracy_test = accuracy_score(y_test, y_test_predicted)
confusion_matrix_val = confusion_matrix(y_val, y_val_predicted)
confusion_matrix_test = confusion_matrix(y_test, y_test_predicted)

print(f'Validation Accuracy: {accuracy_val}')
print(f'Test Accuracy: {accuracy_test}')
print("Validation Confusion Matrix:\n", confusion_matrix_val)
print("Test Confusion Matrix:\n", confusion_matrix_test)

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap="Blues", cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

# Plot confusion matrix for validation and test data
plot_confusion_matrix(confusion_matrix_val, 'Validation Confusion Matrix')
plot_confusion_matrix(confusion_matrix_test, 'Test Confusion Matrix')
