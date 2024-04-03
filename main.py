import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("preprocessed.csv")

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

# train is now 60% of the entire data set
train, proto_test= train_test_split(data, test_size=1 - train_ratio, random_state=17)

# test, validation are now 20% of the initial data set
val, test = train_test_split(proto_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=17)

'''
EQUAL PROPORTION: Uncomment to run model with 4 classification bins
'''
# train_bin1 = train.loc[train["popularity"] < 21].copy()
# train_bin2 = train.loc[(train["popularity"] >= 21) & (data["popularity"] < 61)].copy()
# train_bin3 = train.loc[(train["popularity"] >= 61) & (data["popularity"] < 72)].copy()
# train_bin4 = train.loc[(train["popularity"] >= 72)].copy()
#
# val_bin1 = val.loc[val["popularity"] < 21].copy()
# val_bin2 = val.loc[(val["popularity"] >= 21) & (data["popularity"] < 61)].copy()
# val_bin3 = val.loc[(val["popularity"] >= 61) & (data["popularity"] < 72)].copy()
# val_bin4 = val.loc[(val["popularity"] >= 72)].copy()
#
# test_bin1 = test.loc[test["popularity"] < 21].copy()
# test_bin2 = test.loc[(test["popularity"] >= 21) & (data["popularity"] < 61)].copy()
# test_bin3 = test.loc[(test["popularity"] >= 61) & (data["popularity"] < 72)].copy()
# test_bin4 = test.loc[(test["popularity"] >= 72)].copy()
#
# '''
# reassign popularity values to mimic classifiers
# '''
# train_bin1["popularity"], test_bin1["popularity"], val_bin1["popularity"] = 1, 1, 1
# train_bin2["popularity"], test_bin2["popularity"], val_bin2["popularity"] = 2, 2, 2
# train_bin3["popularity"], test_bin3["popularity"], val_bin3["popularity"] = 3, 3, 3
# train_bin4["popularity"], test_bin4["popularity"], val_bin4["popularity"] = 4, 4, 4
#
# train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4])
# test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4])
# val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4])


'''
EQUAL PROPORTION: Uncomment to run model with 5 classification bins
'''
# train_bin1 = train.loc[train["popularity"] < 17].copy()
# train_bin2 = train.loc[(train["popularity"] >= 17) & (data["popularity"] < 55)].copy()
# train_bin3 = train.loc[(train["popularity"] >= 55) & (data["popularity"] < 66)].copy()
# train_bin4 = train.loc[(train["popularity"] >= 66) & (train["popularity"] < 75)].copy()
# train_bin5 = train.loc[(train["popularity"] >= 75)].copy()
#
# val_bin1 = val[val["popularity"] < 17].copy()
# val_bin2 = val[(val["popularity"] >= 17) & (val["popularity"] < 55)].copy()
# val_bin3 = val[(val["popularity"] >= 55) & (val["popularity"] < 66)].copy()
# val_bin4 = val[(val["popularity"] >= 66) & (val["popularity"] < 75)].copy()
# val_bin5 = val[val["popularity"] >= 75].copy()
#
# test_bin1 = test[test["popularity"] < 17].copy()
# test_bin2 = test[(test["popularity"] >= 17) & (test["popularity"] < 55)].copy()
# test_bin3 = test[(test["popularity"] >= 55) & (test["popularity"] < 66)].copy()
# test_bin4 = test[(test["popularity"] >= 66) & (test["popularity"] < 75)].copy()
# test_bin5 = test[test["popularity"] >= 75].copy()
#
# train_bin1["popularity"], test_bin1["popularity"], val_bin1["popularity"] = 1, 1, 1
# train_bin2["popularity"], test_bin2["popularity"], val_bin2["popularity"] = 2, 2, 2
# train_bin3["popularity"], test_bin3["popularity"], val_bin3["popularity"] = 3, 3, 3
# train_bin4["popularity"], test_bin4["popularity"], val_bin4["popularity"] = 4, 4, 4
# train_bin5["popularity"], test_bin5["popularity"], val_bin5["popularity"] = 5, 5, 5
#
# train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5])
# test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5])
# val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5])


'''
EQUAL PROPORTIONS: Uncomment to run model with 7 classification bins
'''
# The following splits were the most equal that I could produce
train_bin1 = train.loc[train["popularity"] < 12].copy()
train_bin2 = train.loc[(train["popularity"] >= 12) & (data["popularity"] < 30)].copy()
train_bin3 = train.loc[(train["popularity"] >= 30) & (data["popularity"] < 57)].copy()
train_bin4 = train.loc[(train["popularity"] >= 57) & (train["popularity"] < 65)].copy()
train_bin5 = train.loc[(train["popularity"] >= 66) & (train["popularity"] < 72)].copy()
train_bin6 = train.loc[(train["popularity"] >= 72) & (train["popularity"] < 78)].copy()
train_bin7 = train.loc[(train["popularity"] >= 78)].copy()

val_bin1 = val[val["popularity"] < 12].copy()
val_bin2 = val[(val["popularity"] >= 12) & (val["popularity"] < 30)].copy()
val_bin3 = val[(val["popularity"] >= 30) & (val["popularity"] < 57)].copy()
val_bin4 = val[(val["popularity"] >= 57) & (val["popularity"] < 65)].copy()
val_bin5 = val[(val["popularity"] >= 66) & (val["popularity"] < 72)].copy()
val_bin6 = val[(val["popularity"] >= 72) & (val["popularity"] < 78)].copy()
val_bin7 = val[val["popularity"] >= 78].copy()

test_bin1 = test[test["popularity"] < 12].copy()
test_bin2 = test[(test["popularity"] >= 12) & (test["popularity"] < 30)].copy()
test_bin3 = test[(test["popularity"] >= 30) & (test["popularity"] < 57)].copy()
test_bin4 = test[(test["popularity"] >= 57) & (test["popularity"] < 65)].copy()
test_bin5 = test[(test["popularity"] >= 66) & (test["popularity"] < 72)].copy()
test_bin6 = test[(test["popularity"] >= 72) & (test["popularity"] < 78)].copy()
test_bin7 = test[test["popularity"] >= 78].copy()

# Modifying popularity values to reflect classes
train_bin1["popularity"], test_bin1["popularity"], val_bin1["popularity"] = 1, 1, 1
train_bin2["popularity"], test_bin2["popularity"], val_bin2["popularity"] = 2, 2, 2
train_bin3["popularity"], test_bin3["popularity"], val_bin3["popularity"] = 3, 3, 3
train_bin4["popularity"], test_bin4["popularity"], val_bin4["popularity"] = 4, 4, 4
train_bin5["popularity"], test_bin5["popularity"], val_bin5["popularity"] = 5, 5, 5
train_bin6["popularity"], test_bin6["popularity"], val_bin6["popularity"] = 6, 6, 6
train_bin7["popularity"], test_bin7["popularity"], val_bin7["popularity"] = 7, 7, 7

# Concatenate classes for each set
train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5, train_bin6, train_bin7])
test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5, test_bin6, test_bin7])
val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5, val_bin6, val_bin7])

#Translate to numpy array to pass to sklearn
X_train = train_ready.loc[:, train_ready.columns != "popularity"].values
Y_train = train_ready.loc[:, train_ready.columns == "popularity"].values

X_val = val_ready.loc[:, val_ready.columns != "popularity"].values
Y_val = val_ready.loc[:, val_ready.columns == "popularity"].values

# Build and run model
model_test1 = RandomForestClassifier(random_state=17)
model_test1.fit(X_train, Y_train.ravel())
result = np.array(model_test1.predict(X_val))

# Display number of incorrectly predicted classes from validation set
print(sum(Y_val.ravel() != result))












