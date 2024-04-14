import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import math

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
# train_bin1 = train.loc[train["popularity"] < 12].copy()
# train_bin2 = train.loc[(train["popularity"] >= 12) & (data["popularity"] < 30)].copy()
# train_bin3 = train.loc[(train["popularity"] >= 30) & (data["popularity"] < 57)].copy()
# train_bin4 = train.loc[(train["popularity"] >= 57) & (train["popularity"] < 65)].copy()
# train_bin5 = train.loc[(train["popularity"] >= 66) & (train["popularity"] < 72)].copy()
# train_bin6 = train.loc[(train["popularity"] >= 72) & (train["popularity"] < 78)].copy()
# train_bin7 = train.loc[(train["popularity"] >= 78)].copy()
#
# val_bin1 = val[val["popularity"] < 12].copy()
# val_bin2 = val[(val["popularity"] >= 12) & (val["popularity"] < 30)].copy()
# val_bin3 = val[(val["popularity"] >= 30) & (val["popularity"] < 57)].copy()
# val_bin4 = val[(val["popularity"] >= 57) & (val["popularity"] < 65)].copy()
# val_bin5 = val[(val["popularity"] >= 66) & (val["popularity"] < 72)].copy()
# val_bin6 = val[(val["popularity"] >= 72) & (val["popularity"] < 78)].copy()
# val_bin7 = val[val["popularity"] >= 78].copy()
#
# test_bin1 = test[test["popularity"] < 12].copy()
# test_bin2 = test[(test["popularity"] >= 12) & (test["popularity"] < 30)].copy()
# test_bin3 = test[(test["popularity"] >= 30) & (test["popularity"] < 57)].copy()
# test_bin4 = test[(test["popularity"] >= 57) & (test["popularity"] < 65)].copy()
# test_bin5 = test[(test["popularity"] >= 66) & (test["popularity"] < 72)].copy()
# test_bin6 = test[(test["popularity"] >= 72) & (test["popularity"] < 78)].copy()
# test_bin7 = test[test["popularity"] >= 78].copy()
#
# # Modifying popularity values to reflect classes
# train_bin1["popularity"], test_bin1["popularity"], val_bin1["popularity"] = 1, 1, 1
# train_bin2["popularity"], test_bin2["popularity"], val_bin2["popularity"] = 2, 2, 2
# train_bin3["popularity"], test_bin3["popularity"], val_bin3["popularity"] = 3, 3, 3
# train_bin4["popularity"], test_bin4["popularity"], val_bin4["popularity"] = 4, 4, 4
# train_bin5["popularity"], test_bin5["popularity"], val_bin5["popularity"] = 5, 5, 5
# train_bin6["popularity"], test_bin6["popularity"], val_bin6["popularity"] = 6, 6, 6
# train_bin7["popularity"], test_bin7["popularity"], val_bin7["popularity"] = 7, 7, 7


'''
EQUAL PROPORTIONS: Uncomment to run model with 10 classification bins
'''
# train_bin1 = train[train["popularity"] < 8].copy()
# train_bin2 = train[(train["popularity"] >= 8) & (train["popularity"] < 16)].copy()
# train_bin3 = train[(train["popularity"] >= 16) & (train["popularity"] < 31)].copy()
# train_bin4 = train[(train["popularity"] >= 31) & (train["popularity"] < 54)].copy()
# train_bin5 = train[(train["popularity"] >= 54) & (train["popularity"] < 61)].copy()
# train_bin6 = train[(train["popularity"] >= 61) & (train["popularity"] < 66)].copy()
# train_bin7 = train[(train["popularity"] >= 66) & (train["popularity"] < 70)].copy()
# train_bin8 = train[(train["popularity"] >= 70) & (train["popularity"] < 75)].copy()
# train_bin9 = train[(train["popularity"] >= 75) & (train["popularity"] < 79)].copy()
# train_bin10 = train[train["popularity"] >= 79].copy()
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
#
# # Concatenate classes for each set
# train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5, train_bin6, train_bin7,
# train_bin8, train_bin9, train_bin10])
# test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5, test_bin6, test_bin7, test_bin8,
# test_bin9, test_bin10])
# val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5, val_bin6, val_bin7, val_bin8, val_bin9,
# val_bin10])


'''
EQUAL WIDTHS: Uncomment to run model with 10 classification bins
'''
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
#
# # Concatenate classes for each set
# train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5, train_bin6, train_bin7,
# train_bin8, train_bin9, train_bin10])
# test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5, test_bin6, test_bin7, test_bin8,
# test_bin9, test_bin10])
# val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5, val_bin6, val_bin7, val_bin8, val_bin9,
# val_bin10])


'''
EQUAL PROPORTIONS: Uncomment to run model with 13 classification bins
'''
# #Train data
# train_bin1 = train[train["popularity"] < 6].copy()
# train_bin2 = train[(train["popularity"] >= 6) & (train["popularity"] < 13)].copy()
# train_bin3 = train[(train["popularity"] >= 13) & (train["popularity"] < 20)].copy()
# train_bin4 = train[(train["popularity"] >= 20) & (train["popularity"] < 38)].copy()
# train_bin5 = train[(train["popularity"] >= 38) & (train["popularity"] < 54)].copy()
# train_bin6 = train[(train["popularity"] >= 54) & (train["popularity"] < 60)].copy()
# train_bin7 = train[(train["popularity"] >= 60) & (train["popularity"] < 64)].copy()
# train_bin8 = train[(train["popularity"] >= 64) & (train["popularity"] < 67)].copy()
# train_bin9 = train[(train["popularity"] >= 67) & (train["popularity"] < 70)].copy()
# train_bin10 = train[(train["popularity"] >= 70) & (train["popularity"] < 74)].copy()
# train_bin11 = train[(train["popularity"] >= 74) & (train["popularity"] < 77)].copy()
# train_bin12 = train[(train["popularity"] >= 77) & (train["popularity"] < 81)].copy()
# train_bin13 = train[train["popularity"] >= 81].copy()
#
# # Validation data
# val_bin1 = val[val["popularity"] < 6].copy()
# val_bin2 = val[(val["popularity"] >= 6) & (val["popularity"] < 13)].copy()
# val_bin3 = val[(val["popularity"] >= 13) & (val["popularity"] < 20)].copy()
# val_bin4 = val[(val["popularity"] >= 20) & (val["popularity"] < 38)].copy()
# val_bin5 = val[(val["popularity"] >= 38) & (val["popularity"] < 54)].copy()
# val_bin6 = val[(val["popularity"] >= 54) & (val["popularity"] < 60)].copy()
# val_bin7 = val[(val["popularity"] >= 60) & (val["popularity"] < 64)].copy()
# val_bin8 = val[(val["popularity"] >= 64) & (val["popularity"] < 67)].copy()
# val_bin9 = val[(val["popularity"] >= 67) & (val["popularity"] < 70)].copy()
# val_bin10 = val[(val["popularity"] >= 70) & (val["popularity"] < 74)].copy()
# val_bin11 = val[(val["popularity"] >= 74) & (val["popularity"] < 77)].copy()
# val_bin12 = val[(val["popularity"] >= 77) & (val["popularity"] < 81)].copy()
# val_bin13 = val[val["popularity"] >= 81].copy()
#
# # Test data
# test_bin1 = test[test["popularity"] < 6].copy()
# test_bin2 = test[(test["popularity"] >= 6) & (test["popularity"] < 13)].copy()
# test_bin3 = test[(test["popularity"] >= 13) & (test["popularity"] < 20)].copy()
# test_bin4 = test[(test["popularity"] >= 20) & (test["popularity"] < 38)].copy()
# test_bin5 = test[(test["popularity"] >= 38) & (test["popularity"] < 54)].copy()
# test_bin6 = test[(test["popularity"] >= 54) & (test["popularity"] < 60)].copy()
# test_bin7 = test[(test["popularity"] >= 60) & (test["popularity"] < 64)].copy()
# test_bin8 = test[(test["popularity"] >= 64) & (test["popularity"] < 67)].copy()
# test_bin9 = test[(test["popularity"] >= 67) & (test["popularity"] < 70)].copy()
# test_bin10 = test[(test["popularity"] >= 70) & (test["popularity"] < 74)].copy()
# test_bin11 = test[(test["popularity"] >= 74) & (test["popularity"] < 77)].copy()
# test_bin12 = test[(test["popularity"] >= 77) & (test["popularity"] < 81)].copy()
# test_bin13 = test[test["popularity"] >= 81].copy()
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
# train_bin11["popularity"], test_bin11["popularity"], val_bin11["popularity"] = 11, 11, 11
# train_bin12["popularity"], test_bin12["popularity"], val_bin12["popularity"] = 12, 12, 12
# train_bin13["popularity"], test_bin13["popularity"], val_bin13["popularity"] = 13, 13, 13
#
# train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5, train_bin6, train_bin7, train_bin8, train_bin9, train_bin10, train_bin11, train_bin12, train_bin13])
# test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5, test_bin6, test_bin7, test_bin8, test_bin9, test_bin10, test_bin11, test_bin12, test_bin13])
# val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5, val_bin6, val_bin7, val_bin8, val_bin9, val_bin10, val_bin11, val_bin12, val_bin13])


'''
EQUAL WIDTHS: Uncomment to run model with 13 classification bins
'''
# # Train data
# train_bin1 = train[train["popularity"] < 6].copy()
# train_bin2 = train[(train["popularity"] >= 6) & (train["popularity"] < 13)].copy()
# train_bin3 = train[(train["popularity"] >= 13) & (train["popularity"] < 20)].copy()
# train_bin4 = train[(train["popularity"] >= 20) & (train["popularity"] < 28)].copy()
# train_bin5 = train[(train["popularity"] >= 28) & (train["popularity"] < 35)].copy()
# train_bin6 = train[(train["popularity"] >= 35) & (train["popularity"] < 43)].copy()
# train_bin7 = train[(train["popularity"] >= 43) & (train["popularity"] < 50)].copy()
# train_bin8 = train[(train["popularity"] >= 50) & (train["popularity"] < 58)].copy()
# train_bin9 = train[(train["popularity"] >= 58) & (train["popularity"] < 65)].copy()
# train_bin10 = train[(train["popularity"] >= 65) & (train["popularity"] < 73)].copy()
# train_bin11 = train[(train["popularity"] >= 73) & (train["popularity"] < 85)].copy()
# train_bin12 = train[(train["popularity"] >= 85) & (train["popularity"] < 92)].copy()
# train_bin13 = train[(train["popularity"] >= 92) & (train["popularity"] <= 100)].copy()
#
# # Validation data
# val_bin1 = val[val["popularity"] < 6].copy()
# val_bin2 = val[(val["popularity"] >= 6) & (val["popularity"] < 13)].copy()
# val_bin3 = val[(val["popularity"] >= 13) & (val["popularity"] < 20)].copy()
# val_bin4 = val[(val["popularity"] >= 20) & (val["popularity"] < 28)].copy()
# val_bin5 = val[(val["popularity"] >= 28) & (val["popularity"] < 35)].copy()
# val_bin6 = val[(val["popularity"] >= 35) & (val["popularity"] < 43)].copy()
# val_bin7 = val[(val["popularity"] >= 43) & (val["popularity"] < 50)].copy()
# val_bin8 = val[(val["popularity"] >= 50) & (val["popularity"] < 58)].copy()
# val_bin9 = val[(val["popularity"] >= 58) & (val["popularity"] < 65)].copy()
# val_bin10 = val[(val["popularity"] >= 65) & (val["popularity"] < 73)].copy()
# val_bin11 = val[(val["popularity"] >= 73) & (val["popularity"] < 85)].copy()
# val_bin12 = val[(val["popularity"] >= 85) & (val["popularity"] < 92)].copy()
# val_bin13 = val[(val["popularity"] >= 92) & (val["popularity"] <= 100)].copy()
#
# # Test data
# test_bin1 = test[test["popularity"] < 8].copy()
# test_bin2 = test[(test["popularity"] >= 7) & (test["popularity"] < 13)].copy()
# test_bin3 = test[(test["popularity"] >= 13) & (test["popularity"] < 20)].copy()
# test_bin4 = test[(test["popularity"] >= 20) & (test["popularity"] < 28)].copy()
# test_bin5 = test[(test["popularity"] >= 28) & (test["popularity"] < 35)].copy()
# test_bin6 = test[(test["popularity"] >= 35) & (test["popularity"] < 43)].copy()
# test_bin7 = test[(test["popularity"] >= 43) & (test["popularity"] < 50)].copy()
# test_bin8 = test[(test["popularity"] >= 50) & (test["popularity"] < 58)].copy()
# test_bin9 = test[(test["popularity"] >= 58) & (test["popularity"] < 65)].copy()
# test_bin10 = test[(test["popularity"] >= 65) & (test["popularity"] < 73)].copy()
# test_bin11 = test[(test["popularity"] >= 73) & (test["popularity"] < 85)].copy()
# test_bin12 = test[(test["popularity"] >= 85) & (test["popularity"] < 92)].copy()
# test_bin13 = test[(test["popularity"] >= 92) & (test["popularity"] <= 100)].copy()
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
# train_bin11["popularity"], test_bin11["popularity"], val_bin11["popularity"] = 11, 11, 11
# train_bin12["popularity"], test_bin12["popularity"], val_bin12["popularity"] = 12, 12, 12
# train_bin13["popularity"], test_bin13["popularity"], val_bin13["popularity"] = 13, 13, 13
#
# train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5, train_bin6, train_bin7, train_bin8,
#                          train_bin9, train_bin10, train_bin11, train_bin12, train_bin13])
# test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5, test_bin6, test_bin7, test_bin8,
#                         test_bin9, test_bin10, test_bin11, test_bin12, test_bin13])
# val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5, val_bin6, val_bin7, val_bin8, val_bin9,
#                        val_bin10, val_bin11, val_bin12, val_bin13])


'''
EQUAL WIDTHS: Uncomment to run model with 16 classification bins
'''
# # Train data
# train_bin1 = train[train["popularity"] < 6].copy()
# train_bin2 = train[(train["popularity"] >= 6) & (train["popularity"] < 12)].copy()
# train_bin3 = train[(train["popularity"] >= 12) & (train["popularity"] < 18)].copy()
# train_bin4 = train[(train["popularity"] >= 18) & (train["popularity"] < 24)].copy()
# train_bin5 = train[(train["popularity"] >= 24) & (train["popularity"] < 30)].copy()
# train_bin6 = train[(train["popularity"] >= 30) & (train["popularity"] < 36)].copy()
# train_bin7 = train[(train["popularity"] >= 36) & (train["popularity"] < 42)].copy()
# train_bin8 = train[(train["popularity"] >= 42) & (train["popularity"] < 49)].copy()
# train_bin9 = train[(train["popularity"] >= 49) & (train["popularity"] < 55)].copy()
# train_bin10 = train[(train["popularity"] >= 55) & (train["popularity"] < 61)].copy()
# train_bin11 = train[(train["popularity"] >= 61) & (train["popularity"] < 67)].copy()
# train_bin12 = train[(train["popularity"] >= 67) & (train["popularity"] < 73)].copy()
# train_bin13 = train[(train["popularity"] >= 73) & (train["popularity"] < 80)].copy()
# train_bin14 = train[(train["popularity"] >= 80) & (train["popularity"] < 86)].copy()
# train_bin15 = train[(train["popularity"] >= 86) & (train["popularity"] < 92)].copy()
# train_bin16 = train[(train["popularity"] >= 92) & (train["popularity"] <= 100)].copy()
#
# # Validation data
# val_bin1 = val[val["popularity"] < 6].copy()
# val_bin2 = val[(val["popularity"] >= 6) & (val["popularity"] < 12)].copy()
# val_bin3 = val[(val["popularity"] >= 12) & (val["popularity"] < 18)].copy()
# val_bin4 = val[(val["popularity"] >= 18) & (val["popularity"] < 24)].copy()
# val_bin5 = val[(val["popularity"] >= 24) & (val["popularity"] < 30)].copy()
# val_bin6 = val[(val["popularity"] >= 30) & (val["popularity"] < 36)].copy()
# val_bin7 = val[(val["popularity"] >= 36) & (val["popularity"] < 42)].copy()
# val_bin8 = val[(val["popularity"] >= 42) & (val["popularity"] < 49)].copy()
# val_bin9 = val[(val["popularity"] >= 49) & (val["popularity"] < 55)].copy()
# val_bin10 = val[(val["popularity"] >= 55) & (val["popularity"] < 61)].copy()
# val_bin11 = val[(val["popularity"] >= 61) & (val["popularity"] < 67)].copy()
# val_bin12 = val[(val["popularity"] >= 67) & (val["popularity"] < 73)].copy()
# val_bin13 = val[(val["popularity"] >= 73) & (val["popularity"] < 80)].copy()
# val_bin14 = val[(val["popularity"] >= 80) & (val["popularity"] < 86)].copy()
# val_bin15 = val[(val["popularity"] >= 86) & (val["popularity"] < 92)].copy()
# val_bin16 = val[(val["popularity"] >= 92) & (val["popularity"] <= 100)].copy()
#
# # Test data
# test_bin1 = test[test["popularity"] < 6].copy()
# test_bin2 = test[(test["popularity"] >= 6) & (test["popularity"] < 12)].copy()
# test_bin3 = test[(test["popularity"] >= 12) & (test["popularity"] < 18)].copy()
# test_bin4 = test[(test["popularity"] >= 18) & (test["popularity"] < 24)].copy()
# test_bin5 = test[(test["popularity"] >= 24) & (test["popularity"] < 30)].copy()
# test_bin6 = test[(test["popularity"] >= 30) & (test["popularity"] < 36)].copy()
# test_bin7 = test[(test["popularity"] >= 36) & (test["popularity"] < 42)].copy()
# test_bin8 = test[(test["popularity"] >= 42) & (test["popularity"] < 49)].copy()
# test_bin9 = test[(test["popularity"] >= 49) & (test["popularity"] < 55)].copy()
# test_bin10 = test[(test["popularity"] >= 55) & (test["popularity"] < 61)].copy()
# test_bin11 = test[(test["popularity"] >= 61) & (test["popularity"] < 67)].copy()
# test_bin12 = test[(test["popularity"] >= 67) & (test["popularity"] < 73)].copy()
# test_bin13 = test[(test["popularity"] >= 73) & (test["popularity"] < 80)].copy()
# test_bin14 = test[(test["popularity"] >= 80) & (test["popularity"] < 86)].copy()
# test_bin15 = test[(test["popularity"] >= 86) & (test["popularity"] < 92)].copy()
# test_bin16 = test[(test["popularity"] >= 92) & (test["popularity"] <= 100)].copy()

# train_ready = pd.concat([train_bin1, train_bin2, train_bin3, train_bin4, train_bin5, train_bin6, train_bin7, train_bin8,
#                          train_bin9, train_bin10, train_bin11, train_bin12, train_bin13, train_bin14, train_bin15, train_bin16])
# test_ready = pd.concat([test_bin1, test_bin2, test_bin3, test_bin4, test_bin5, test_bin6, test_bin7, test_bin8, test_bin9,
#                         test_bin10, test_bin11, test_bin12, test_bin13, test_bin14, test_bin15, test_bin16])
# val_ready = pd.concat([val_bin1, val_bin2, val_bin3, val_bin4, val_bin5, val_bin6, val_bin7, val_bin8, val_bin9,
#                        val_bin10, val_bin11, val_bin12, val_bin13, val_bin14, val_bin15, val_bin16])





# print(train_bin1.shape[0], train_bin2.shape[0], train_bin3.shape[0], train_bin4.shape[0], train_bin5.shape[0],
#       train_bin6.shape[0], train_bin7.shape[0], train_bin8.shape[0], train_bin9.shape[0],
#       train_bin10.shape[0])# , train_bin11.shape[0], train_bin12.shape[0], train_bin13.shape[0])



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
print('Number of incorrectly classified songs:', sum(Y_val.ravel() != result))
differences = (Y_val.ravel()[np.where(Y_val.ravel() != result)]-result[np.where(Y_val.ravel() != result)])**2
print('Average difference in class for incorrectly classified songs:', math.sqrt(sum(differences)/sum(Y_val.ravel() != result)))
