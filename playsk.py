import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import kagglehub

# Load data and create a dataframe
path = "/Users/ishaansendhil/.cache/kagglehub/datasets/dansbecker/melbourne-housing-snapshot/versions/5/melb_data.csv"
melbourne_data = pd.read_csv(path)

"""
selecting our target 'y' as the price column in our dataframe
and selecting our feature 'x' as the rooms column in our dataframe
we will essentially use x to predict y
"""
y = melbourne_data.Price
x = melbourne_data[['Rooms']]

"""
To improve our splitting and training / testing of our model, we can instead use K-fold CV
train_test_split is not optimal as:
    - the random sample trained on may be biased
    - might overfit to the whole test set
    - small changes in the split may change performance drastically

K-fold Cv allows us to split the model into k equal parts, fitting the model on k-1 parts and testing
on the kth subset. This process is then repeated k times so that the final performance score is the
average of all scores.
    - this results in a stable estimate of the model
    - less risk of a bad random split
    - no single test set, harder to overfit

"""

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# creating the decsision tree objecting and training it on the training data from above
model = DecisionTreeRegressor(random_state=1)

"""
we can use cross_val_score to automate the proccess once we have our KFold object. This function
splits our data using a cross validation (kf in our case), trains, tests, and returns a numpy array of 
all the scores from each fold
"""
scores = cross_val_score(model, x, y, cv=kf, scoring='neg_mean_absolute_error')

print(scores)

"""
This results in [-384217.14991043 -394535.25636982 -371254.0944823  -386999.7817237
 -392045.69796707]

 Which highlights how much the scores vary from each split and why CV is helpful

 We are only using one feature, how do we input categorical data?
    - decision trees are feature hungry, we need to find more splits for the model
"""