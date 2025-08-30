import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
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
Using sklearn's train_test_split() method to split our data into training and validation data
we will use 80% of our data for training and 20% for validation

This is done to avoid overfitting our model, but is not the most optimal way to do so since we 
are only training the model on a single split of the data when we could do multiple iterations of 
splitting and training.   
"""
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# creating the decsision tree objecting and training it on the training data from above
model = DecisionTreeRegressor(random_state=1)
model.fit(train_x, train_y)

# predicting values of the validation data
predictions = model.predict(val_x)

# Calculating the MAE (average of the sum of the differences between prediction and actual value (mouthful))
print(mean_absolute_error(val_y, predictions))

# the output is ~380236.6. How can we improve?