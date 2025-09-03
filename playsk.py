import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import kagglehub

# Load data and create a dataframe
path = "/Users/ishaansendhil/.cache/kagglehub/datasets/dansbecker/melbourne-housing-snapshot/versions/5/melb_data.csv"
melbourne_data = pd.read_csv(path)

y = melbourne_data.Price

"""
How do we use categorical data?
    - we need to encode the data so models can understand it
    - this is a data preprocessing step    

Since many of our features in the dataset may be categorical, we have to be able to use them as features.
Suburbs are an example of a categorical feature, and since it is not ordinal, we have to use one hot encoding.
this turns the suburbs into binary columns so that machine learning models can work with them, and not treat
them as distinct categories with no false numerical meaning.
    - if the feature is ordinal (very strong, strong, etc) then we would use a different technique

"""


rooms = melbourne_data[['Rooms']]

suburbs = melbourne_data[['Suburb']]

# Creating our encoder object // sparse false makes it return numpy array instead of a sparse matrix (idk what that is)
encoder = OneHotEncoder(sparse_output=False)

# Now encoding the data
encoded_suburbs = encoder.fit_transform(suburbs)

"""
Now that we have our encoded column, we need to put it into a data frame, then concatonate it with the other 
features so that we can train the model
"""
suburbs_df = pd.DataFrame(encoded_suburbs, columns=encoder.get_feature_names_out(suburbs.columns)) #setting column names of df

"""
Axis=1 stacks columns side by side

reset index makes the indicies start from 0 so that concatonating doesn't misalign them. drop=true discards old index instead 
of keeping a column (inplace=true would modify in place as another parameter)
"""
X = pd.concat([rooms.reset_index(drop=True), suburbs_df.reset_index(drop=True)], axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# creating the decsision tree objecting and training it on the training data from above
model = DecisionTreeRegressor(random_state=1)

scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')

print(scores)
"""
outputs [-247684.81806187 -252741.15885361 -244061.93189833 -250398.218159
 -251545.64020982]

 which is ALOT closer than before (probably because we aren't only using 1 feature)

 next time: implement the rest of the features, feature engineering? random forest
    - fix the obvious underfitting
"""