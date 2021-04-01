# ClimateAi Coding Challenge
# By Scott Burstein

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Part 1: Summary Statistics and Data Transformation

df = pd.read_csv("RiverData.csv")

plt.hist(df.flow, bins = 100)
plt.xlabel("River Flow (m^3/s)")
plt.ylabel("Frequency")
plt.title("Measurements taken between 1/4/1958 and 12/31/2015", size="small")
plt.suptitle("Average Daily River Flow")
plt.xlim(0, 40)
plt.show()
plt.savefig("DailyRiverFlow_Histogram")

'''
# 1 day prior
N = 1

# target measurement of flow
feature = "obs_tas_1"

# total number of rows
rows = df.shape[0]

# a list representing Nth prior measurements of feature
# notice that the front of the list needs to be padded with N
# None values to maintain the constistent rows length for each N
nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]

# make a new column name of feature_N and add to DataFrame
col_name = "{}_{}".format(feature, N)
df[col_name] = nth_prior_measurements
'''

def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

features = ["date",
            "obs_tas_1", "obs_tas_2", "obs_tas_3",
            "obs_tas_4", "obs_tas_5", "obs_tas_6", 
            "obs_tas_7", "obs_tas_8", "obs_tas_9",
            "obs_pr_1", "obs_pr_2", "obs_pr_3", 
            "obs_pr_4", "obs_pr_5", "obs_pr_6", 
            "obs_pr_7", "obs_pr_8", "obs_pr_9"]


for feature in features:
    if feature != 'date':
        for N in range(1, 4): # N days (1,4) is 3 days prior
            derive_nth_day_feature(df, feature, N)

# make list of original features without desired variables
to_remove = [feature 
             for feature in features 
             if feature not in ["obs_tas_1", "obs_tas_2", "obs_tas_3", 
                                "obs_tas_4", "obs_tas_5", "obs_tas_6", 
                                "obs_tas_7", "obs_tas_8", "obs_tas_9",
                                "obs_pr_1", "obs_pr_2", "obs_pr_3", 
                                "obs_pr_4", "obs_pr_5", "obs_pr_6", 
                                "obs_pr_7", "obs_pr_8", "obs_pr_9"
                                ]
            ]

# make a list of columns to keep
to_keep = [col for col in df.columns if col not in to_remove]

# select only the columns in to_keep and assign to df
df = df[to_keep]
df.columns

print("Dataframe df index dtype and columns, non-null values and memory usage:")
print(df.info())

# Call describe on df and transpose it due to the large number of columns
spread = df.describe().T

# precalculate interquartile range for ease of use in next calculation
IQR = spread['75%'] - spread['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))

# Display the features containing extreme outliers:

#spread.info()
#spread.ix[spread.outliers,]
#print(spread)

#It appears that precipitation measurements have 3 IQR outliers, where as temperature values do not.

'''
Example Histogram of the precipitation measurements taken at observation site 1. 
This gives us an idea of the typical distribution of values at a given observation 
site for precipiation values
'''

#%matplotlib inline
plt.rcParams['figure.figsize'] = [14, 8]
df.obs_pr_1.hist()
plt.title('Distribution of Precipitation at Location 1')
plt.xlabel('obs_pr_1')
plt.show()
plt.savefig("obs_pr_1-histogram")

'''
These perceived outliers are in fact not unusual, but rather a consequence of usual precipitation 
patterns in the region. It makes sense that most days have no precipitation, and then some days have a lot.
Many underlying statistical methods assume a normal distribution of data, so it will be important to consider 
this later on.
''' 

#-----------------------------------------
# Part 2: Regression Analysis

df.corr()[['flow']].sort_values('flow')
#print(df.corr()[['flow']].sort_values('flow'))

'''
Since the range of correlation coefficients between the indepenent variables and flow variable 
have a small range (0.200689 - 0.353181), I will not remove any input variables for my prediction.
'''

# make list of prediction features without desired variables

predictors = [var 
             for var in df.columns 
             if var not in ["flow", "date",
                            "obs_tas_1", "obs_tas_2", "obs_tas_3", 
                            "obs_tas_4", "obs_tas_5", "obs_tas_6", 
                            "obs_tas_7", "obs_tas_8", "obs_tas_9",
                            "obs_pr_1", "obs_pr_2", "obs_pr_3", 
                            "obs_pr_4", "obs_pr_5", "obs_pr_6", 
                            "obs_pr_7", "obs_pr_8", "obs_pr_9"
                            ]
            ]

df2 = df[['flow'] + predictors]

#%matplotlib inline

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [18, 22]

# call subplots specifying the grid structure we desire and that 
# the y axes should be shared
fig, axes = plt.subplots(nrows=9, ncols=6, sharey=True, constrained_layout=True)

# Since it would be nice to loop through the features in to build this plot
# let us rearrange our data into a 2D array of 9 rows and 6 columns
arr = np.array(predictors).reshape(9, 6)

# use enumerate to loop over the arr 2D array of rows and columns
# and create scatter plots of each flow vs each feature
for row, col_arr in enumerate(arr):
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2['flow'])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel='flow')
        else:
            axes[row, col].set(xlabel=feature)
plt.show()
plt.savefig("flow_prediction_var_corrs")

'''
From the plots above it is apparent that none of the predictor variables have a 
linear relationship with the response variable (flow). That being said, all of the 
temperature predictor subplots show a left-skew distribution, whereas the precipitation 
predictor subplots' values are concentrated at values below 40 mm.
'''

# separate out my predictor variables (X) from my outcome variable y
# Also remove first 3 rows where X has NaN values
X = df2[predictors].dropna()
y = df2['flow'].drop([0, 1, 2])


# import the relevant module
import statsmodels.api as sm

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)
X.iloc[:5, :5]

# (1) select a significance value
alpha = 0.05

# (2) Fit the model
model = sm.OLS(endog = y, exog = X).fit()

# (3) evaluate the coefficients' p-values
model.summary()
print("Model Summary:")
print(model.summary())

from sklearn.model_selection import train_test_split

# first remove the const column because unlike statsmodels, SciKit-Learn will add that in for us
#X = X.drop('const', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

from sklearn.linear_model import LinearRegression

# instantiate the regressor class
regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f m^3/s flow" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f m^3/s flow" % median_absolute_error(y_test, prediction))

#-----------------------------------------
# Part 3: Using ML to Predict River Flow

import tensorflow as tf
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split

# Create new df for part 3 with date added back:
#df2['date'] = pd.date_range(start='1/4/1958', periods=len(df2), freq='D')
#df2.set_index('date')

# Remove rows 0,1,2 where Null values present for columns representing 
# precip. and temp. measurements from date-1, date-2, date-3 days
df2 = df2.drop([0,1,2])

# execute the describe() function and transpose the output so that it doesn't overflow the width of the screen
df2.describe().T
df2.info()

# X will be a pandas dataframe of all columns except flow
X = df2[[col for col in df2.columns if col != 'flow']]

# y will be a pandas series of the flow
y = df2['flow']

# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

# take the remaining 20% of data in X_tmp, y_tmp and split them evenly
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[50, 50],
                                      model_dir='tf_wx_model')

def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
    return(
    tf.compat.v1.estimator.inputs.pandas_input_fn(x=X,
                                                  y=y,
                                                  num_epochs=num_epochs,
                                                  shuffle=shuffle,
                                                  batch_size=batch_size))

evaluations = []
STEPS = 400
for i in range(100):
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))

#%matplotlib inline
# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)
plt.xlabel('Training steps (Epochs = steps / 2)')
plt.ylabel('Loss (SSE)')
plt.show()
plt.savefig("Loss_SSE_Graph")

pred = regressor.predict(input_fn=wx_input_fn(X_test,
                                              num_epochs=1,
                                              shuffle=False))  

predictions = np.array([p['predictions'][0] for p in pred])

print("The Explained Variance: %.2f" % explained_variance_score(
                                            y_test, predictions))  
print("The Mean Absolute Error: %.2f m^3/s flow" % mean_absolute_error(
                                            y_test, predictions))  
print("The Median Absolute Error: %.2f m^3/s flow" % median_absolute_error(
                                            y_test, predictions))

'''
The Explained Variance: 0.38
The Mean Absolute Error: 4.59 m^3/s flow
The Median Absolute Error: 2.41 m^3/s flow
'''