## https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
## download data from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

np.random.seed(123)

data = pd.read_csv('../input/data.csv')

# Removing the Id and the Unnamed columns
data = data.iloc[:,1:-1]

# Next, we encode the Categorical Variable
label_encoder = LabelEncoder()
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')

# Generating the correlation matrix
corr = data.corr()

# Generating the correlation heat-map
sns.heatmap(corr)


# Next, we compare the correlation between features and remove one of two features that have a correlation higher than 0.9
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = Falseselected_columns = data.columns[columns]data = data[selected_columns]

# Now, the dataset has only those columns with correlation less than 0.9


# Selecting columns based on p-value
# Next we will be selecting the columns based on how they affect the p-value. We are the removing the column diagnosis because it is the column we are trying to predict
selected_columns = selected_columns[1:].valuesimport statsmodels.formula.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columnsSL = 0.05

data_modeled, selected_columns = backwardElimination(data.iloc[:,1:].values, data.iloc[:,0].values, SL, selected_columns)

# This is what we are doing in the above code block:
#  - We assume to null hypothesis to be “The selected combination of dependent variables do not have any effect on the independent variable”.
#  - Then we build a small regression model and calculate the p values.
#  - If the p values is higher than the threshold, we discard that combination of features.


# Next, we move the result to a new Dataframe.
result = pd.DataFrame()
result['diagnosis'] = data.iloc[:,0]

# Creating a Dataframe with the columns selected using the p-value and correlation
data = pd.DataFrame(data = data_modeled, columns = selected_columns)




# Visualizing the selected features

# Plotting the data to visualize their distribution

fig = plt.figure(figsize = (20, 25))
j = 0
for i in data.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(data[i][result['diagnosis']==0], color='g', label = 'benign')
    sns.distplot(data[i][result['diagnosis']==1], color='r', label = 'malignant')
    plt.legend(loc='best')
fig.suptitle('Breast Cance Data Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()



# Now we split the data to train and test set. 20% of the data is used to create the test data and 80% to create the train data
x_train, x_test, y_train, y_test = train_test_split(data.values, result.values, test_size = 0.2)


# Building a model with the selected features
# We are using a Support Vector Classifier with a Gaussian Kernel to make the predictions. We will train the model on our train data and calculate the accuracy of the model using the test data

svc=SVC() # The default kernel used by SVC is the gaussian kernel
svc.fit(x_train, y_train)

# Making the predictions and calculating the accuracy

prediction = svc.predict(x_test)

# We are using a confusion matrix here

cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)

# The accuracy obtained was 0.9298245614035088
