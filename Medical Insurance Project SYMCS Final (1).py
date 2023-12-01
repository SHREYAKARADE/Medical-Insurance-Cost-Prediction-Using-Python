#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


import pandas as pd
import random



# In[3]:


# Create an empty DataFrame
data = pd.DataFrame()


# In[4]:


# Define the number of records and columns
num_records = 1000
num_columns = 10


# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_records = 1000

# Randomly generate values for each feature
age = np.random.uniform(18, 65, n_records)
bmi = np.random.uniform(15, 40, n_records)
children = np.random.randint(0, 4, n_records)
sex = np.random.choice(['Male', 'Female'], n_records)
smoker = np.random.choice(['Yes', 'No'], n_records)
region = np.random.choice(['East', 'North', 'South', 'West'], n_records)
cholesterol = np.random.choice(['High', 'Normal'], n_records)
blood_pressure = np.random.choice(['High', 'Normal'], n_records)
diabetes = np.random.choice(['Yes', 'No'], n_records)

# Generate insurance cost based on a linear combination of features
insurance_cost = 300 * age + 100 * bmi + 200 * children + \
                 3000 * (sex == 'Female') - 5000 * (sex == 'Male') + \
                 15000 * (smoker == 'Yes') - 2000 * (smoker == 'No') + \
                 100 * (region == 'East') + 50 * (region == 'North') + \
                 30 * (region == 'South') + 20 * (region == 'West') + \
                 2000 * (cholesterol == 'High') + 1000 * (cholesterol == 'Normal') + \
                 1000 * (blood_pressure == 'High') + 1000 * (blood_pressure == 'Normal') + \
                 700 * (diabetes == 'Yes') + 500 * (diabetes == 'No') + \
                 np.random.normal(0, 1000, n_records)

# Create a DataFrame
data_encoded = pd.DataFrame({
    'Age': age,
    'BMI': bmi,
    'Children': children,
    'Sex': sex,
    'Smoker': smoker,
    'Region': region,
    'Cholesterol': cholesterol,
    'BloodPressure': blood_pressure,
    'Diabetes': diabetes,
    'InsuranceCost': insurance_cost
})

print(data_encoded)


# In[6]:


data_encoded


# In[7]:


# Apply label encoding to categorical columns
label_encoder = LabelEncoder()
categorical_columns = ['Sex', 'Smoker', 'Region', 'Cholesterol', 'BloodPressure', 'Diabetes']
for col in categorical_columns:
    data_encoded[col] = label_encoder.fit_transform(data_encoded[col])


# In[8]:


# Save the dataset to a CSV file
data_encoded.to_csv('medicaldataset.csv', index=False) 


# In[9]:


#used to get concise summary of dataframe
data_encoded.info()


# In[10]:


# used to view basic statistical details like percentile, mean, std etc of dataframe
data_encoded.describe()


# In[11]:


# used to return tuple of shape (rows, columns) of dataframe
data_encoded.shape


# In[12]:


# Total number of elements in dataframe
data_encoded.size


# In[13]:


# missing values

missing_values = data_encoded.isna().sum()
print("Missing values in the dataset:")
print(missing_values)


# In[14]:


missing_values = data_encoded.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)


# In[15]:


data_encoded.head()


# In[16]:


data_encoded.columns


# In[17]:


dependent_variable = "InsuranceCost"


# In[18]:


#Step 3: Drop columns that aren’t useful (Unnamed Column)
df = data_encoded[['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region', 'Cholesterol',
       'BloodPressure', 'Diabetes', 'InsuranceCost']]
df.head()


# In[19]:


#Checking null values
print(df.isnull().sum())
print("No. of rows:",len(df.axes[0]))


# In[20]:


#Checking duplicate rows
print("No. of Duplicated Rows:", df.duplicated().sum())


# In[21]:


data_encoded.columns = df.columns.tolist()
print(data_encoded.columns)


# In[22]:


from sklearn.model_selection import train_test_split

#Step 5: Splitting Data into Train and Split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[23]:


print(pd.DataFrame(X_train).head())


# In[24]:


X.head()


# In[25]:


print(pd.DataFrame(y_train).head())


# In[26]:


y.head()


# In[27]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# Extract features and target variable
features = data_encoded[['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region', 'Cholesterol',
       'BloodPressure', 'Diabetes']]
target = data_encoded['InsuranceCost']

# Apply Min-Max scaling to features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Display the scaled features DataFrame
print(features_scaled_df)


# In[28]:


# Concatenate features and target into a single DataFrame for correlation analysis
data_for_correlation = pd.concat([features_scaled_df, target], axis=1)

# Calculate the correlation matrix
correlation_matrix = data_for_correlation.corr().abs()

# Display the correlation matrix
print(correlation_matrix)

import pandas as pd

correlation_matrix = data_encoded.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

# Step 8: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test data:", accuracy)
# In[29]:


# printing unique items from columns 
def uniquevals(col):
    print(f'Details of the particular col {col} is : {df[col].unique()}')
    
for col in df.columns:
    uniquevals(col)
    print("-"*75)


# In[30]:


# printing items counts from columns
def valuecounts(col):
    print(f'Valuecounts of the particular col {col} is : {df[col].value_counts()}')  
    
for col in df.columns:
    valuecounts(col)
    print("-"*75)


# In[31]:


#univariate analysis


# In[32]:


# viewing the distribution of the InsuranceCost column
import seaborn as sns
sns.distplot(df['InsuranceCost'],color='red')


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_encoded' DataFrame and 'dependent_variable' is the column name you want to plot
plt.figure(figsize=(5, 6))
sns.boxplot(y=data_encoded['InsuranceCost'])
plt.show()


# In[34]:


numerical_var = list(data_encoded.describe().columns[1:])
numerical_var


# In[35]:


# Count plot for the categorical features

for col in numerical_var:
  plt.figure(figsize=(10,12))
  sns.countplot(data=data_encoded,x=col)
  plt.title(col)
  plt.show()


# In[36]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming 'data_encoded' is your DataFrame and 'numerical_var' is a list of numerical column names
for col in numerical_var:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = data_encoded[col]
    label = data_encoded['InsuranceCost']
    
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('InsuranceCost')
    ax.set_title('InsuranceCost vs ' + col)

    z = np.polyfit(data_encoded[col], data_encoded['InsuranceCost'], 1)
    y_hat = np.poly1d(z)(data_encoded[col])

    plt.plot(data_encoded[col], y_hat, "r--", lw=1)

plt.show()


# In[37]:


import matplotlib.pyplot as plt

# List of categorical variables you want to create pie charts for
categorical_variables = ['Region', 'Sex', 'Cholesterol', 'BloodPressure', 'Diabetes']

# Set up subplots for the pie charts
fig, axes = plt.subplots(1, len(categorical_variables), figsize=(15, 7))

# Iterate through each categorical variable and create a pie chart
for i, categorical_variable in enumerate(categorical_variables):
    category_counts = data_encoded[categorical_variable].value_counts()
    axes[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    axes[i].set_title(f'Distribution of {categorical_variable}')

plt.tight_layout()
plt.show()


# In[38]:


import matplotlib.pyplot as plt

# Define your features (independent variables) and the target variable
features =  ['Age', 'Sex', 'BMI', 'Smoker', 'Children']
target = 'InsuranceCost'

# Set the figure size
plt.figure(figsize=(12, 14))

# Iterate through each feature and create a histogram
for i, feature in enumerate(features, start=1):
    plt.subplot(len(features), 1, i)
    plt.hist(data_encoded[feature], bins=20, edgecolor='k')
    plt.title(f'{feature} Histogram')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()


# In[39]:


# Define your features (independent variables) and the target variable
features =  ['Region', 'Cholesterol', 'BloodPressure', 'Diabetes']
target = 'InsuranceCost'

# Set the figure size
plt.figure(figsize=(12, 14))

# Iterate through each feature and create a histogram
for i, feature in enumerate(features, start=1):
   plt.subplot(len(features), 1, i)
   plt.hist(data_encoded[feature], bins=20, edgecolor='k')
   plt.title(f'{feature} Histogram')
   plt.xlabel(feature)
   plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()


# In[40]:


# We required all the features to train the model to predict InsuranceCost 


# In[41]:


#Correlation of independent variables and target variable


# In[42]:


import pandas as pd

# Define your features (independent variables) and the target variable
features = ['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region', 'Cholesterol',
       'BloodPressure', 'Diabetes']
target = 'InsuranceCost'

# Calculate the correlation coefficients
correlations = data_encoded[features + [target]].corr().abs()

# Display the correlation coefficients
print(correlations[target])

#Positive values indicate a positive correlation, negative values indicate a negative correlation, 
#and values close to 0 suggest a weak correlation


# In[43]:


X = data_encoded.drop(columns='InsuranceCost',axis=1)
Y = data_encoded['InsuranceCost']


# In[44]:


print(X)


# In[45]:


print(Y)


# In[46]:


# K-Means algorithm
from sklearn.cluster import KMeans
import pandas as pd

# Select the relevant features for clustering
features = ['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region', 'Cholesterol',
       'BloodPressure', 'Diabetes']

# Select the data
X = data_encoded[features]

# Choose the number of clusters (you can adjust this based on your needs)
num_clusters = 3


# Create and fit the K-Means model
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
kmeans.fit(X)

# Add the cluster labels to the DataFrame
data_encoded['Cluster'] = kmeans.labels_

# Now, you can analyze the clusters or use them for further tasks
print(data_encoded['Cluster'].value_counts())


# In[47]:


print(df.columns)


# In[48]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("medical_insurance_cost_label_encoded.csv")


# Define your features and target variable
features_scaled_df = ['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region', 'Cholesterol',
       'BloodPressure', 'Diabetes']
target = 'InsuranceCost'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features_scaled_df], df[target], test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')


# In[49]:


accuracy_score = r2 * 100
print(f"accuracy_score: {accuracy_score}%")


# In[50]:


model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test,y_test))


# In[51]:


# Create and train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Use the trained model to predict the target variable on the test set
y_pred = regressor.predict(X_test)

# Create a DataFrame to display predicted and actual values side by side
result_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})

# Display the DataFrame
print(result_df)

# Plotting the best-fit line along with the scatter plot
plt.scatter(result_df['Actual'], result_df['Predicted'], label='Actual vs Predicted')
plt.plot([min(result_df['Actual']), max(result_df['Actual'])], [min(result_df['Actual']), max(result_df['Actual'])], linestyle='-', color='orange', label='Best-Fit Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values with Best-Fit Line')
plt.legend()
plt.show()


# In[52]:


# Predicting Output for row 1 from csv file


# In[53]:


# Load the entire dataset
df = pd.read_csv("medicaldataset.csv")

# Select a specific row from your dataset for prediction
row_to_predict = df.iloc[[0]]  # Change the index [0] to any row index you want to predict

# Extract features from the selected row
features_to_predict = row_to_predict[features_scaled_df]

# Make predictions on the selected row
predicted_insurance_cost = model.predict(features_to_predict)

print(f"Predicted Insurance Cost: {predicted_insurance_cost[0]}")


# In[54]:


pip install xgboost


# In[55]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Load your dataset
df = pd.read_csv("medical_insurance_cost_label_encoded.csv")

# Define your features and target variable
features = ['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region', 'Cholesterol',
       'BloodPressure', 'Diabetes']
target = 'InsuranceCost'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Create an XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=100)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Calculate metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Mean Squared Error: {mse_xgb}')
print(f'XGBoost R-squared: {r2_xgb}')


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost import XGBRegressor

# Hyperparameters for XGBRegressor
xgb_params = {'max_depth':[5,6,7,8],
              'learning_rate':[0.05,0.1,0.2,0.3],
            'n_estimators':[80,100,150],
            'colsample_bytree':[0.5,0.7]
            }

# creating instance of XGBRegressor
xgb_reg = XGBRegressor()

# Grid Search
xgb_grid = GridSearchCV(estimator= xgb_reg, param_grid= xgb_params, cv =3,
                        scoring= 'neg_mean_squared_error', verbose=2)

# training the model on the xgb_grid
xgb_grid.fit(X_train,y_train)


# In[ ]:


# evaluating the best parameters
xgb_grid.best_params_


# In[ ]:


# creating best estimator model
xgb_optimal_model = xgb_grid.best_estimator_


# In[ ]:


# model score
xgb_optimal_model.score(X_train, y_train)


# In[ ]:


# making prediction on train and test data
train_pred_xgb = xgb_optimal_model.predict(X_train)
test_pred_xgb = xgb_optimal_model.predict(X_test)


# In[ ]:


#  plotting the graph of actual and preddicted values
plt.figure(figsize=(12,6))
plt.plot(test_pred_xgb[:100])
plt.plot(np.array(y_test)[:100])
plt.legend(["Predicted","Actual"])
plt.show()


# In[ ]:


# model score
xgb_optimal_model.score(X_train, y_train)


# In[ ]:


# Plotting the best-fit line
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='-', color='orange', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values with Best-Fit Line (XGBoost)')
plt.show()


# In[ ]:


# making prediction on train and test data
train_pred_xgb = xgb_optimal_model.predict(X_train)
test_pred_xgb = xgb_optimal_model.predict(X_test)


# In[ ]:


accuracy_score = r2 * 100
print(f"accuracy_score: {accuracy_score}%")


# In[ ]:


xgb_model.fit(X_train, y_train)
ypredtrain4=xgb_model.predict(X_train)
ypredtest4=xgb_model.predict(X_test)
print(r2_score(y_train,ypredtrain4))
print(r2_score(y_test,ypredtest4))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)


mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Mean Absolute Error: {mae_rf}')
print(f'Random Forest R-squared: {r2_rf}')


# In[ ]:


rf_model.fit(X_train, y_train)
ypredtrain2=rf_model.predict(X_train)
ypredtest2=rf_model.predict(X_test)
print(r2_score(y_train,ypredtrain2))
print(r2_score(y_test,ypredtest2))


# In[ ]:


accuracy_score = r2 * 100
print(f"accuracy_score: {accuracy_score}%")


# In[ ]:


# Plotting the scatter plot
plt.scatter(y_test, y_pred_rf)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='-', color='orange', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values with Best-Fit Line (Random Forest)')
plt.show()


# In[ ]:


from sklearn.svm import SVR

# Create an SVM regressor
svm_model = SVR(kernel='linear')

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test)


mae_svm = mean_absolute_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)


print(f'SVM Mean Absolute Error: {mae_svm}')
print(f'SVM R-squared: {r2_svm}')


# In[ ]:


# Plotting the scatter plot
plt.scatter(y_test, y_pred_svm)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='-', color='orange', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values with Best-Fit Line (SVM)')
plt.show()


# In[ ]:


svm_model.fit(X_train, y_train)
ypredtrain1=svm_model.predict(X_train)
ypredtest1=svm_model.predict(X_test)
print(r2_score(y_train,ypredtrain1))
print(r2_score(y_test,ypredtest1))


# In[ ]:


accuracy_score = r2 * 100
print(f"accuracy_score: {accuracy_score}%")


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

# Create a Gradient Boosting regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_gb = gb_model.predict(X_test)


mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)


print(f'Gradient Boosting Mean Absolute Error: {mae_gb}')
print(f'Gradient Boosting R-squared: {r2_gb}')


# In[ ]:


# Plotting the scatter plot
plt.scatter(y_test, y_pred_gb)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='-', color='orange', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values with Best-Fit Line (Gradient Boosting)')
plt.show()


# In[ ]:


gb_model.fit(X_train, y_train)
ypredtrain3=gb_model.predict(X_train)
ypredtest3=gb_model.predict(X_test)
print(r2_score(y_train,ypredtrain3))
print(r2_score(y_test,ypredtest3))


# In[ ]:


accuracy_score = r2 * 100
print(f"accuracy_score: {accuracy_score}%")


# In[ ]:





# In[ ]:




