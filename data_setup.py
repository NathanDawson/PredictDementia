# Create X and y Variables
# In[36]:


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

# Use the dataframe after Exploratory_Analysis.py has ran
df = pd.read_csv('Data/df_after_pre-processing.csv')

# Create X and y variables
X = df.drop(['CDR', 'MRI_ID'], axis=1)
y = df['CDR']

# Transform y variable to numeric
# In[69]:


# Transform categorical CDR feature to numeric format, retaining order
encoder = OrdinalEncoder(categories=[['none', 'very_mild', 'mild', 'moderate']])
y = encoder.fit_transform(y.values.reshape(-1, 1)).ravel()

# Assign CDR score of Moderate to Mild
y[y == 3] = 2

# Check Moderate class (3) has been removed
print()
print("Ensure Moderate class has been moved into Mild class:", (y == 3).sum() == 0)
print()

# Split Data
# In[70]:


# 'X.ID' contains the patient IDs for grouping
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, X.ID, test_size=0.2, random_state=42, stratify=y)

# Reset indices to align them for grouping purposes
X_train = X_train.reset_index(drop=True)
y_train = pd.Series(y_train).reset_index(drop=True)
groups_train = groups_train.reset_index(drop=True)

# Create Nested Cross Validations
# In[71]:


# Outer CV for model evaluation
outer_cv = StratifiedGroupKFold(n_splits=10)

# Inner CV for model selection (feature selection and hyperparameter tuning)
inner_cv = StratifiedGroupKFold(n_splits=10)

df.to_csv('Data/df_after_data_setup.csv', index=False)
