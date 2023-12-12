# Create X and y Variables
# In[36]:


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE

# Use the dataframe after Exploratory_Analysis.py has ran
df = pd.read_csv('data/df_after_pre-processing.csv')

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

# Split data
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
