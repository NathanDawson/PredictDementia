# In[10]:


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Use the dataframe after Exploratory_Analysis.py has ran
df = pd.read_csv('Data/df_after_eda.csv')

# Check for object type features
objects = df.select_dtypes(include=[object])
print()
print("Object Type Features:", objects.head())
print()
# Investigate ASF Column
# In[11]:


# Convert ASF to numeric
converted_column = pd.to_numeric(df['ASF'], errors='coerce')

# Check for rows which could not be converted
print()
print("Check for rows which could not be converted:", df['ASF'][converted_column.isna() & df['ASF'].notna()])
print()
# In[12]:


# Look up row which could not be converted
print()
print("Row which could not be converted:", df.loc[128])
print()
# In[13]:


# Fix row to allow for conversion to numeric
df.at[128, 'ASF'] = df.at[128, 'ASF'].replace(',', '.')

# In[14]:


# Look up row to ensure it is correctly formatted
print()
print("Ensure row correctly formatted:", df.loc[128])
print()
# In[15]:


# Convert column to numeric
df['ASF'] = pd.to_numeric(df['ASF'], errors='coerce')

# OneHotEncode Columns: Sex, Hand, CDR
# In[17]:


columns_to_encode = df[['sex', 'hand', 'CDR']]

# One Hot Encode columns selected
enc = OneHotEncoder()
enc.fit(columns_to_encode)
onehotlabels = enc.transform(columns_to_encode).toarray()

# In[18]:


# Assign One Hot Encoded values to columns selected
encoded_df = pd.DataFrame(onehotlabels, columns=enc.get_feature_names_out())

# In[19]:


# Check onehotlabel feature names
print()
print("OneHotEncoded Feature Names:", enc.get_feature_names_out())
print()
# Fix CDR very mild and CDR mild feature names
# In[20]:


cdr_very_mild_columns = ['CDR_very mild', 'CDR_very midl', 'CDR_very miId', 'CDR_vry mild']

# Join misspelt columns together
encoded_df['CDR_very_mild'] = encoded_df[cdr_very_mild_columns].sum(axis=1)
encoded_df['CDR_mild'] = encoded_df[['CDR_mild', 'CDR_midl']].sum(axis=1)

# In[21]:


# Drop misspelt columns
encoded_df.drop(cdr_very_mild_columns, axis=1, inplace=True)
encoded_df.drop('CDR_midl', axis=1, inplace=True)

# In[22]:


# Drop original columns which have been one-hot encoded
df.drop(['sex', 'hand', 'CDR'], axis=1, inplace=True)

# Join one-hot encoded columns into dataframe
df = df.reset_index(drop=True)
encoded_df = encoded_df.reset_index(drop=True)
df = pd.concat([df, encoded_df], axis=1)

#  Check for Duplicated Data
# In[23]:

print()
print("Check for Duplicated Data:", df.duplicated().sum())
print()
# Analyse Missing Data
# In[24]:


# Total number of missing data per column
print()
print("Total number of missing data per column:", df.isnull().sum())
print()
# In[25]:


# Calculate the percentage of missing data per column
missing_percent = df.isnull().sum() * 100 / len(df)

# Filter out columns with no missing data
missing_percent = missing_percent[missing_percent > 0]

# Plot
plt.figure(figsize=(12, 6))
missing_percent.plot(kind='bar')
plt.xlabel('Columns with Missing Data')
plt.ylabel('Percentage of Missing Data')
plt.title('Percentage of Missing Data per Column')
plt.xticks(rotation=45)

# Save figure
plt.savefig('Graphs/Pre-Processing/Percentage_of_Missing_Data.png', bbox_inches='tight')

# In[26]:


columns_to_impute = ['SES', 'MMSE', 'eTIV', 'ASF']
imputation_data = df[columns_to_impute]

# Impute using MICE
# In[27]:


mice_imputer = IterativeImputer(max_iter=10, random_state=0)

imputed_data_mice = mice_imputer.fit_transform(imputation_data)

MICE_Imputed = pd.DataFrame(imputed_data_mice, columns=columns_to_impute)

print()
print("Ensure data has been imputed:", MICE_Imputed.isnull().sum())
print()
# In[29]:


# Investigate imputed data
print()
print("Investigate imputed data:", MICE_Imputed.head())
print()
# Investigate SES Feature
# In[30]:


# Check columns unique values
print()
print("Check imputed columns unique values:", MICE_Imputed.SES.unique())
print()
# In[31]:


# Round imputed values to be in range 1 - 5 as intended
limit_values = np.clip(MICE_Imputed.SES, 1, 5)

round_SES = np.rint(limit_values)

print()
print("Ensure values have been rounded to numbers 1 - 5:", round_SES.unique())
print()
# In[32]:


# Update column with rounded values
MICE_Imputed.SES = round_SES

# In[33]:


# Update dataframe with imputed data
df_without_missing = df.drop(columns=columns_to_impute)

df = pd.concat([df_without_missing, MICE_Imputed], axis=1)

print()
print("Ensure dataframe contains imputed data with no missing data present:", df.isnull().sum())
print()

# Investigate ASF column for outliers
# In[34]:


# Investigate ASF Column for outliers
ASF_outliers = df[df['ASF'] > 2]

print()
print(ASF_outliers)
print()

# In[35]:


# Update ASF value
df.loc[df['MRI_ID'] == '0185_MR1', 'ASF'] = 1.03

# Verify change
print(df[df['MRI_ID'] == '0185_MR1'])

# Reverse One-Hot Encoding of Target Variable (CDR)
# In[34]:


def get_cdr(row):
    """
    Remove sub-string before underscore(_)

    Removes the first part of the column names, only leaving the categories.
    Example:
    CDR_mild = mild

    Returns:
    String: sub-string starting from underscore(_)
    """
    for cdr in ['CDR_mild', 'CDR_moderate', 'CDR_none', 'CDR_very_mild']:
        if row[cdr] == 1:
            parts = cdr.split('_')
            return '_'.join(parts[1:])
    return None


# Use function to create CDR column
df['CDR'] = df.apply(get_cdr, axis=1)

# Check column values match before dropping One-Hot encoded columns
print("Check column values match before dropping One-Hot encoded columns:")
print((df['CDR'] == 'none').sum() == (df['CDR_none'].sum()))
print((df['CDR'] == 'very_mild').sum() == (df['CDR_very_mild'].sum()))
print((df['CDR'] == 'mild').sum() == (df['CDR_mild'].sum()))
print((df['CDR'] == 'moderate').sum() == (df['CDR_moderate'].sum()))

# In[35]:


# Drop one-hot encoded CDR columns
df.drop(['CDR_mild', 'CDR_moderate', 'CDR_none', 'CDR_very_mild'], axis=1, inplace=True)

df.to_csv('Data/df_after_pre-processing.csv', index=False)
