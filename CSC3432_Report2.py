#!/usr/bin/env python
# coding: utf-8

# # Exploring Dementia and Alzheimer's Disease Using Patient Records From Longitudinal MRI Data

# ### Combine The Data

# In[1]:


import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


data1 = pd.read_csv("visit-1.csv")
data2 = pd.read_csv("visit-2.csv")
data3 = pd.read_csv("visit-3.csv")
data4 = pd.read_csv("visit-4.csv")
data5 = pd.read_csv("visit-5.csv")

# In[3]:


data1.head(7), data2.head(7), data3.head(7), data4.head(7), data5.head(7)

# In[4]:


# Join datasets together
df = pd.concat([data1, data2, data3, data4, data5], ignore_index=True)

# In[5]:


df.head(10)

# ### Initial Exploratory Data Analysis

# In[6]:


df.shape

# In[7]:


df.info()

# In[8]:


df.describe()


# In[9]:


# Graph each columns data distribution
def plot_histogram(column):
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(column, bin=alt.Bin(maxbins=50), title=column),
        y='count()',
    ).properties(
        title=f'Distribution of {column}',
        width=400,
        height=200
    )
    return chart


charts = [plot_histogram(col) for col in df.columns]
alt.vconcat(*charts)

# ### Data Pre-Processing

# In[10]:


# Check for object type features
objects = df.select_dtypes(include=[object])
objects.head()

# #### Investigate ASF Column

# In[11]:


# Convert ASF to numeric
converted_column = pd.to_numeric(df['ASF'], errors='coerce')

# Check for rows which could not be converted
df['ASF'][converted_column.isna() & df['ASF'].notna()]

# In[12]:


# Look up row which could not be converted
df.loc[128]

# In[13]:


# Fix row to allow for conversion to numeric
df.at[128, 'ASF'] = df.at[128, 'ASF'].replace(',', '.')

# In[14]:


# Look up row to ensure it is correctly formatted
df.loc[128]

# In[15]:


# Convert column to numeric
df['ASF'] = pd.to_numeric(df['ASF'], errors='coerce')

# #### OneHotEncode Columns: Sex, Hand, CDR

# In[16]:


from sklearn.preprocessing import OneHotEncoder

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
enc.get_feature_names_out()

# ##### Fix CDR very mild and CDR mild feature names

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

# #### Check for Duplicated Data

# In[23]:


df.duplicated().sum()

# #### Analyse Missing Data

# In[24]:


# Total number of missing data per column
df.isnull().sum()

# In[25]:


# Percentage of missing data per column
df.isnull().sum() * 100 / len(df)

# In[26]:


columns_to_impute = ['SES', 'MMSE', 'eTIV', 'ASF']
imputation_data = df[columns_to_impute]

# #### Impute using MICE

# In[27]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# In[28]:


mice_imputer = IterativeImputer(max_iter=10, random_state=0)

imputed_data_mice = mice_imputer.fit_transform(imputation_data)

MICE_Imputed = pd.DataFrame(imputed_data_mice, columns=columns_to_impute)

MICE_Imputed.isnull().sum()

# In[29]:


# Investigate imputed data
MICE_Imputed.head()

# ##### Investigate SES Feature

# In[30]:


# Check columns unique values
MICE_Imputed.SES.unique()

# In[31]:


# Round imputed values to be in range 1 - 5 as intended
limit_values = np.clip(MICE_Imputed.SES, 1, 5)

round_SES = np.rint(limit_values)

round_SES.unique()

# In[32]:


# Update column with rounded values
MICE_Imputed.SES = round_SES

# In[33]:


# Update dataframe with imputed data
df_without_missing = df.drop(columns=columns_to_impute)

df = pd.concat([df_without_missing, MICE_Imputed], axis=1)

df.isnull().sum()


# #### Reverse One-Hot Encoding of Target Variable (CDR)

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
print((df['CDR'] == 'none').sum() == (df['CDR_none'].sum()))
print((df['CDR'] == 'very_mild').sum() == (df['CDR_very_mild'].sum()))
print((df['CDR'] == 'mild').sum() == (df['CDR_mild'].sum()))
print((df['CDR'] == 'moderate').sum() == (df['CDR_moderate'].sum()))

# In[35]:


# Drop one-hot encoded CDR columns
df.drop(['CDR_mild', 'CDR_moderate', 'CDR_none', 'CDR_very_mild'], axis=1, inplace=True)

# # Task 1: Attempt to Distinguish Healthy Patients Records From Cases of Dementia

# ##### Create X and y Variables

# In[36]:


# Create X and y variables
X = df.drop(['CDR', 'MRI_ID'], axis=1)
y = df['CDR']

# ##### Transform y variable to numeric

# In[69]:


from sklearn.preprocessing import OrdinalEncoder

# Transform categorical CDR feature to numeric format, retaining order
encoder = OrdinalEncoder(categories=[['none', 'very_mild', 'mild', 'moderate']])
y = encoder.fit_transform(y.values.reshape(-1, 1)).ravel()

# Assign CDR score of Moderate to Very Mild
y[y == 3] = 2

# Check Moderate class (3) has been removed
print((y == 3).sum() == 0)

# ##### Split Data

# In[70]:


from sklearn.model_selection import train_test_split

# 'X.ID' contains the patient IDs for grouping
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, X.ID, test_size=0.2, random_state=42, stratify=y)

# Reset indices to align them for grouping purposes
X_train = X_train.reset_index(drop=True)
y_train = pd.Series(y_train).reset_index(drop=True)
groups_train = groups_train.reset_index(drop=True)

# ##### Create Nested Cross Validations

# In[71]:


from sklearn.model_selection import StratifiedGroupKFold

# Outer CV for model evaluation
outer_cv = StratifiedGroupKFold(n_splits=10)

# Inner CV for model selection (feature selection and hyperparameter tuning)
inner_cv = StratifiedGroupKFold(n_splits=10)

# ## Model 1: Random Forest Classifier

# In[72]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ##### Initialise Model and Perform Feature Selection

# In[105]:


# Initialise RandomForestClassifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Feature Selection with RFECV (n_jobs=8, assumes 8 processors available)
rf_rfecv = RFECV(estimator=rf, step=1, cv=StratifiedGroupKFold(n_splits=5), scoring='f1_macro', n_jobs=8)
rf_rfecv.fit(X_train, y_train, groups=groups_train)

# Identify the selected features
selected_features = X_train.columns[rf_rfecv.support_]

# ##### Setup GridSearchCV With Pipeline And Hyperparameter Grid

# In[106]:


# Define hyperparameter grid
rf_params = {
    'classifier__max_depth': [6, 8, 10],
    'classifier__n_estimators': [150, 175],
    'classifier__min_samples_leaf': [2, 4],
    'classifier__min_samples_split': [2, 4]
}

# Setup pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', rf)
])

# Setup GridSearchCV with pipeline and hyperparameter grid
# n_jobs=8, assumes 8 processors available
rf_search = GridSearchCV(rf_pipeline, rf_params, cv=inner_cv, n_jobs=8, scoring='f1_macro')

# ##### Setup Results Lists

# In[107]:


# Initialise lists to store results
rf_accuracies, rf_precisions, rf_recalls, rf_f1_scores = [], [], [], []
rf_best_params_list = []

# ##### Configure Inner Loop

# In[108]:


# Estimated 5 Minute Run Time

X = X.reset_index(drop=True)

# Outer CV loop
for train_idx, valid_idx in outer_cv.split(X_train[selected_features], y_train, groups=groups_train):
    rf_X_train, rf_X_valid = X_train[selected_features].iloc[train_idx], X_train[selected_features].iloc[valid_idx]
    rf_y_train, rf_y_valid = y_train[train_idx], y_train[valid_idx]
    groups_train_fold = groups_train.iloc[train_idx]

    # Inner CV: Hyperparameter tuning with GridSearchCV
    rf_search.fit(rf_X_train, rf_y_train, groups=groups_train_fold)

    # Retrieve best hyperparameters
    rf_best_params_list.append(rf_search.best_params_)

    # Evaluate the model using best estimator on the validation set
    best_rf = rf_search.best_estimator_
    rf_predictions = best_rf.predict(rf_X_valid)

    # Calculate performance metrics using best estimator on the validation set
    rf_accuracies.append(accuracy_score(rf_y_valid, rf_predictions))
    rf_precisions.append(precision_score(rf_y_valid, rf_predictions, average='macro'))
    rf_recalls.append(recall_score(rf_y_valid, rf_predictions, average='macro'))
    rf_f1_scores.append(f1_score(rf_y_valid, rf_predictions, average='macro'))

# ##### Visualise F1 Score For Each Fold

# In[109]:


fold_numbers = range(1, len(rf_f1_scores) + 1)
plt.plot(fold_numbers, rf_f1_scores, marker='o')
plt.xlabel('Fold Number')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each RF Fold')
plt.xticks(fold_numbers)
plt.grid(True)
plt.show()

# ##### Retrieve Best Hyperparameters From Best F1 Scoring Fold

# In[110]:


# Find the index of the best performing fold
best_fold_index = rf_f1_scores.index(max(rf_f1_scores))

# Retrieve the hyperparameters used in the best performing fold
best_hyperparameters = rf_best_params_list[best_fold_index]

# Removing the 'classifier__' prefix
best_hyperparameters = {k.replace('classifier__', ''): v for k, v in best_hyperparameters.items()}

# ##### Train Final Model Using Best Hyperparameters Found

# In[111]:


# Train the final model using the best hyperparameters
final_rf = RandomForestClassifier(**best_hyperparameters, random_state=42, class_weight='balanced')
final_rf.fit(X_train, y_train)

# Predict on test set using final model
y_pred = final_rf.predict(X_test)

# ##### Perform Dimensionality Reduction and Visualise Clustering

# In[112]:


from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE

# Perform t-SNE for dimensionality reduction to 2 components for visualisation
tsne = TSNE(n_components=2, random_state=42)
X_test_tsne = tsne.fit_transform(X_test)

# Define color map for classes
colors = ['blue', 'green', 'red']
class_names = ['None', 'Very Mild', 'Mild']

# Scatter plot using two t-SNE components
plt.figure(figsize=(10, 8))

# Plot each class separately to assign labels and create a legend
for i, color in zip(range(3), colors):
    plt.scatter(X_test_tsne[y_pred == i, 0], X_test_tsne[y_pred == i, 1], c=color, label=class_names[i], alpha=0.7,
                edgecolors='b')

plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Visualisation of RF Predicted CDR Classes')
plt.legend(title='Predicted CDR Classes')
plt.show()

# ##### Calculate Performance Metrics Using Each Fold and Final Model

# In[113]:


metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Calculate the average of each metric across all folds
rf_avg_accuracy = sum(rf_accuracies) / len(rf_accuracies)
rf_avg_precision = sum(rf_precisions) / len(rf_precisions)
rf_avg_recall = sum(rf_recalls) / len(rf_recalls)
rf_avg_f1 = sum(rf_f1_scores) / len(rf_f1_scores)

# Calculate performance metrics using final model
final_accuracy = accuracy_score(y_test, y_pred)
final_precision = precision_score(y_test, y_pred, average='macro')
final_recall = recall_score(y_test, y_pred, average='macro')
final_f1 = f1_score(y_test, y_pred, average='macro')

# Join results for visualisation purposes
avg_values = [rf_avg_accuracy, rf_avg_precision, rf_avg_recall, rf_avg_f1]
final_values = [final_accuracy, final_precision, final_recall, final_f1]

# ##### Visualise Performance Comparison Between Average Across Folds and Final Model

# In[138]:


# Assign label locations
x = np.arange(len(metrics))
# Assign width of bars
width = 0.35

# Provide data
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, avg_values, width, label='Avg Across Folds')
rects2 = ax.bar(x + width / 2, final_values, width, label='Model')

# Add text to visualisation
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison: Average Across Folds vs Final RF Model')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right')


# Add labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()

# ## Model 2: Support Vector Machine

# In[51]:


from sklearn.svm import SVC

# ##### Initialise Model and Perform Feature Selection

# In[115]:


# Estimated 5 Minute Run Time

# Initialise Support Vector Machine
svm = SVC(kernel='linear', random_state=42, class_weight='balanced')

# Feature Selection with RFECV (n_jobs=8, assumes 8 processors available)
svm_rfecv = RFECV(estimator=svm, step=1, cv=StratifiedGroupKFold(n_splits=5), scoring='f1_macro', n_jobs=8)
svm_rfecv.fit(X_train, y_train, groups=groups_train)

# Identify selected features
svm_selected_features = X_train.columns[svm_rfecv.support_]

# ##### Setup GridSearchCV With Pipeline And Hyperparameter Grid

# In[116]:


# Define hyperparameter Grid
svm_params = {'classifier__C': [1, 10, 100, ],
              'classifier__gamma': [1, 0.1, 0.01],
              'classifier__kernel': ['linear']}

# Setup pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', svm)
])

# Setup GridSearchCV with the pipeline and hyperparameter grid
# n_jobs=8, assumes 8 processors available
svm_search = GridSearchCV(svm_pipeline, svm_params, cv=inner_cv, n_jobs=8, scoring='f1_macro')

# ##### Setup Results Lists

# In[117]:


# Initialise lists to store results
svm_accuracies, svm_precisions, svm_recalls, svm_f1_scores = [], [], [], []
svm_best_params_list = []

# ##### Configure Inner Loop

# In[118]:


X = X.reset_index(drop=True)

# Outer CV loop
for train_idx, valid_idx in outer_cv.split(X_train[svm_selected_features], y_train, groups=groups_train):
    svm_X_train, svm_X_valid = X_train[svm_selected_features].iloc[train_idx], X_train[svm_selected_features].iloc[
        valid_idx]
    svm_y_train, svm_y_valid = y_train[train_idx], y_train[valid_idx]
    svm_groups_train_fold = groups_train.iloc[train_idx]

    # Inner CV: Hyperparameter tuning with GridSearchCV
    svm_search.fit(svm_X_train, svm_y_train, groups=svm_groups_train_fold)

    # Retrieve the best hyperparameters
    svm_best_params_list.append(svm_search.best_params_)

    # Evaluate the model using the best estimator on the validation set
    best_svm = svm_search.best_estimator_
    svm_predictions = best_svm.predict(svm_X_valid)

    # Calculate performance metrics using best estimator on the validation set
    svm_accuracies.append(accuracy_score(svm_y_valid, svm_predictions))
    svm_precisions.append(precision_score(svm_y_valid, svm_predictions, average='macro'))
    svm_recalls.append(recall_score(svm_y_valid, svm_predictions, average='macro'))
    svm_f1_scores.append(f1_score(svm_y_valid, svm_predictions, average='macro'))

# ##### Visualise F1 Score For Each Fold

# In[119]:


svm_fold_numbers = range(1, len(svm_f1_scores) + 1)
plt.plot(svm_fold_numbers, svm_f1_scores, marker='o')
plt.xlabel('Fold Number')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each SVM Fold')
plt.xticks(svm_fold_numbers)
plt.grid(True)
plt.show()

# ##### Retrieve Best Hyperparameters From Best F1 Scoring Fold

# In[120]:


# Find index of best performing fold
svm_best_fold_index = svm_f1_scores.index(max(svm_f1_scores))

# Retrieve hyperparameters used in best performing fold
svm_best_hyperparameters = svm_best_params_list[svm_best_fold_index]

# Removing the 'classifier__' prefix
svm_best_hyperparameters = {k.replace('classifier__', ''): v for k, v in svm_best_hyperparameters.items()}

# ##### Train Final Model Using Best Hyperparameters Found

# In[121]:


# Train final model using best hyperparameters
final_svm = SVC(**svm_best_hyperparameters, random_state=42, class_weight='balanced')
final_svm.fit(X_train, y_train)

# Evaluate final model on test set
svm_y_pred = final_svm.predict(X_test)

# ##### Perform Dimensionality Reduction and Visualise Clustering

# In[122]:


from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE

# Perform t-SNE for dimensionality reduction to 2 components for visualisation
tsne = TSNE(n_components=2, random_state=42)
svm_X_test_tsne = tsne.fit_transform(X_test)

# Define color map for classes
colors = ['blue', 'green', 'red']
class_names = ['None', 'Very Mild', 'Mild']

# Scatter plot using two t-SNE components
plt.figure(figsize=(10, 8))

# Plot each class separately to assign labels and create a legend
for i, color in zip(range(3), colors):
    plt.scatter(svm_X_test_tsne[svm_y_pred == i, 0], svm_X_test_tsne[svm_y_pred == i, 1], c=color, label=class_names[i],
                alpha=0.7, edgecolors='b')

plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Visualisation of SVM Predicted CDR Classes')
plt.legend(title='Predicted CDR Classes')
plt.show()

# ##### Calculate Performance Metrics Using Each Fold and Final Model

# In[123]:


metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Calculate average of each metric across all folds
svm_avg_accuracy = sum(svm_accuracies) / len(svm_accuracies)
svm_avg_precision = sum(svm_precisions) / len(svm_precisions)
svm_avg_recall = sum(svm_recalls) / len(svm_recalls)
svm_avg_f1 = sum(svm_f1_scores) / len(svm_f1_scores)

# Calculate performance metrics using final model
svm_final_accuracy = accuracy_score(y_test, svm_y_pred)
svm_final_precision = precision_score(y_test, svm_y_pred, average='macro')
svm_final_recall = recall_score(y_test, svm_y_pred, average='macro')
svm_final_f1 = f1_score(y_test, svm_y_pred, average='macro')

# Join results for visualisation purposes
svm_avg_values = [svm_avg_accuracy, svm_avg_precision, svm_avg_recall, svm_avg_f1]
svm_final_values = [svm_final_accuracy, svm_final_precision, svm_final_recall, svm_final_f1]

# ##### Visualise Performance Comparison Between Average Across Folds and Final Model

# In[140]:


# Assign label locations
x = np.arange(len(metrics))
# Assign width of bars
width = 0.35

# Provide Data
fig, ax = plt.subplots()
svm_rects1 = ax.bar(x - width / 2, svm_avg_values, width, label='Average Across Folds')
svm_rects2 = ax.bar(x + width / 2, svm_final_values, width, label='Model')

# Add text to visualisation
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison: Average Across Folds vs Final SVM Model')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc="upper right", framealpha=0.5)


# Add labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(svm_rects1)
autolabel(svm_rects2)

fig.tight_layout()
plt.show()

# ## Comparative Analysis Between Models

# ##### Visualise Both Models Average Across Folds Metrics

# In[128]:


# Assign label locations
x = np.arange(len(metrics))
# Assign width of bars
width = 0.35

# Provide Data
fig, ax = plt.subplots()
rf_avg_rects = ax.bar(x + width / 2, avg_values, width, label='RF')
svm_avg_rects = ax.bar(x - width / 2, svm_avg_values, width, label='SVM')

# Add text to visualisation
ax.set_ylabel('Scores')
ax.set_title('Average Across Folds: RF vs SVM')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc="upper right", framealpha=0.5)


# Add labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(svm_avg_rects)
autolabel(rf_avg_rects)

fig.tight_layout()
plt.show()

# ##### Visualise Both Final Models Metrics

# In[129]:


# Assign label locations
x = np.arange(len(metrics))
# Assign width of bars
width = 0.35

# Provide Data
fig, ax = plt.subplots()
rf_final_rects = ax.bar(x + width / 2, final_values, width, label='RF')
svm_final_rects = ax.bar(x - width / 2, svm_final_values, width, label='SVM')

# Add text to visualisation
ax.set_ylabel('Scores')
ax.set_title('Final Models: RF vs SVM')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc="upper right", framealpha=0.5)


# Add labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(svm_final_rects)
autolabel(rf_final_rects)

fig.tight_layout()
plt.show()

# ##### Box-Plot And Swarm-Plot of Performance Metrics Across All Folds

# In[130]:


# Join fold metrics together
metrics_df = pd.DataFrame({
    'Accuracy': svm_accuracies + rf_accuracies,
    'Precision': svm_precisions + rf_precisions,
    'Recall': svm_recalls + rf_recalls,
    'F1 Score': svm_f1_scores + rf_f1_scores,
    'Model': ['SVM'] * 10 + ['RandomForest'] * 10  # x10 due to use of 10 splits each
})

# Transform data to long format for visualisation
long_df = metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')

# In[131]:


plt.figure(figsize=(10, 6))

# Create boxplot
sns.boxplot(x='Score', y='Metric', hue='Model', data=long_df, showfliers=False)

# Add jitter with swarmplot
sns.swarmplot(x='Score', y='Metric', hue='Model', data=long_df, alpha=0.8)

# Add Text
plt.title('Distribution of Performance Metric Scores Across All Folds')
plt.xlabel('Scores')
plt.ylabel('Performance Metric')
plt.legend(title='Model')

plt.show()

# # Task 2: Examine Which Attributes Are The Most Predictive

# ##### Access Feature Importances

# In[132]:


# Obtain feature importances for each model
rf_feature_importances = final_rf.feature_importances_
svm_feature_importances = final_svm.coef_[0]

# ##### Visualise Feature Importances For Each Model

# In[133]:


# Obtain feature names
features = X_train.columns

# Visualise RandomForest feature importances
plt.bar(features, rf_feature_importances)
plt.xticks(rotation=45)
plt.title("RandomForest Feature Importances")
plt.show()

# Visualise SVM feature importances
plt.bar(features, svm_feature_importances)
plt.xticks(rotation=45)
plt.title("SVM Feature Importances")
plt.show()

# ##### Analyse Random Forest Predictions using SHAP

# In[134]:


# !pip install shap
import shap

# Create SHAP explainer
rf_explainer = shap.TreeExplainer(final_rf)
rf_shap_values = rf_explainer.shap_values(X_train)

# Summarise effects of all features
shap.summary_plot(rf_shap_values, X_train, plot_type="bar")
