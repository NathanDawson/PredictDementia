# In[72]:


from data_setup import *
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Initialise Model and Perform Feature Selection
# In[105]:

# Use the pre-processed dataframe
df = pd.read_csv('data/df_after_pre-processing.csv')

# Initialise RandomForestClassifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Feature Selection with RFECV (n_jobs=8, assumes 8 processors available)
rf_rfecv = RFECV(estimator=rf, step=1, cv=StratifiedGroupKFold(n_splits=5), scoring='f1_macro', n_jobs=8)
rf_rfecv.fit(X_train, y_train, groups=groups_train)

# Identify the selected features
selected_features = X_train.columns[rf_rfecv.support_]

# Setup GridSearchCV With Pipeline And Hyperparameter Grid
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

# Setup Results Lists
# In[107]:


# Initialise lists to store results
rf_accuracies, rf_precisions, rf_recalls, rf_f1_scores = [], [], [], []
rf_best_params_list = []

# Configure Inner Loop
# In[108]:


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

# Visualise F1 Score For Each Fold
# In[109]:


plt.figure(figsize=(10, 8))
fold_numbers = range(1, len(rf_f1_scores) + 1)
plt.plot(fold_numbers, rf_f1_scores, marker='o')
plt.xlabel('Fold Number')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each RF Fold')
plt.xticks(fold_numbers)
plt.grid(True)

# Save figure
plt.savefig('results/Model1/RF_F1score_Across_Folds.png', bbox_inches='tight')

# Retrieve Best Hyperparameters From Best F1 Scoring Fold
# In[110]:


# Find the index of the best performing fold
best_fold_index = rf_f1_scores.index(max(rf_f1_scores))

# Retrieve the hyperparameters used in the best performing fold
best_hyperparameters = rf_best_params_list[best_fold_index]

# Removing the 'classifier__' prefix
best_hyperparameters = {j.replace('classifier__', ''): i for j, i in best_hyperparameters.items()}

# Train Final Model Using Best Hyperparameters Found
# In[111]:


# Train the final model using the best hyperparameters
final_rf = RandomForestClassifier(**best_hyperparameters, random_state=42, class_weight='balanced')
final_rf.fit(X_train[selected_features], y_train)

# Predict on test set using final model
y_pred = final_rf.predict(X_test[selected_features])

# Perform Dimensionality Reduction and Visualise Clustering
# In[112]:


# Perform t-SNE for dimensionality reduction to 2 components for visualisation
tsne = TSNE(n_components=2, random_state=42)
X_test_tsne = tsne.fit_transform(X_test[selected_features])

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

# Save figure
plt.savefig('results/Model1/RF_Predicted_CRR_Classes.png', bbox_inches='tight')

# Calculate Performance Metrics Using Each Fold and Final Model
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

# Visualise Performance Comparison Between Average Across Folds and Final Model
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

# Save figure
plt.savefig('results/Model1/compare_avg_folds_finalModel.png', bbox_inches='tight')
