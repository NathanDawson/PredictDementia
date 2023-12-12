# In[51]:
from data_setup import *
from sklearn.svm import SVC

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Initialise Model and Perform Feature Selection
# In[115]:


# Use the pre-processed dataframe
df = pd.read_csv('data/df_after_pre-processing.csv')


# Initialise Support Vector Machine
svm = SVC(kernel='linear', random_state=42, class_weight='balanced')

# Feature Selection with RFECV (n_jobs=8, assumes 8 processors available)
svm_rfecv = RFECV(estimator=svm, step=1, cv=StratifiedGroupKFold(n_splits=5), scoring='f1_macro', n_jobs=8)
svm_rfecv.fit(X_train, y_train, groups=groups_train)

# Identify selected features
svm_selected_features = X_train.columns[svm_rfecv.support_]

# Setup GridSearchCV With Pipeline And Hyperparameter Grid
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

# Setup Results Lists
# In[117]:


# Initialise lists to store results
svm_accuracies, svm_precisions, svm_recalls, svm_f1_scores = [], [], [], []
svm_best_params_list = []

# Configure Inner Loop
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

# Visualise F1 Score For Each Fold
# In[119]:


plt.figure(figsize=(10, 8))
svm_fold_numbers = range(1, len(svm_f1_scores) + 1)
plt.plot(svm_fold_numbers, svm_f1_scores, marker='o')
plt.xlabel('Fold Number')
plt.ylabel('F1 Score')
plt.title('F1 Score for Each SVM Fold')
plt.xticks(svm_fold_numbers)
plt.grid(True)

# Save figure
plt.savefig('results/Model2/SVM_F1score_Across_Folds.png', bbox_inches='tight')


#  Retrieve Best Hyperparameters From Best F1 Scoring Fold
# In[120]:


# Find index of best performing fold
svm_best_fold_index = svm_f1_scores.index(max(svm_f1_scores))

# Retrieve hyperparameters used in best performing fold
svm_best_hyperparameters = svm_best_params_list[svm_best_fold_index]

# Removing the 'classifier__' prefix
svm_best_hyperparameters = {j.replace('classifier__', ''): i for j, i in svm_best_hyperparameters.items()}

# Train Final Model Using Best Hyperparameters Found
# In[121]:


# Train final model using best hyperparameters
final_svm = SVC(**svm_best_hyperparameters, random_state=42, class_weight='balanced')
final_svm.fit(X_train[svm_selected_features], y_train)

# Evaluate final model on test set
svm_y_pred = final_svm.predict(X_test[svm_selected_features])

# Perform Dimensionality Reduction and Visualise Clustering
# In[122]:


# Perform t-SNE for dimensionality reduction to 2 components for visualisation
tsne = TSNE(n_components=2, random_state=42)
svm_X_test_tsne = tsne.fit_transform(X_test[svm_selected_features])

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

# Save figure
plt.savefig('results/Model2/SVM_Predicted_CRR_Classes.png', bbox_inches='tight')


# Calculate Performance Metrics Using Each Fold and Final Model
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

# Visualise Performance Comparison Between Average Across Folds and Final Model
# In[140]:


# Assign label locations
x = np.arange(len(metrics))
# Assign width of bars
width = 0.35

# Provide data
fig, ax = plt.subplots()
svm_rects1 = ax.bar(x - width / 2, svm_avg_values, width, label='Avg Across Folds')
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

# Save figure
plt.savefig('results/Model2/compare_avg_folds_finalSVMModel.png', bbox_inches='tight')
