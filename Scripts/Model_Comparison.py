

# Visualise Both Models Average Across Folds Metrics
# In[128]:

from data_setup import *
from Model1 import metrics, avg_values, final_values, rf_accuracies, rf_precisions, rf_recalls, rf_f1_scores
from Model2 import svm_avg_values, svm_final_values, svm_accuracies, svm_precisions, svm_recalls, svm_f1_scores

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


# Use the dataframe after Model2.py has ran
df = pd.read_csv('Data/df_after_Model2.csv')


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

# Save figure
plt.savefig('Graphs/Model_Comparison/Average_Across_Folds.png', bbox_inches='tight')

# Visualise Both Final Models Metrics
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

# Save figure
plt.savefig('Graphs/Model_Comparison/Final_Models.png', bbox_inches='tight')

# Box-Plot And Swarm-Plot of Performance Metrics Across All Folds
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

# Save figure
plt.savefig('Graphs/Model_Comparison/Distribution_Across_Folds.png', bbox_inches='tight')

df.to_csv('Data/df_after_model_comparison.csv', index=False)
