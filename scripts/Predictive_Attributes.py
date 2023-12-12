# Access Feature Importances
# In[132]:

from data_setup import *
from Model1 import final_rf, selected_features
from Model2 import final_svm, svm_selected_features

import shap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Use the pre-processed dataframe
df = pd.read_csv('data/df_after_pre-processing.csv')


# Obtain feature importances for each model
rf_feature_importances = final_rf.feature_importances_
svm_feature_importances = final_svm.coef_[0]

# Visualise Feature Importances For Each Model
# In[133]:


# Obtain feature names
features = X_train.columns

# Visualise RandomForest feature importances
plt.figure(figsize=(10, 8))
plt.bar(features, rf_feature_importances)
plt.xticks(rotation=45)
plt.title("RandomForest Feature Importances")

# Save figure
plt.savefig('results/Predictive_Attributes/RF_Feature_Importance.png', bbox_inches='tight')

# Visualise SVM feature importances
plt.figure(figsize=(10, 8))
plt.bar(features, svm_feature_importances)
plt.xticks(rotation=45)
plt.title("SVM Feature Importances")

# Save figure
plt.savefig('results/Predictive_Attributes/SVM_Feature_Importance.png', bbox_inches='tight')


# Analyse Random Forest Predictions using SHAP
# In[134]:


# Create SHAP explainer
rf_explainer = shap.TreeExplainer(final_rf)
rf_shap_values = rf_explainer.shap_values(X_train)

# Matplotlib Config to enable saving of SHAP Summary Plot
plt.ioff()
matplotlib.use('Agg')

shap.summary_plot(rf_shap_values, X_train, plot_type="bar")

plt.savefig('results/Predictive_Attributes/RF_shap_summary_plot.png', bbox_inches='tight')

plt.close()
