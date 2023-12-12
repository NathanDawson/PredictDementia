# Access Feature Importances
# In[132]:

from data_setup import *
from Model1 import final_rf, selected_features
from Model2 import final_svm, svm_selected_features

import shap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Obtain feature importances for each model
rf_feature_importances = final_rf.feature_importances_
svm_feature_importances = final_svm.coef_[0]

# Visualise Feature Importances For Each Model
# In[133]:


# Obtain Random Forest feature names
rf_features = X_train[selected_features].columns

# Visualise RandomForest feature importances
plt.figure(figsize=(10, 8))
plt.bar(rf_features, rf_feature_importances)
plt.xticks(rotation=45)
plt.title("RandomForest Feature Importances")

# Save figure
plt.savefig('results/Predictive_Attributes/RF_Feature_Importance.png', bbox_inches='tight')

# Obtain Support Vector Machine feature names
svm_features = X_train[svm_selected_features].columns

# Visualise SVM feature importances
plt.figure(figsize=(10, 8))
plt.bar(svm_features, svm_feature_importances)
plt.xticks(rotation=45)
plt.title("SVM Feature Importances")

# Save figure
plt.savefig('results/Predictive_Attributes/SVM_Feature_Importance.png', bbox_inches='tight')


# Analyse Random Forest Predictions using SHAP
# In[134]:


# Create SHAP explainer
rf_explainer = shap.TreeExplainer(final_rf)
rf_shap_values = rf_explainer.shap_values(X_train[selected_features])

# Matplotlib Config to enable saving of SHAP Summary Plot
plt.ioff()
matplotlib.use('Agg')

shap.summary_plot(rf_shap_values, X_train[selected_features], plot_type="bar")

plt.savefig('results/Predictive_Attributes/RF_shap_summary_plot.png', bbox_inches='tight')

plt.close()
