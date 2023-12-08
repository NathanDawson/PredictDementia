# Access Feature Importances
# In[132]:

from data_setup import *
from Model1 import final_rf
from Model2 import final_svm

import shap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Use the dataframe after model_comparison.py has ran
df = pd.read_csv('Data/df_after_model_comparison.csv')


# Obtain feature importances for each model
rf_feature_importances = final_rf.feature_importances_
svm_feature_importances = final_svm.coef_[0]

# Visualise Feature Importances For Each Model
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

# Save figure
plt.savefig('Graphs/Predictive_Attributes/Models_Feature_Importance.png', bbox_inches='tight')


# Analyse Random Forest Predictions using SHAP
# In[134]:


# Create SHAP explainer
rf_explainer = shap.TreeExplainer(final_rf)
rf_shap_values = rf_explainer.shap_values(X_train)

# Summarise effects of all features
shap.summary_plot(rf_shap_values, X_train, plot_type="bar")


df.to_csv('Data/df_after_Predictive_Attributes.csv', index=False)
