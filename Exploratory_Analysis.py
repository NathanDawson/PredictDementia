# In[1]:


import pandas as pd
import altair as alt

# In[2]:


data1 = pd.read_csv("Data/visit-1.csv")
data2 = pd.read_csv("Data/visit-2.csv")
data3 = pd.read_csv("Data/visit-3.csv")
data4 = pd.read_csv("Data/visit-4.csv")
data5 = pd.read_csv("Data/visit-5.csv")

# In[3]:


data1.head(7), data2.head(7), data3.head(7), data4.head(7), data5.head(7)

# In[4]:


# Join datasets together
df = pd.concat([data1, data2, data3, data4, data5], ignore_index=True)

# In[5]:


print("Dataframe Head:", df.head(10))

# In[6]:


print("Dataframe Shape:", df.shape)

# In[7]:


print("Dataframe Info:", df.info())

# In[8]:


print("Dataframe Describe:", df.describe())


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
(alt.vconcat(*charts)).save('Graphs/Exploratory_Analysis/charts.html')
