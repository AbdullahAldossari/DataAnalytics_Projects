#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# 
# 

# In[ ]:


### lets import all the necessary packages !


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


comments = pd.read_csv('fetch data from excel file "UScomments" in your machine ' , error_bad_lines=False)


# In[ ]:


### above is a warning , u can ignore that ..


# In[3]:


comments.head()


# In[4]:


## lets find out missing values in your data
comments.isnull().sum()


# In[5]:


## drop missing values as we have very few & lets update dataframe as well..
comments.dropna(inplace=True)


# In[6]:


comments.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 2.. Perform Sentiment Analysis
#     
#     In short , Sentiment analysis is all about analyszing sentiments of Users

# In[ ]:





# In[ ]:


get_ipython().system('pip install textblob')
### lets perform sentiment analysis using TextBlob which is a NLP library built on top of NLTK )..


# In[ ]:


## if you are getting error in textblob while installing using pip ..
## you can install textblob using conda in Anazonda prompt 

## conda install -c conda-forge textblob  


# In[8]:


from textblob import TextBlob


# In[9]:


comments.head(6)


# In[13]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity

### its a neutral sentence !


# In[14]:


comments.shape


# In[15]:


## for those of you who dont have good specifications , considering sample of data is a good option !

sample_df = comments[0:1000]


# In[16]:


sample_df.shape


# In[ ]:





# In[ ]:





# In[17]:


polarity = []

for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[18]:


len(polarity)


# In[19]:


comments['polarity']  = polarity

### Inserting polarity values into comments dataframe while defining feature name as "polarity"


# In[20]:


comments.head(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 3..  Wordcloud Analysis of your data

# In[ ]:


### Lets perform EDA for the highly Positve sentences ie Polarity value will be 1 


# In[21]:


filter1 = comments['polarity']==1


# In[25]:


comments_positive = comments[filter1]


# In[ ]:





# In[23]:


filter2 = comments['polarity']==-1


# In[26]:


comments_negative = comments[filter2]


# In[ ]:





# In[27]:


comments_positive.head(5)


# In[ ]:





# In[ ]:


get_ipython().system('pip install wordcloud')


# In[28]:


from wordcloud import WordCloud , STOPWORDS


# In[29]:


set(STOPWORDS)


# In[30]:


comments['comment_text']


# In[32]:


type(comments['comment_text'])


# In[34]:


### for wordcloud , we need to frame our 'comment_text' feature into string ..
total_comments_positive = ' '.join(comments_positive['comment_text'])


# In[35]:


wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_positive)


# In[37]:


plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:


### Conclusion-->> positive Users are emphasizing more on best , awesome , perfect , amazing , look , happy  etc..


# In[ ]:





# In[ ]:





# In[38]:


total_comments_negative = ' '.join(comments_negative['comment_text'])


# In[39]:


wordcloud2 = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)


# In[40]:


plt.imshow(wordcloud2)
plt.axis('off')


# In[ ]:


### Conclusion-->> Negative Users are emphasizing more on Terrible , worst ,horrible ,boring , disgusting etc..


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 4.. Perform Emoji's Analysis

# In[ ]:


get_ipython().system('pip install emoji==2.2.0 ## 2.2.0 is a most stable version till date , hence installing this version makes sense !')


# In[41]:


import emoji


# In[42]:


emoji.__version__


# In[44]:


comments['comment_text'].head(6)


# In[ ]:





# In[ ]:


### lets extract emoji from below comment


# In[45]:


comment = 'trending 😉'


# In[46]:


[char for char in comment if char in emoji.EMOJI_DATA]


# In[ ]:


## lets try to write above code in a more simpler & readable way :


# In[47]:


emoji_list = []

for char in comment:
    if char in emoji.EMOJI_DATA:
        emoji_list.append(char)


# In[48]:


emoji_list


# In[ ]:





# In[ ]:





# In[49]:


all_emojis_list = []

for comment in comments['comment_text'].dropna(): ## in case u have missing values , call dropna()
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emojis_list.append(char)


# In[50]:


all_emojis_list[0:10]


# In[ ]:





# In[ ]:


### NOw we have to compute frequencies of each & every emoji in "all_emojis_list"..


# In[51]:


from collections import Counter


# In[52]:


Counter(all_emojis_list).most_common(10)


# In[53]:


Counter(all_emojis_list).most_common(10)[0]


# In[54]:


Counter(all_emojis_list).most_common(10)[0][0]


# In[56]:


Counter(all_emojis_list).most_common(10)[1][0]


# In[57]:


Counter(all_emojis_list).most_common(10)[2][0]


# In[59]:


emojis = [Counter(all_emojis_list).most_common(10)[i][0] for i in range(10)]


# In[ ]:





# In[ ]:





# In[ ]:





# In[55]:


Counter(all_emojis_list).most_common(10)[0][1]


# In[60]:


Counter(all_emojis_list).most_common(10)[1][1]


# In[61]:


Counter(all_emojis_list).most_common(10)[2][1]


# In[62]:


freqs = [Counter(all_emojis_list).most_common(10)[i][1] for i in range(10)]


# In[63]:


freqs


# In[ ]:





# In[64]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[65]:


trace = go.Bar(x=emojis , y=freqs)


# In[66]:


iplot([trace])


# In[ ]:


## Conclusions : Majority of the customers are happy as most of them are using emojis like: funny , love , heart , outstanding..


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 5.. Collect Entire data of Youtube !

# In[67]:


import os


# In[69]:


files= os.listdir(r'Z:\1.. Youtube\Datasets\additional_data')


# In[70]:


files


# In[ ]:





# In[ ]:





# In[72]:


## extracting csv files only from above list ..

files_csv = [file for file in files if '.csv' in file]


# In[73]:


files_csv


# In[ ]:





# In[75]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# #### different types of encoding-->>
#     Note : encoding may change depending upon data  , country data , sometimes regional data as well.
#     Fore more inforation on Encoding -- Follow below
# ### https://docs.python.org/3/library/codecs.html#standard-encodings¶

# In[77]:


full_df = pd.DataFrame()
path = r'Z:\1.. Youtube\Datasets\additional_data'


for file in files_csv:
    current_df = pd.read_csv(path+'/'+file , encoding='iso-8859-1' , error_bad_lines=False)
    
    full_df = pd.concat([full_df , current_df] , ignore_index=True)


# In[78]:


full_df.shape


# In[ ]:





# In[ ]:





# In[ ]:





# ## 6.. How to export your data into (csv, json, db)

# In[ ]:


### upto some extent your data is cleaned , so lets export this data into various formats for further analysis :


# In[ ]:


'''

export your data into :
    a) csv 
    b) json
    c) db


'''


# In[80]:


full_df[full_df.duplicated()].shape


# In[2]:


'''
default value of keep='first'

It means that the method will consider the first instance of a row to be unique and the remaining instances to be duplicates.


keep='first' (default): mark all rows as duplicates except for the first occurrence.

keep='last': mark all rows as duplicates except for the last occurrence.

keep=False: mark all rows as all duplicates.


'''


# In[ ]:





# In[81]:


full_df = full_df.drop_duplicates() ## lets drop duplicate rows ..


# In[82]:


full_df.shape


# In[ ]:





# #### a... Storing data into csv ..

# In[83]:


### you can consider sample of data depending on how efficient your system is..

full_df[0:1000].to_csv(r'Z:\1.. Youtube\export_data/youtube_sample.csv' , index=False)


# In[ ]:





# #### b... Storing data into json

# In[84]:


full_df[0:1000].to_json(r'Z:\1.. Youtube\export_data/youtube_sample.json')


# In[ ]:





# #### c... Storing data into database

# In[94]:


#create engine allows us to connect to database
from sqlalchemy import create_engine


# In[104]:


# Lets create sql_alchemy engine by using create_engine method ie create engine allows us to connect to database
engine = create_engine(r'sqlite:///Z:\1.. Youtube\export_data/youtube_sample.sqlite')


# In[105]:


### we will store first 1000 rows into Users table..
full_df[0:1000].to_sql('Users' , con=engine , if_exists='append')


# In[ ]:


## As soon as u have u have your data into 'youtube_sample.sqlite' which has table has 'Users', now u can read data from this db file 'youtube_whole_data.sqlite' using sqlite3 & pandas


# In[ ]:





# In[ ]:





# In[ ]:





# ## 7.. Which Category has the maximum likes ?

# In[106]:


full_df.head(5)


# In[107]:


full_df['category_id'].unique()


# In[ ]:





# In[108]:


## lets read json file ..
json_df = pd.read_json(r'Z:\1.. Youtube\Datasets\additional_data/US_category_id.json')


# In[109]:


json_df


# In[111]:


json_df['items'][0]

### each row of 'Items' feature is dictionary .. 


# In[112]:


json_df['items'][1]


# In[ ]:





# In[113]:


cat_dict = {}

for item in json_df['items'].values:
    ## cat_dict[key] = value (Syntax to insert key:value in dictionary)
    cat_dict[int(item['id'])] = item['snippet']['title']


# In[114]:


cat_dict


# In[116]:


full_df['category_name'] = full_df['category_id'].map(cat_dict)


# In[117]:


full_df.head(4)


# In[ ]:





# In[ ]:





# In[120]:


plt.figure(figsize=(12,8))
sns.boxplot(x='category_name' , y='likes' , data=full_df)
plt.xticks(rotation='vertical')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 8.. Find out whether audience is engaged or not

# In[123]:


full_df['like_rate'] = (full_df['likes']/full_df['views'])*100
full_df['dislike_rate'] = (full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate'] = (full_df['comment_count']/full_df['views'])*100


# In[124]:


full_df.columns


# In[ ]:





# In[126]:


plt.figure(figsize=(8,6))
sns.boxplot(x='category_name' , y='like_rate' , data=full_df)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


### analysing relationship between views & likes


# In[127]:


sns.regplot(x='views' , y='likes' , data = full_df)


# In[ ]:





# In[128]:


full_df.columns


# In[130]:


full_df[['views', 'likes', 'dislikes']].corr() ### finding co-relation values between ['views', 'likes', 'dislikes']


# In[132]:


sns.heatmap(full_df[['views', 'likes', 'dislikes']].corr() , annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 9.. Which channels have the largest number of trending videos?

# In[133]:


full_df.head(6)


# In[ ]:





# In[134]:


full_df['channel_title'].value_counts()


# In[ ]:





# In[ ]:


### lets obtain above frequency table using groupby approach : 


# In[139]:


cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()


# In[142]:


cdf = cdf.rename(columns={0:'total_videos'})


# In[143]:


cdf


# In[ ]:





# In[144]:


import plotly.express as px


# In[145]:


px.bar(data_frame=cdf[0:20] , x='channel_title' , y='total_videos')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 10.. Does Punctuations in title and tags have any relation with views, likes, dislikes comments?

# In[147]:


full_df['title'][0]


# In[148]:


import string


# In[149]:


string.punctuation


# In[151]:


len([char for char in full_df['title'][0] if char in string.punctuation])


# In[ ]:





# In[152]:


def punc_count(text):
    return len([char for char in text if char in string.punctuation])


# In[ ]:





# In[153]:


sample = full_df[0:10000]


# In[154]:


sample['count_punc'] = sample['title'].apply(punc_count)


# In[155]:


sample['count_punc']


# In[ ]:





# In[156]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc' , y='views' , data=sample)
plt.show()


# In[157]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc' , y='likes' , data=sample)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




