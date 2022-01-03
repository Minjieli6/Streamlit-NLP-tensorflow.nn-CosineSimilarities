# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 19:45:50 2021

@author: MINJIE.LI
"""
import re
import string
import math
from datetime import date
from nltk.stem import PorterStemmer

import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from PIL import Image


######################
# Functions
######################

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just']
    
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',str(text))


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


def remove_punctuation(text):
    table=str.maketrans(' ',' ',string.punctuation)
    return text.translate(table)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def final_preprocess(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text
    

######################
# Page Title
######################

image = Image.open('icon.png')

st.sidebar.image(image, use_column_width=True)

st.title('Claim Recommendation Engine')

st.markdown("""
This app is to use machine learning algorithms to recommend the most relevant claims to users.
* **Data:** [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/data)
* **Code:** [Github](https://github.com/minjieli6).
* ** Database ODS_CLM:** CLM, NOTE
* **Python libraries:** base64, pandas, streamlit, nltk, tensorflow
""")



######################
# Create Sample Data
######################
with st.expander("code of creating sample data from Kaggle"):
    st.code("""
    from zipfile import ZipFile
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    nlp_list = api.competitions_list(search='nlp')
    nlp_list
    nlp_files= api.competition_list_files(str(nlp_list[0]))
    nlp_files
    api.dataset_download_file(str(nlp_list[0]))
    
    
    zf = ZipFile(str(nlp_list[0])+'.zip')
    zf.extractall() 
    zf.close()"""
    , language="python")

data=pd.read_csv('comments_to_score.csv')
data = data.rename(columns={'comment_id':'CLM_NUM','text':'CLM_DESC_TXT'})

data['CLAIMOWNER'] =np.random.randint(1, 30, data.shape[0])
data['CLAIMOWNER'] = 'CLAIMOWNER_'+data['CLAIMOWNER'].map(str)

data['CLM_LOB_TYP_CD'] =np.random.randint(1, 2, data.shape[0])
data['CLM_LOB_TYP_CD'] = 'CLM_LOB_TYP_CD_'+data['CLM_LOB_TYP_CD'].map(str)

data['CLM_TYP_CD'] =np.random.randint(1, 2, data.shape[0])
data['CLM_TYP_CD'] = 'CLM_TYP_CD_'+data['CLM_TYP_CD'].map(str)

data['CLM_SBTYP_CD'] =np.random.randint(1,2, data.shape[0])
data['CLM_SBTYP_CD'] = 'CLM_SBTYP_CD_'+data['CLM_SBTYP_CD'].map(str)

data['temp1'] = np.random.randint(0, 100, data.shape[0])
data['temp2'] = np.random.randint(0, 30000000, data.shape[0])
data['PAYMENT_AMT']=np.where(data['temp1']<=np.random.randint(5, 65),0,data['temp2'])


data['temp1'] = np.random.randint(0, 100, data.shape[0])
data['temp2'] = np.random.randint(0, 30000, data.shape[0])
data['RECOVERY_AMT']=np.where(data['temp1']<=np.random.randint(0, 75),0,data['temp2'])
data = data.drop(['temp1', 'temp2'], axis=1)

data['CLM_STS_CD'] = np.random.choice(['open', 'closed'], size=len(data))

data['TXT'] =  None

data['CLM_CLS_DTTM'] = np.where(data['CLM_STS_CD']=='closed',date.today(),None)

######################
# Side Bar
######################
#@st.cache
#data_load_state = st.text('Loading data...')
#data_load_state.text("Done! (using st.cache)")
df1 =data[data['CLM_STS_CD']=='closed']
df2 = data[data['CLM_STS_CD']=='open']
keep_cols = ['CLM_NUM','CLM_CLS_DTTM','CLM_STS_CD','CLM_DESC_TXT','CLAIMOWNER', 'PAYMENT_AMT','RECOVERY_AMT']

st.sidebar.header('Option 1: Claim Number')


sorted_unique_openclm = sorted(df2[(df2['CLM_STS_CD']=='open')].CLM_NUM.unique())
selected_claim = st.sidebar.selectbox('CLM_NUM', sorted_unique_openclm) #list(reversed(range(1950,2020)))
check1 = st.sidebar.button("Check claim or not?")

#from state import provide_state

# @provide_state
# def main(state):
#     state.inputs = state.inputs or set()

#     c1, c2 = st.sidebar.beta_columns([2, 1])

#     input_string = c1.text_input("Input")
#     state.inputs.add(input_string)

#     # Get the last index of state.inputs if it's not empty
#     last_index = len(state.inputs) - 1 if state.inputs else None

#     # Automatically select the last input with last_index
#     c2.selectbox("Input presets", options=list(state.inputs), index=last_index)

# if __name__ == "__main__":
#     main()



st.sidebar.header('Option 2: Claim Description')
# Sidebar - Team selection
sorted_unique_lob = sorted(df1.CLM_LOB_TYP_CD.unique())
selected_lob = st.sidebar.selectbox('CLM_LOB_TYP_CD', sorted_unique_lob)

# Sidebar - Team selection
sorted_unique_clmtyp = sorted(list(filter(None,df1.CLM_TYP_CD.unique())))
selected_clmtyp = st.sidebar.selectbox('CLM_TYP_CD', sorted_unique_clmtyp)

input_text = st.sidebar.text_area("Enter some text to decribe claim")
check2 = st.sidebar.button("Check description or not?")



if check1:
    st.markdown("Based on the **selected claim number**:")
    #selected_claim = '5015038130-1'
    input_clm_typ_cd = df2[df2['CLM_NUM']==selected_claim]['CLM_TYP_CD'].values[0]
    input_clm_sbtyp_cd = df2[df2['CLM_NUM']==selected_claim]['CLM_SBTYP_CD'].values[0]
    input_clm_lob_typ_cd = df2[df2['CLM_NUM']==selected_claim]['CLM_LOB_TYP_CD'].values[0]
    input_clm_desc =  df2[df2['CLM_NUM']==selected_claim]['CLM_DESC_TXT'].values[0]
    input_clm_txt =  df2[df2['CLM_NUM']==selected_claim]['TXT'].values[0]

    st.write("This is ",input_clm_lob_typ_cd, ", ",input_clm_typ_cd,", and ",input_clm_sbtyp_cd," claim.")
    with st.expander("Input text"):
        st.write("""**CLM_DESC_TXT:**""", input_clm_desc)
        st.write("""**CLM_NOTE:**""")
        st.write(input_clm_txt)
    temp = df1[((df1['CLM_STS_CD'] =='closed')&(df1['CLM_TYP_CD']==input_clm_typ_cd)&(df1['CLM_SBTYP_CD']==input_clm_sbtyp_cd)&(df1['CLM_LOB_TYP_CD']==input_clm_lob_typ_cd))]
    df = pd.concat([df2[df2['CLM_NUM']==selected_claim][keep_cols+['TXT']],temp[keep_cols+['TXT']][:19999]])

if check2:
    st.markdown("Based on the **input claim description**:")
    #input_text= 'water damage'
    #selected_clmtyp = 'property'
    #selected_lob = 'HomeownersLine_HOE'
    st.write("This is ",selected_lob, " ",selected_clmtyp," claim.")
    with st.expander("Input text"):
        st.write(input_text)
    d = {'CLM_NUM': 9999999999, 'CLM_CLS_DTTM': None,'CLM_STS_CD':['open'],'CLM_DESC_TXT':None,'TXT':[input_text],'CLAIMOWNER':None, 'PAYMENT_AMT':None,'RECOVERY_AMT':None}
    temp = df1[((df1['CLM_STS_CD'] =='closed')&(df1['CLM_TYP_CD']==selected_clmtyp)&(df1['CLM_LOB_TYP_CD']==selected_lob))]
    df2 = pd.DataFrame(data=d)
    df = pd.concat([df2[keep_cols+['TXT']], temp[keep_cols+['TXT']][:19999]])
    
 

######################
# Sample Data
######################


df = df.drop_duplicates(subset=None, keep='first')
df_closed  = df.copy()
df_closed.isna().sum()


for col in ['TXT','CLM_DESC_TXT']:
    df_closed[col] = df_closed[col].apply(remove_URL)
    df_closed[col] = df_closed[col].apply(remove_emoji)
    df_closed[col] = df_closed[col].apply(remove_punctuation)
    df_closed[col] = df_closed[col].apply(decontracted)
    df_closed[col] = df_closed[col].apply(final_preprocess)
    df_closed[col] = df_closed[col].str.lower()
    #df_closed[col] = df_closed[col].map(lambda x : x.replace(' ', ''))
    
df_closed['ALL_TXT'] = df_closed['TXT'].str.replace('none', '')+' '+df_closed['CLM_DESC_TXT'].str.replace('none', '')
df_closed = df_closed[(df_closed['ALL_TXT'].str.len()>=3)]


#############################
# Universal Sentence Encoder
# source: https://www.kaggle.com/deepakd14/sentence-similarity#6.-universal-sentence-encoder-Model
# install issue: pip uninstall tf-estimator-nightly tensorflow-estimator && pip install tf-estimator-nightly==2.4.0.dev2020101001
#############################
import tensorflow as tf
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def run_sts_benchmark(batch, col1='ALL_TXT',col2='target_TXT'):
  sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(batch[col1].tolist())), axis=1)
  sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(col2)), axis=1)
  cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
  clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
  scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
  """Return the similarity scores"""
  return scores

if check1:
    #selected_claim = '5015038130-1'
    print(df_closed[df_closed['CLM_NUM']==selected_claim]['ALL_TXT'].values)
    target_text = df_closed[df_closed['CLM_NUM']==selected_claim]['ALL_TXT'].values
    df_closed['tf_score'] = run_sts_benchmark(df_closed, col1='ALL_TXT',col2=target_text)
    df_closed['tf_score']=df_closed['tf_score'].astype(float,errors='ignore')
if check2:
    target_text = df_closed[df_closed['CLM_NUM']==9999999999]['ALL_TXT'].values
    df_closed['tf_score'] = run_sts_benchmark(df_closed, col1='ALL_TXT',col2=target_text)
    df_closed['tf_score']=df_closed['tf_score'].astype(float,errors='ignore')
    
######################
# Sample Data
######################
temp = df[keep_cols+['TXT']]
temp = temp.merge(df_closed[['CLM_NUM','tf_score']],on='CLM_NUM')
temp = temp.sort_values(by='tf_score', ascending=False)
temp = temp.reset_index(drop=True)

st.header('Display Sample Close Claim')
st.write('Data Dimension: ' + str(temp.shape[0]) + ' rows and ' + str(temp.shape[1]) + ' columns.')

#temp = temp.style.format({"PAYMENT_AMT": lambda x : '{:,.2f}'.format(x)}).format({"RECOVERY_AMT": lambda x : '{:,.2f}'.format(x)})
st.dataframe(temp[:1000])

# Download sample data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="df.csv">Download CSV File (1000 rows)</a>'
    return href

st.markdown(filedownload(temp[:1000]), unsafe_allow_html=True)


# Heatmap
import plotly.graph_objects as go
#import numpy as np
#if st.button('Intercorrelation Heatmap'):
df_plot = temp.copy()
df_plot = df_plot[df_plot['CLM_STS_CD']=='closed']
df_plot['CWP'] = np.where(df_plot['PAYMENT_AMT']==0,1,0)
df_plot['Recovered'] = np.where(df_plot['RECOVERY_AMT']>0,1,0)

n = df_plot[df_plot['tf_score']>=.65].shape[0]
pct_cwp_sub = df_plot['CWP'][:n].sum()/df_plot['CWP'][:n].count()*100
pct_recovered_sub = df_plot['Recovered'][:n].sum()/df_plot['Recovered'][:n].count()*100

pct_cwp = df_plot['CWP'].sum()/df_plot['CWP'].count()*100
pct_recovered = df_plot['Recovered'].sum()/df_plot['Recovered'].count()*100



######################
# KPI: payment distribution
######################
# Quantile lines
quant_5, quant_25, quant_50, quant_75, quant_95 = np.percentile(list(df_plot['PAYMENT_AMT']), 5), np.percentile(list(df_plot['PAYMENT_AMT']), 25), np.percentile(list(df_plot['PAYMENT_AMT']), 50),np.percentile(list(df_plot['PAYMENT_AMT']), 75), np.percentile(list(df_plot['PAYMENT_AMT']), 95)

st.header('KPI: payment distribution')
recommend_amt = sum([df_plot['PAYMENT_AMT'].median(),df_plot['PAYMENT_AMT'].mean(),df_plot[:1000]['PAYMENT_AMT'].median(),df_plot[:1000]['PAYMENT_AMT'].mean(),df_plot[:100]['PAYMENT_AMT'].median(),df_plot[:100]['PAYMENT_AMT'].mean()])/6
st.write('recommend to book reserve of $',"{:,.2f}".format(recommend_amt))

d = {'Min':[int(df_plot['PAYMENT_AMT'].min())],
     '25th_percentile': [int(quant_25)],
     'Median':[int(df_plot['PAYMENT_AMT'].median())],
     '75th_percentile':[int(quant_75)],
     "Max": [int(df_plot['PAYMENT_AMT'].max())]}


#pd.options.display.float_format ='{:,d}'.format
payment = pd.DataFrame(data=d, index=[0])
payment = payment.style.format('{:,}')
st.dataframe(payment)

######################
# plot
######################
def xrg(x):
    if x>=.9: return 'superior recommend'
    elif x>=.8: return 'great recommend'
    elif x >=.7: return 'good recommend'
    else: return 'fair recommend'

df_plot['score_grp'] = df_plot['tf_score'].apply(xrg)
#import plotly.figure_factory as ff    
import plotly.express as px
#x = px.data.tips()
with st.expander("Histogram of non-zero indemnity between 5th and 95th percentile"):
    #fig = plt.figure(figsize=(10, 4))
    #sns.histplot(data=df_plot[(df_plot['PAYMENT_AMT']>max([quant_5,0]))&(df_plot['PAYMENT_AMT']<quant_95)], x='PAYMENT_AMT', kde=True)
    fig =px.histogram(df_plot[(df_plot['PAYMENT_AMT']>0)], x="PAYMENT_AMT",color="score_grp", marginal="violin",# or violin, rug
                       hover_data=df_plot[['CLM_NUM','CLAIMOWNER','PAYMENT_AMT','RECOVERY_AMT','tf_score','score_grp']].columns)
    plt.axvline(recommend_amt,linewidth=4, color='r')
    st.plotly_chart(fig, use_container_width=True)

######################
# KPI: closed without payment
######################
st.header('KPI: % chance of claims closed without payment')
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = pct_cwp_sub,
    mode = "gauge+number+delta",
    title = {'text': "% chance of claims closed without payment"},
    delta = {'reference': pct_cwp},
    gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 20], 'color': "red"},
                 {'range': [20, 40], 'color': "orange"},
                 {'range': [40, 60], 'color': "yellow"},
                 {'range': [60, 80], 'color': "greenyellow"},
                 {'range': [80, 100], 'color': "limegreen"}],
             'threshold' : {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': pct_cwp}}))

#fig.show()
st.plotly_chart(fig, use_container_width=True)


######################
# KPI: recovery distribution
######################
st.header('KPI: recovery distribution')

# Quantile lines
quant_5, quant_25, quant_50, quant_75, quant_95 = np.percentile(list(df_plot['RECOVERY_AMT']), 5), np.percentile(list(df_plot['RECOVERY_AMT']), 25), np.percentile(list(df_plot['RECOVERY_AMT']), 50),np.percentile(list(df_plot['RECOVERY_AMT']), 75), np.percentile(list(df_plot['RECOVERY_AMT']), 95)
recommend_amt = sum([df_plot['RECOVERY_AMT'].median(),df_plot['RECOVERY_AMT'].mean(),df_plot[:1000]['RECOVERY_AMT'].median(),df_plot[:1000]['RECOVERY_AMT'].mean(),df_plot[:100]['RECOVERY_AMT'].median(),df_plot[:100]['RECOVERY_AMT'].mean()])/6
st.write('potential recovery amount of $',"{:,.2f}".format(recommend_amt))

d = {'Min':[int(df_plot['RECOVERY_AMT'].min())],
     '25th_percentile': [int(quant_25)],
     'Median':[int(df_plot['RECOVERY_AMT'].median())],
     '75th_percentile':[int(quant_75)],
     "Max": [int(df_plot['RECOVERY_AMT'].max())]}


#pd.options.display.float_format ='{:,d}'.format
recovery = pd.DataFrame(data=d, index=[0])
recovery = recovery.style.format('{:,}')
st.dataframe(recovery)


######################
# plot
######################
with st.expander("Histogram of non-zero recovery between 5th and 95th percentile"):
    #st.subheader('histplot of recovery between 5th and 95th percentile')
    #fig = plt.figure(figsize=(10, 4))
    #sns.histplot(data=df_plot[(df_plot['RECOVERY_AMT']>max([quant_5,0]))&(df_plot['RECOVERY_AMT']<quant_95)], x='RECOVERY_AMT', kde=True)
    fig =px.histogram(df_plot[(df_plot['RECOVERY_AMT']>0)], x="RECOVERY_AMT",color="score_grp", marginal="violin",# or violin, rug
                       hover_data=df_plot[['CLM_NUM','CLAIMOWNER','PAYMENT_AMT','RECOVERY_AMT','tf_score','score_grp']].columns)
    plt.axvline(quant_75,linewidth=4, color='r')
    st.plotly_chart(fig, use_container_width=True)

######################
# KPI: % recovery
######################
st.header('KPI: % chance of recovery')
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = pct_recovered_sub,
    mode = "gauge+number+delta",
    title = {'text': "% chance of recovery"},
    delta = {'reference': pct_recovered},
    gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 20], 'color': "red"},
                 {'range': [20, 40], 'color': "orange"},
                 {'range': [40, 60], 'color': "yellow"},
                 {'range': [60, 80], 'color': "greenyellow"},
                 {'range': [80, 100], 'color': "limegreen"}],
             'threshold' : {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': pct_recovered}}))

#fig.show()
st.plotly_chart(fig, use_container_width=True)





  
