# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st

# Load data
@st.cache(suppress_st_warning=True)
def load_data():
  df = pd.read_csv('bgg_dataset.csv', sep=';')
  return df

df=load_data()

#change datatype
df.loc[13984, ['Year Published']] = 1855 #locate game and update the year from research
df['Year Published'].astype(int)
df['Rating Average'] = df['Rating Average'].replace({',': '.'}, regex=True).astype(float) #convert Rating Average to float
df['Complexity Average'] = df['Complexity Average'].replace({',' : '.'}, regex=True).astype(float)  #convert Complexity Average to float
df['Mechanics'] = df['Mechanics'].fillna('Not Specified')
df['Domains'] = df['Domains'].fillna('Not Specified')

#clean text columns
@st.cache(suppress_st_warning=True)
def Cleaned_text(text):
  clean = re.sub('[^a-zA-Z0-9 ,]', '', str(text))  # substitute any character not in the character list with empty string
  clean = clean.replace("/", ",")  #replace '/' with ','
  return clean

df['Mechanics'] = df['Mechanics'].apply(Cleaned_text)

#create dummy variables
all_dummy_variables = pd.get_dummies(df['Mechanics'].str.split(', ', expand=True).stack(), prefix='mechanic').groupby(level=0).sum()

#create new df
df1 = pd.concat([df, all_dummy_variables], axis=1)
df1 = df1.drop(['Mechanics', 'Users Rated', 'Name', 'Owned Users', 'BGG Rank', 'Domains', 'ID'], axis=1)

#define numeric values
X_continuous = df1[['Year Published', 'Min Players', 'Max Players', 'Play Time', 'Min Age', 'Rating Average', 'Complexity Average']]
#scale numeric values
scaler = StandardScaler()
X_cont_scaled = scaler.fit_transform(X_continuous)
X_cont_scaled_df = pd.DataFrame(X_cont_scaled, columns=X_continuous.columns)
#combine scaled with dummy mechanics variables
X_combined = pd.concat([X_cont_scaled_df, all_dummy_variables], axis=1)

#perform PCA with optimal number of components
optimal_pca = PCA(n_components=8)
X_pca = optimal_pca.fit_transform(X_combined)


#calculate cosine similarity metric  --- dot product of two vectors or items
cosine_similarity_matrix = cosine_similarity(X_pca)

all_titles = df['Name'].values

#create content-based recommendation algorithm using similarity score
#find similar game titles
@st.cache(suppress_st_warning=True)
def find_similar_titles(input_title, all_titles=all_titles):
    similar_titles = []
    for title in all_titles:
        if input_title.lower() in title.lower() and input_title.lower() != title.lower() or "expansion" in title.lower():
            similar_titles.append(title)
    return similar_titles

@st.cache(suppress_st_warning=True)
def recommendations1(name, cosine_similarity_matrix = cosine_similarity_matrix):
  name_lower = name.lower()
  game_found = name_lower in df['Name'].str.lower().values
  if not game_found:  #check if game is in data
    print(f"No game found with the name '{name}'.")
    return
  else:
    game_index = df[df['Name'].str.lower() == name_lower].index[0]
    print(f"Loading recommendations for {name}...")

  sim_score = list(enumerate(cosine_similarity_matrix[game_index]))  #get list of scores
  sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)  #sort the scores highest to lowest
  sim_score = sim_score[1:200]  #get top 15 scores
  item_indices = [i[0] for i in sim_score]  #get index

  return df[['Name', 'Year Published', 'Rating Average', 'Complexity Average', 'Domains', 'Mechanics', 'ID']].iloc[item_indices]  #return title recommendations

@st.cache(suppress_st_warning=True)
def bg_recommendation(name):
  exclusions = find_similar_titles(name) # find similar titles
  recs = recommendations1(name)
  filtered_recs = recs[~recs['Name'].isin(exclusions)] # filter out similar titles
  top_25 = filtered_recs.head(25).reset_index(drop=True) # get top 25

  html_output = ""  #link to the game
  for index, row in top_25.iterrows():
    rec_url = f"https://boardgamegeek.com/boardgame/{row['ID']}/"
    html_output += f"<a href='{rec_url}'>{row['Name']}</a><br>"

  return html_output, top_25

#load pickle model
with open('recmodel.pk1', 'rb') as f:
    load_model = pickle.load(f)
    bg_rec = load_model[0]

# Streamlit UI
st.title('Game Recommendation System 🎲')

# Input field for user to enter game name
game_name = st.text_input('Enter the name of the game:', 'Catan')

# Button to trigger recommendation generation
if st.button('Generate Recommendations'):
    with st.spinner('Generating recommendations...'):
      html_output, recommendations = bg_rec(game_name)
      st.markdown(html_output, unsafe_allow_html=True)

      if not recommendations.empty:
        st.subheader('🌟 Top Recommendations:')
        st.write(recommendations)
      else:
        st.warning('❌ No recommendations found for the entered game name. Please try again with a different name.')
