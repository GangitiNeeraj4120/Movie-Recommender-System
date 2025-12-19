import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
   response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=4b81b5eea02ab7b298f6a995abd00b5e&language=en-US".format(movie_id))
   data = response.json()
   return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def get_base_title(title, n_words=2):
  clean_title = title.split(':')[0].split('-')[0]
  clean_title = clean_title.lower().strip()
  words = clean_title.split()
  return " ".join(words[:n_words])

import re
def get_franchise(idx, threshold=0.25):
  base_title = get_base_title(movies.iloc[idx].title)
  pattern = rf"\b{re.escape(base_title)}\b"
  franchise = movies[movies['title'].str.lower().str.contains(pattern, regex=True, na=False)].index.tolist()
  return [i for i in franchise if i != idx]

def recommend(movie):
    recommended_movies = []
    recommended_movies_poster =[]
    if movie not in movies['title'].values:
     st.write('Movie not found')
     return [], []
    idx = movies[movies['title'] == movie].index[0]
    #Franchise detection
    franchise_idxs = get_franchise(idx)
    if len(franchise_idxs) >= 2:
        for i in franchise_idxs[:5]:
            movie_id = movies.iloc[i].id
            recommended_movies.append(movies.iloc[i].title)
            recommended_movies_poster.append(fetch_poster(movie_id))
        return recommended_movies, recommended_movies_poster
    distances = similarity[idx]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch poster from API
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_poster

st.title("Movie Recommender System")
movies_list = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_list)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

Selected_movie_name = st.selectbox(
    "How would you like to be connected?",
    movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(Selected_movie_name)
    n = min(5, len(names))
    cols = st.columns(n)
    for i in range(len(cols)):
       with cols[i]:
          st.image(
             posters[i],
             caption = names[i],
             use_container_width=True
          )