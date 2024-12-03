import streamlit as st
import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
reader = Reader()
svd=SVD()

movies=pd.read_csv('movies_metadata.csv')
movies=movies.drop([19730, 29503, 35587])
movies['id']=movies['id'].astype(int)
links=pd.read_csv('links_small.csv')
links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')
ratings=pd.read_csv('ratings_small.csv')
df=movies[movies['id'].isin(links)]   #contains the required movies which are in the small dataset
def convert(text):
    L=[]
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L
def extract_genre(dff):
    gen_list=[]
    for i in dff['genres']:
        gen_list.append(i)
    return gen_list
def choose_genre(gen_list):
    st.title("Movie Recommender System")
    st.header("Select up to 5 genres")

    # Initialize session state for genre selection if not already set
    if 'selected_genres' not in st.session_state:
        st.session_state.selected_genres = []

    # Display checkboxes for genres along with session state tracking
    columns = st.columns(2)
    for idx, genre in enumerate(gen_list):
        col = columns[idx % 2]
        with col:
            if st.checkbox(genre, key=genre):
                if genre not in st.session_state.selected_genres:
                    if len(st.session_state.selected_genres) < 5:
                        st.session_state.selected_genres.append(genre)
                    else:
                        st.warning("You can only select up to 5 genres.")
            else:
                if genre in st.session_state.selected_genres:
                    st.session_state.selected_genres.remove(genre)

    # Showing selected genres for verification
    st.text("Currently Selected Genres: " + ", ".join(st.session_state.selected_genres))

    # Confirmation button to finalize the selection
    if st.button("Confirm Genre Selection"):
        if len(st.session_state.selected_genres) > 0:
            st.success("Selected Genres: " + ", ".join(st.session_state.selected_genres))
            # Clearing any previous movie selection/rating states if needed
            st.session_state.selected_movies = []
            st.session_state.movie_ratings = {}
        else:
            st.warning("Please select at least one genre before proceeding.")

    return st.session_state.selected_genres
def build_chart(genres, percentile=0.85):
    dff = movies[movies['genres'].apply(lambda x: any(g in x for g in genres))]

    vote_counts = dff[dff['vote_count'].notnull()]['vote_count'].astype('int')
    vote_avg = dff[dff['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_avg.mean()
    m = vote_counts.quantile(percentile)
    qualified = dff[(dff['vote_count'] >= m) & (dff['vote_count'].notnull()) & (dff['vote_average'].notnull())][
        ['title', 'id', 'overview', 'genres', 'original_language', 'popularity', 'production_companies', 'release_date',
         'vote_average', 'vote_count']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
        axis=1)
    qualified = qualified.sort_values('wr', ascending=False)
    movie_list = pd.DataFrame()
    for genre in genres:
        movie_list_genre = qualified[qualified['genres'].apply(lambda x: genre in x)]
        movie_list_genre.sort_values('wr', ascending=False)
        i = 0
        row = 0
        for index, row in movie_list_genre.iterrows():
            if movie_list.empty:
                movie_list = pd.concat([movie_list, movie_list_genre.iloc[[0]]], ignore_index=True)
                i += 1
            if not ((movie_list == row).all(axis=1)).any():
                movie_list = pd.concat([movie_list, pd.DataFrame([row])], ignore_index=True)
                i += 1
            if i == 8:
                break
    return movie_list
def choose_movies(chart):
    st.text("Enter 5 best movies out of the following")

    # Initialize session state for selected movies and rating flow
    if 'selected_movies' not in st.session_state:
        st.session_state.selected_movies = []
    if 'movie_ratings' not in st.session_state:
        st.session_state.movie_ratings = {}
    if 'selection_submitted' not in st.session_state:
        st.session_state.selection_submitted = False
    if 'ratings_submitted' not in st.session_state:
        st.session_state.ratings_submitted = False

    # Display movies as checkboxes
    st.header("Select Movies")
    st.write("Choose up to 5 movies:")

    columns = st.columns(2)  # Create columns for better layout

    # Movie selection with checkboxes
    for idx, row in chart.iterrows():
        col = columns[idx % 2]
        with col:
            if st.checkbox(row['title'], key=row['id']):
                if row['title'] not in st.session_state.selected_movies and len(st.session_state.selected_movies) < 5:
                    st.session_state.selected_movies.append(row['title'])
            else:
                if row['title'] in st.session_state.selected_movies:
                    st.session_state.selected_movies.remove(row['title'])

    st.text("Currently Selected Movies: " + str(st.session_state.selected_movies))

    # Movie selection submission
    if st.button("Submit Selection") and not st.session_state.selection_submitted:
        if len(st.session_state.selected_movies) > 0:
            st.session_state.selection_submitted = True  # Track that selection was submitted

    # Showing rating inputs only if selection has been submitted
    if st.session_state.selection_submitted:
        st.subheader("Rate Your Selected Movies")

        # Display selected movies and input boxes for ratings
        for movie_selected in st.session_state.selected_movies:
            movie_id = chart[chart['title'] == movie_selected]['id'].values[0]  # Get the movie ID based on the title
            # Initialize rating to 0 if not already set
            if movie_selected not in st.session_state.movie_ratings:
                st.session_state.movie_ratings[movie_selected] = 0.0

            # Display rating input
            rating = st.number_input(f"Rate {movie_selected} (0.0 - 5.0):", min_value=0.0, max_value=5.0,
                                     value=st.session_state.movie_ratings[movie_selected], format="%.1f",
                                     key=f"rating_{movie_selected}")
            st.session_state.movie_ratings[movie_selected] = rating  # Update the session state with the rating

        # Rating submission button
        if st.button("Submit Ratings") or st.session_state.ratings_submitted:   #The or part is so so important
            # Create a DataFrame of selected movies, their IDs, and ratings
            selected_movies_df = pd.DataFrame({
                'movie_id': [chart[chart['title'] == movie_selected]['id'].values[0] for movie_selected in st.session_state.selected_movies],
                'title': st.session_state.selected_movies,
                'rating': list(st.session_state.movie_ratings.values())
            })


            # Mark ratings as submitted and reset selections
            st.session_state.ratings_submitted = True
            st.session_state.selected_movies = []
            st.session_state.movie_ratings = {}
            return selected_movies_df

        elif not st.session_state.selected_movies:
            st.warning("Please select at least one movie before submitting.")
def hybrid(userid, title):
    idx = indices[title]
    tmdbid = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']

    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    moviess = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
    moviess['est'] = moviess['id'].apply(lambda x: svd.predict(userid, indices_map.loc[x]['movieId']).est)
    moviess = moviess.sort_values('est', ascending=False)
    return moviess.head(10)
def recommend_movies(user_similarity, user_movie_matrix, target_user=0, num_recommendations=10):
    #Finding similar users
    similar_users = user_similarity[target_user]  # Index 0 for target user
    similar_users = pd.Series(similar_users, index=user_movie_matrix.index).sort_values(ascending=False)

    # Getting movies rated by similar users that the target user hasn't rated
    target_user_ratings = user_movie_matrix.loc[target_user]
    unrated_movies = target_user_ratings[target_user_ratings == 0].index  # Movies target user hasn't rated

    # Aggregating the ratings for unrated movies which are weighted by similarity score
    weighted_ratings = {}
    for user in similar_users.index[1:]:  # Skip the target user
        similarity_score = similar_users[user]
        user_ratings = user_movie_matrix.loc[user, unrated_movies]
        for movie, rating in user_ratings.items():
            if rating > 0:  # Only consider movies actually rated by similar users
                if movie not in weighted_ratings:
                    weighted_ratings[movie] = 0
                weighted_ratings[movie] += rating * similarity_score

    # Now sorting the movies by weighted rating and returning top recommendations
    recommended_movies = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    recommended_movies = [movie for movie, score in recommended_movies]

    return recommended_movies


df['genres']=df['genres'].apply(convert)
gen_list=extract_genre(df)
flat_gen_list = [item for sublist in gen_list for item in sublist] if any(isinstance(i, list) for i in gen_list) else gen_list
gen_list = list(set(flat_gen_list))
user_genre=choose_genre(gen_list)

chart = build_chart(user_genre).head(30)
pref_movies = choose_movies(chart)
user_pref = pref_movies[['movie_id','rating']]
user_pref['userId']=0
user_pref = user_pref[['userId', 'movie_id', 'rating']]
user_pref = user_pref.rename(columns={'movie_id':'movieId'})
ratings=pd.concat([user_pref,ratings],ignore_index=True)
cosine_sim=pickle.load(open('cosine_sim.pkl','rb'))
indices=pickle.load(open('indices.pkl','rb'))
id_map=pickle.load(open('id_map.pkl','rb'))
smd=pickle.load(open('smd.pkl','rb'))
indices_map = id_map.set_index('id')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd.fit(trainset)

movie=st.selectbox("Select a movie:",options=chart['title'].values)
if st.button('Recommend'):
    st.text(movie)
    # Display recommendations only if a valid movie name is entered
    if movie:
        try:
            result = hybrid(0, movie)
            st.write('You might also like:')
            st.text(result['title'])
        except KeyError:
            st.warning("Movie not found. Please enter a valid movie title.")

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)\
# Computing cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
# user_similarity
recommended=False
if st.button('Recommend movies') or recommended:
    # Using the function to get recommendations for the first user (userId = 0)
    recommended_movies = recommend_movies(user_similarity, user_movie_matrix, target_user=0, num_recommendations=10)
    recommended_titles = movies[movies['id'].isin(recommended_movies)]['title'].values
    recommended=True
    st.subheader("Movie Recommendations for You:")
    for title in recommended_titles:
        st.write(title)

