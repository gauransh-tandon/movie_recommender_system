import numpy as np
import pandas as pd
import streamlit as st
from streamlit_star_rating import st_star_rating
from streamlit_text_rating.st_text_rater import st_text_rater

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    layout="wide", page_title="Movie Recommender System", page_icon="❄️"
)


import streamlit as st
import pandas as pd
import numpy as np

# Constants for file paths or URLs
URL_MOVIES = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
URL_RATINGS = 'https://liangfgithub.github.io/MovieData/ratings.dat?raw=true'
URL_USERS = 'https://liangfgithub.github.io/MovieData/users.dat?raw=true'
URL_RATING_MATRIX = 'https://project4-movie-recommender.s3.amazonaws.com/project_4_Rmat.csv'

@st.cache_data
def read_data():
    """Reading data from URLs"""
    try:
        logger.info("Reading data from URLs")
        users = pd.read_csv(URL_USERS, sep='::', engine='python', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode'], dtype={'UserID': int})
        movies = pd.read_csv(URL_MOVIES, sep='::', engine='python', encoding="ISO-8859-1", header=None, names=['MovieID', 'Title', 'Genres'], dtype={'MovieID': int})
        ratings = pd.read_csv(URL_RATINGS, sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype={'MovieID': int, 'UserID': int, 'Rating': int})
        ratings_matrix = pd.read_csv(URL_RATING_MATRIX, sep=',')
        logger.info("Data successfully read")
        return ratings, movies, users, ratings_matrix
    except Exception as e:
        logger.error(f"Error reading data: {e}", exc_info=True)
        st.error(f"Error reading data: {e}")
        return None, None, None, None

@st.cache_data
def process_movie_ratings(user_ratings, movies_info):
    """Process movie ratings to compute average and count, and combine with movie info."""
    try:
        logger.info("Processing movie ratings to compute average and count, and combine with movie info.")
        ratings_agg = user_ratings.groupby("MovieID")['Rating'].agg(AvgRating='mean', RatingCount='count').reset_index()
        avg_count = ratings_agg['RatingCount'].mean()
        min_rating = ratings_agg['AvgRating'].min()
        ratings_agg['WeightedRating'] = (ratings_agg['AvgRating'] * ratings_agg['RatingCount'] + min_rating * avg_count) / (ratings_agg['RatingCount'] + avg_count)

        return movies_info.merge(ratings_agg, on="MovieID", how='left')
    except Exception as e:
        error_message = f'Error processing movie ratings due to: {e}'
        logger.error(error_message, exc_info=True)
        st.error(error_message)
        return None


@st.cache_data
def explode_movie_genres(movies_with_ratings):
    """Expand movie genres into separate rows."""
    try:
        logger.info("Expanding movie genres into separate rows")
        return movies_with_ratings.assign(Genres=movies_with_ratings['Genres'].str.split('|')).explode('Genres')
    except Exception as e:
        error_message = f'Error expanding all unique genres due to: {e}'
        logger.error(error_message, exc_info=True)
        st.error(error_message)
        return None
    
@st.cache_data
def list_all_genres(genre_ratings):
    """List all unique genres."""
    try:
        logger.info("Listing all unique genres...")
        return genre_ratings['Genres'].unique()
    except Exception as e:
        error_message = f'Error listing all unique genres due to: {e}'
        logger.error(error_message, exc_info=True)
        st.error(error_message)
        return None

@st.cache_data
def find_top_movies_by_genre(genre_ratings, genre, top_n=10):
    """Find top N movies for a given genre based on weighted ratings."""
    return genre_ratings[genre_ratings['Genres'] == genre].sort_values(by='WeightedRating', ascending=False).head(top_n)

@st.cache_data
def find_top_movies_by_genre(genre, top_n=10):
    """Find top N movies for a given genre based on weighted ratings."""
    top_movies = genre_movie_ratings[genre_movie_ratings['Genres'] == genre]
    top_movies = top_movies.sort_values(by='Weighted_Rating', ascending=False)    
    top_movies = top_movies[0:top_n]
    return top_movies
   
@st.cache_data
def sample_random_movies(movies_info, sample_size=10):
    """Sample a random set of movies."""
    try:
        logger.info("Sampling random set of movies...")
        return movies_info.sample(n=sample_size)
    except Exception as e:
        error_message = f"Error sampling random set of movies due to: {e}"
        logger.error(error_message, exc_info=True)
        st.error(error_message)
        return None

@st.cache_data
def calculate_similarity_matrix(ratings_matrix):
    """Calculate cosine similarity matrix from ratings matrix."""
    try:
        # Start the calculation process
        logger.info("Starting to calculate the similarity matrix.")

        # Normalize the ratings matrix by subtracting the mean
        logger.info("Normalizing the ratings matrix.")
        normalized_matrix = ratings_matrix.subtract(ratings_matrix.mean(axis=1), axis='rows').T.fillna(0)

        # Calculate the numerator of the cosine similarity formula
        logger.info("Calculating the numerator for cosine similarity.")
        numerator = normalized_matrix @ normalized_matrix.T

        # Calculate the denominator of the cosine similarity formula
        logger.info("Calculating the denominator for cosine similarity.")
        squared_normalized = (normalized_matrix ** 2).dot((normalized_matrix != 0).T)
        denominator = np.sqrt(squared_normalized) * np.sqrt(squared_normalized.T)

        # Compute cosine similarity
        logger.info("Computing the cosine similarity.")
        cosine_similarity = numerator / denominator

        # Convert cosine similarity to a similarity measure that ranges from 0 to 1
        similarity_matrix = (1 + cosine_similarity) / 2

        # Set diagonal elements to NaN and filter out low-cardinality pairs
        logger.info("Adjusting diagonal and filtering low-cardinality pairs.")
        np.fill_diagonal(similarity_matrix.values, np.nan)
        similarity_matrix[similarity_matrix.count() < 3] = None

        logger.info("Similarity matrix calculation completed successfully.")
        return similarity_matrix
    except Exception as e:
        error_message = f"Error calculating similarity matrix: {e}"
        logger.error(error_message, exc_info=True)
        st.error(error_message)

# Note: myIBCF may not need caching as it might be user-specific and dynamic
def myIBCF(similarity_mat, user_ratings, top_n=10):
    """Implement Item-Based Collaborative Filtering."""
    try:
        logger.info("Starting Item-Based Collaborative Filtering")

        # Replace NaN values with zero in similarity matrix and user ratings
        similarity_mat = similarity_mat.fillna(0)
        user_ratings = user_ratings.fillna(0)
        logger.info("NaN values replaced with zero in matrices")

        # Creating a binary identity matrix to identify rated movies
        identity = (~user_ratings.isna()).astype(int)

        # Compute the recommended movies based on similarity matrix and user ratings
        recommended_movies = (user_ratings @ similarity_mat) / identity.dot(similarity_mat)
        recommended_movies = recommended_movies.sort_values(ascending=False).head(top_n).dropna()
        logger.info(f"Computed top {top_n} recommended movies")

        # Check if enough recommendations are available, backfill if necessary
        if recommended_movies.size < top_n:
            backfill_count = top_n - recommended_movies.size
            logger.info(f"Not enough recommendations, backfilling {backfill_count} movies")
            random_genre = np.random.choice(list_all_genres(genre_ratings))
            backfill_movies = find_top_movies_by_genre(genre_ratings, random_genre, backfill_count)
            backfill_series = pd.Series(data=backfill_movies["WeightedRating"].values, index="m" + backfill_movies["MovieID"].astype(str))
            recommended_movies = pd.concat([recommended_movies, backfill_series], axis=0)
            logger.info("Backfilling completed")

        return recommended_movies
    except Exception as e:
        logger.error(f"Error implementing Item-Based Collaborative Filtering: {e}", exc_info=True)
        return None

def test_myIBCF(rating_matrix, similarity_matrix):
    """Test to verify that the custom myIBCF function works."""
    try:
        logger.info("Starting the test to verify custom myIBCF function.")
        user_rating_1 = rating_matrix.loc["u1181"].copy()
        logger.info(myIBCF(similarity_matrix, user_rating_1))
        user_rating_2 = rating_matrix.loc["u1351"].copy()
        logger.info(myIBCF(similarity_matrix, user_rating_2))

        row = similarity_matrix.iloc[0, :]
        user_rating_new = row.copy()
        user_rating_new[:] = np.nan
        user_rating_new["m1613"] = 5
        user_rating_new["m1755"] = 4

        logger.info(myIBCF(similarity_matrix, user_rating_new))

        row = similarity_matrix.iloc[0, :]
        user_rating_nan = row.copy()
        user_rating_nan[:] = np.nan
        logger.info(myIBCF(similarity_matrix, user_rating_nan))
    except Exception as e:
            logger.error(f"Error implementing Item-Based Collaborative Filtering: {e}", exc_info=True)

@st.cache_data
def generate_similarity_matrix():
    try:
        logger.info("Generating similarity matrix")

        # Ensure the rating matrix URL or path is defined globally or within this function
        URL_RATING_MATRIX = 'https://project4-movie-recommender.s3.amazonaws.com/project_4_Rmat.csv'

        # Load the rating matrix
        rating_matrix = pd.read_csv(URL_RATING_MATRIX, sep=',')

        # Call calculate_similarity_matrix to compute the similarity matrix
        similarity_matrix = calculate_similarity_matrix(rating_matrix)

        logger.info("Similarity matrix generated successfully")
        return similarity_matrix
    except Exception as e:
        error_message = f"Error generating similarity matrix: {e}"
        logger.error(error_message, exc_info=True)
        st.error(error_message)
        return None

@st.cache_data
def generate_genre_based_recommendations():
    """Generate movie recommendations based on genres."""
    try:
        logger.info("Starting the generation of genre-based recommendations.")

        # Read data from source
        logger.info("Reading movies and ratings data.")
        ratings, movies, _, _ = read_data()

        # Merging movie ratings with movie details
        logger.info("Merging ratings with movie data.")
        rating_merged = ratings.merge(movies, left_on='MovieID', right_on='MovieID')
        
        # Calculate mean and count of ratings for each movie
        logger.info("Calculating mean and count of ratings for each movie.")
        movie_rating = rating_merged[['MovieID', 'Rating']].groupby('MovieID').agg(['mean', 'count']).droplevel(0, axis=1).reset_index()
        movie_rating.rename(columns={'mean': 'Rating', 'count': 'Rating_count'}, inplace=True)

        # Calculate weighted rating
        logger.info("Calculating weighted ratings for movies.")
        avg_rating_count = movie_rating['Rating_count'].mean()
        avg_rating = movie_rating['Rating'].min()
        movie_rating['Weighted_Rating'] = (movie_rating['Rating'] * movie_rating['Rating_count'] + avg_rating * avg_rating_count) / (movie_rating['Rating_count'] + avg_rating_count)

        # Join the weighted ratings with the movies data
        logger.info("Joining weighted ratings with the movies data.")
        movie_with_rating = movies.join(movie_rating.set_index('MovieID'), on='MovieID')
        movie_with_rating['Weighted_Rating'].fillna(value=avg_rating, inplace=True)

        # Expand the genres into separate rows
        logger.info("Expanding genres for each movie.")
        genre_movie_ratings = movie_with_rating.copy()
        genre_movie_ratings['Genres'] = genre_movie_ratings['Genres'].str.split('|')
        genre_movie_ratings = genre_movie_ratings.explode('Genres')

        logger.info("Genre-based recommendations generated successfully.")
        return genre_movie_ratings
    except Exception as e:
        error_message = f"Error generating genre-based recommendations: {e}"
        logger.error(error_message, exc_info=True)
        st.error(error_message)
        return None

(ratings, movies, users, rating_matrix) = read_data()
(genre_movie_ratings) = generate_genre_based_recommendations()
similarity_matrix = generate_similarity_matrix()
test_myIBCF(rating_matrix, similarity_matrix)

@st.cache_data
def get_all_genre():
    genres = genre_movie_ratings['Genres'].unique()
    return genres

@st.cache_data
def get_set_size():
    return [5,10,50,100]

@st.cache_data
def get_random_movies_collection(n=10):
    movies_collection = movies.sample(n)
    return movies_collection

@st.cache_data
def get_recommendations():
    """
    Generate movie recommendations based on user ratings.
    """
    try:
        logger.info("Starting to generate recommendations based on user ratings.")

        # Convert user star ratings to a NumPy array and assign it to the movie set
        logger.info("Converting user star ratings to an array.")
        movies_collection["star"] = np.array(star_list)

        # Prepare user ratings for the recommendation algorithm
        logger.info("Preparing user ratings for the recommendation algorithm.")
        row = similarity_matrix.iloc[0]
        user_ratings = row.copy()
        user_ratings[:] = np.nan

        # Loop through the movie set and update user ratings
        logger.info("Updating user ratings based on the movie set.")
        for i in range(movies_collection.shape[0]):
            key = "m" + str(movies_collection.iloc[i]["MovieID"])
            if key in user_ratings:
                value = movies_collection.iloc[i]["star"]
                user_ratings.loc[key] = value

        # Call the myIBCF function to get recommendations
        logger.info("Calling the myIBCF function to get recommendations.")
        recommended_movies = myIBCF(similarity_matrix, user_ratings)

        # Filter the movies dataframe to get the final recommended movies
        logger.info("Filtering the movies dataframe to obtain final recommendations.")
        recommended_movies = movies[movies["MovieID"].isin(recommended_movies.index.str.slice(1).astype(int))]

        logger.info("Recommendations generated successfully.")
        return recommended_movies
    except Exception as e:
        error_message = f"Error in generating recommendations: {e}"
        logger.error(error_message, exc_info=True)

def display_genre_based_recommendations():
    st.header("System I - Genre-Based Recommendations")
    selected_genre = st.selectbox('Select a Genre', get_all_genre())
    
    top_movies = find_top_movies_by_genre(genre=selected_genre, top_n=10)

    cols = st.columns(10)

    (row, _) = top_movies.shape
    for i in range(row):
        record = top_movies.iloc[i, :]
        with cols[i % 10]:
            title = record['Title']
            st.subheader(f"{title}")
            st.text(f"Rank: {i + 1}")
            rating = f"{np.round(record['Rating'], 2)} ⭐"
            st.text(f"Rating: {rating}")
            image_url = f'https://liangfgithub.github.io/MovieImages/{record["MovieID"]}.jpg'
            st.image(image_url)
            st.divider()
        

# Streamlit UI components
st.header("Movie Recommender System")

# Sidebar markdown and tabs
with st.sidebar:
    st.markdown("# Navigation Bar")
    selected_tab = st.radio("Choose a system:", ["System I", "System II", "None"])

if selected_tab == "System I":
    display_genre_based_recommendations()

elif selected_tab == "System II":
    with st.container(border=True):

        with st.expander("Step 1: ", expanded=True):
            movies_collection_size = st.selectbox('Select number of movies to choose from', get_set_size())

            st.info("Rate as many movies as possible")
            movies_collection = get_random_movies_collection(n=movies_collection_size)

            cols = st.columns(10)

            # Show Movie Set for User Rating
            star_list = list()
            (row, _) = movies_collection.shape
            for i in range(row):
                record = movies_collection.iloc[i, :]
                with cols[i % 10]:
                    title = record['Title']
                    st.subheader(f"{title}")

                    image_url = f'https://liangfgithub.github.io/MovieImages/{record["MovieID"]}.jpg'
                    st.image(image_url)

                    star = st_star_rating("Give your rating", maxValue=5, defaultValue=0, key=f"stars_{i}")
                    star_list.append(star)

        with st.expander("Step 2:", expanded=True):
            st.info("Discover movies you might like")
            if st.button("Click here to get your recommendations", type="primary"):
                recommended_movies = get_recommendations()
                (row, _) = recommended_movies.shape

                cols = st.columns(10)

                for i in range(row):
                    record = recommended_movies.iloc[i, :]
                    with cols[i % 10]:
                        title = record['Title']
                        st.subheader(f"{title}")
                        st.text(f"Rank: {i + 1}")

                        image_url = f'https://liangfgithub.github.io/MovieImages/{record["MovieID"]}.jpg'
                        st.image(image_url)

elif selected_tab == "None":
    st.info("Thank you for visiting my movie recommender tool.")

# Feedback section
st.title("Feedback")
for text in ["Is this text helpful?", "Do you like this text?"]:
    response = st_text_rater(text=text)