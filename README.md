# README for Movie Recommender System

## Overview
This Movie Recommender System is a Streamlit-based application designed to recommend movies to users based on their preferences. It utilizes various Python libraries such as Pandas, Numpy, and custom Streamlit components for an interactive experience. The system has two primary recommendation mechanisms: genre-based recommendations and personalized recommendations using Item-Based Collaborative Filtering (IBCF).

## Features
- **Genre-Based Recommendations**: Users can select a movie genre to receive top movie recommendations within that category.
- **Personalized Recommendations**: The system uses IBCF to recommend movies based on user ratings of a random set of movies.
- **Interactive UI**: Built with Streamlit, the application offers an interactive and user-friendly interface.

## Installation
To run this application, you need to install the following dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To start the application, run:
```bash
streamlit run streamlit_app.py
```

### Key Functions
- `read_data()`: Reads movie, user, and rating data from provided URLs.
- `process_movie_ratings()`: Processes movie ratings to compute average and count.
- `explode_movie_genres()`: Expands movie genres into separate rows for detailed analysis.
- `list_all_genres()`: Lists all unique genres available in the dataset.
- `find_top_movies_by_genre()`: Finds top movies for a given genre based on weighted ratings.
- `sample_random_movies()`: Samples a random set of movies.
- `calculate_similarity_matrix()`: Calculates a cosine similarity matrix from the ratings matrix.
- `myIBCF()`: Implements Item-Based Collaborative Filtering for personalized recommendations.
- `test_myIBCF()`: Tests the IBCF function.
- `generate_similarity_matrix()`: Generates a similarity matrix for IBCF.
- `generate_genre_based_recommendations()`: Generates recommendations based on movie genres.
- `get_all_genre()`, `get_set_size()`, `get_random_movies_collection()`, `get_recommendations()`: Utility functions for the Streamlit interface.

### Streamlit UI Components
The UI is built with various Streamlit components, allowing users to interact with the system, rate movies, and receive recommendations.

## Feedback
The application includes a feedback section using `st_text_rater` for user feedback.

## Important Notes
- The application uses caching (`@st.cache_data`) for efficiency.
- Logging and error handling are implemented for debugging and monitoring.
