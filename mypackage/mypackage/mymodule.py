"""
Movie Recommendation System Script

This script builds a movie recommendation system using the SVD (Singular Value Decomposition) algorithm
from the Surprise library. The script performs the following tasks:

1. Loads several datasets related to movie ratings, movie metadata, and additional tags.
2. Defines a `Reader` and `Dataset` for the Surprise library to process the user ratings.
3. Trains an SVD model on the full training dataset.
4. Predicts movie ratings for a test dataset using the trained SVD model.
5. Adjusts the test dataset to include the necessary user and movie IDs for predictions.
6. Generates a submission file containing the predicted ratings in the required format.

Modules:
- pandas: Used for data manipulation and handling of DataFrames.
- numpy: Used for numerical operations.
- surprise: Used for building and training the SVD model.
- sklearn: Used for splitting the data into training and testing sets.

Functions:
- predict_ratings(test_df, svd): Predicts ratings for the given test DataFrame using the provided SVD model.

Outputs:
- A CSV file named 'predictions.csv' containing the predicted ratings in the required format.
"""


import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD, accuracy
from sklearn.model_selection import train_test_split

# Load the datasets
train = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/train.csv')
movies = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/movies.csv')
imdb_data = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/imdb_data.csv')
test = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/test.csv')
links = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/links.csv')
tags = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/tags.csv')
genome_scores = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/genome_scores.csv')
genome_tags = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/genome_tags.csv')
sample_submission = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/sample_submission.csv')

# Define the Reader and Dataset
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)

# Build the trainset
data_trainset = data.build_full_trainset()

# Train the SVD model
svd = SVD()
svd.fit(data_trainset)

# Function to predict ratings for the test DataFrame
def predict_ratings(test_df, svd):
    predictions = []
    for index, row in test_df.iterrows():
        userId = row['userId']
        movieId = row['movieId']
        prediction = svd.predict(userId, movieId)
        predictions.append({
            'userId': userId,
            'movieId': movieId,
            'predicted_rating': prediction.est
        })

    return pd.DataFrame(predictions)

# Adjust the sample_submission DataFrame to include 'userId' and 'movieId' columns
sample_submission[['userId', 'movieId']] = sample_submission['Id'].str.split('_', expand=True).astype(int)

# Make predictions on the test set
predictions_df = predict_ratings(sample_submission, svd)

# Generate the 'Id' column
predictions_df['Id'] = predictions_df['userId'].astype(str) + '_' + predictions_df['movieId'].astype(str)

# Prepare the final submission DataFrame
final_submission = predictions_df[['Id', 'predicted_rating']].rename(columns={'predicted_rating': 'rating'})

# Save the predictions to a CSV file
final_submission.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")

