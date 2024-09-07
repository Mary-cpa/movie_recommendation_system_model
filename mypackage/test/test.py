from mypackage import mymodule

def test_movie_recommendation_system():
    """
    Test function for the movie recommendation system script.

    This function performs a basic test to ensure that:
    1. The datasets are loaded successfully.
    2. The SVD model is trained without errors.
    3. Predictions are generated and saved to a CSV file.
    """

    try:
        # Load the datasets (using a small sample if datasets are large for quick testing)
        train = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/train.csv', nrows=100)
        test = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/test.csv', nrows=10)
        sample_submission = pd.read_csv('/kaggle/input/alx-movie-recommendation-project-2024/sample_submission.csv', nrows=10)

        # Ensure the dataset loaded correctly
        assert not train.empty, "Train dataset is empty!"
        assert not test.empty, "Test dataset is empty!"
        assert not sample_submission.empty, "Sample submission dataset is empty!"

        # Define the Reader and Dataset
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)

        # Build the trainset
        data_trainset = data.build_full_trainset()

        # Train the SVD model
        svd = SVD()
        svd.fit(data_trainset)

        # Make predictions on the test set
        sample_submission[['userId', 'movieId']] = sample_submission['Id'].str.split('_', expand=True).astype(int)
        predictions_df = predict_ratings(sample_submission, svd)

        # Check if predictions were made
        assert not predictions_df.empty, "Predictions DataFrame is empty!"
        assert 'predicted_rating' in predictions_df.columns, "'predicted_rating' column is missing in predictions DataFrame!"

        # Save predictions to a CSV file
        predictions_df['Id'] = predictions_df['userId'].astype(str) + '_' + predictions_df['movieId'].astype(str)
        final_submission = predictions_df[['Id', 'predicted_rating']].rename(columns={'predicted_rating': 'rating'})
        final_submission.to_csv('predictions.csv', index=False)

        print("Test passed: Predictions saved to predictions.csv")

    except Exception as e:
        print(f"Test failed: {e}")

# Run the test function
test_movie_recommendation_system()
