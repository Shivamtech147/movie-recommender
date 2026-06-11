from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    """Abstract base class for all recommendation models."""

    @abstractmethod
    def fit(self, train_df):
        """Fit the model to the training dataframe.
        
        Args:
            train_df (pd.DataFrame): Dataframe containing 'user_id', 'movie_id', 'rating', 'date'.
        """
        pass

    @abstractmethod
    def predict(self, user_id, movie_id):
        """Predict the rating for a given user and movie.
        
        Args:
            user_id (int): User ID.
            movie_id (int): Movie ID.
            
        Returns:
            float: Predicted rating.
        """
        pass

    @abstractmethod
    def recommend(self, user_id, n_items=10, exclude_rated_items=True):
        """Generate Top-K recommendations for a given user.
        
        Args:
            user_id (int): User ID.
            n_items (int): Number of recommendations to return.
            exclude_rated_items (bool): If True, exclude items the user has already rated in train.
            
        Returns:
            list of tuple: List of (movie_id, predicted_rating).
        """
        pass
