import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import os
from typing import Tuple, Any

class ModelTrainer:
    """ Trains a model to predict optimal channel balances """

    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """ Initialize the model trainer
        
        Args:
            test_size (float): Proportion of data to reserve for testing
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def prepare_data(self,
                     data: pd.DataFrame,
                     target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """ prepare data for model training by splitting and scaling 
        
        Args:
            data (pd.DataFrame): Input data with features and target
            target_column (str): Name of the target column

        Returns:
            Tuple containing:
                x_train (pd.DataFrame): Training features
                x_test (pd.DataFrame): Testing features
                y_train (np.ndarray): Training targets
                y_test (np.ndarray): Testing targets
        """
        # Seperate features and target ...
        y = data[target_column].values

        # Remove non-feature columns ...
        x = data.drop([target_column, 'channel_id', 'timestamp'], axis=1, errors='ignore')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        # Scale features ...
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns
        )

        return X_train_scaled, X_test_scaled, y_train, y_test

        