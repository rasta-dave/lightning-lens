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