import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.models.trainer import ModelTrainer

class TestModelTrainer:
    @pytest.fixture
    def sample_features(self):
        """ Create sample feature data """
        n_samples = 100
        # Create repeating channel IDs of exact length
        channel_ids = ['1', '2', '3'] * (n_samples // 3) + ['1', '2', '3'][:n_samples % 3]
        
        return pd.DataFrame({
            'channel_id': channel_ids[:n_samples],  # Ensure exact length
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
            'balance_velocity': np.random.rand(n_samples),
            'liquidity_stress': np.random.uniform(0, 1, n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'balance_ratio': np.random.uniform(0, 1, n_samples)
        })
    
    @pytest.fixture
    def sample_target(self):
        """ Create sample target data """
        n_samples = 100  # Match sample_features size
        return np.random.uniform(0.3, 0.8, n_samples)
    
    def test_model_initialization(self):
        """ Test model trainer initialization """
        trainer = ModelTrainer()
        assert trainer is not None
        assert trainer.test_size == 0.2
        assert trainer.random_state == 42

    def test_data_preparation(self, sample_features, sample_target):
        """ Test data preparation functionality """
        trainer = ModelTrainer()

        # Add target to features
        features_df = sample_features.copy()
        features_df['optimal_ratio'] = sample_target

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            features_df, target_column='optimal_ratio'
        )

        # Check shapes with correct expected sizes
        n_samples = len(sample_features)
        expected_train_size = int(n_samples * 0.8)
        assert len(X_train) == expected_train_size
        assert len(y_train) == expected_train_size
        assert len(X_test) == n_samples - expected_train_size
        assert len(y_test) == n_samples - expected_train_size

        # Check column removal ...
        assert 'channel_id' not in X_train.columns
        assert 'timestamp' not in X_train.columns
        assert 'optimal_ratio' not in X_train.columns

        