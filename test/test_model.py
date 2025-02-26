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
        return pd.DataFrame({
            'channel_id': ['1', '2', '3'] * 100,
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'balance_velocity': np.random.rand(100),
            'liquidity_stress': np.random.uniform(0, 1, 100),
            'hour_of_day': np.random.randint(0, 24, 100),
            'day_of_week': np.random.randint(0, 7, 100),
            'balance_ratio': np.random.uniform(0, 1, 100)
        })
    
    @pytest.fixture
    def sample_target(self):
        """ Create sample target data """
        return np.random.uniform(0.3, 0.8, 300)
    
    def test_model_initialization(self):
        """ Test model trainer initialization """
        trainer = ModelTrainer()
        assert trainer is not None
        assert trainer.test_size == 0.2
        assert trainer.random_state == 42

    def test_data_preparation(self, sample_features, sample_target):
        