import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.features import FeatureProcessor

class TestFeatureProcessor:
    @pytest.fixture
    def sample_data(self):
        """ Create sample channel metrics data """
        dates = pd.date_range(start='2024-01-01', periods=24, freq='H')
        data = []

        for timestamp in dates:
            data.append({
                'timestamp': timestamp,
                'channel_id': '123456',
                'capacity': 1000000,
                'local_balance': 500000 + np.random.randint(-10000, 10000),
                'remote_balance': 500000 + np.random.randint(-10000, 10000),
                'remote_pubkey': 'abc123',
                'balance_ratio': 0.5
            })

        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_processor(self):
        return FeatureProcessor()
    
    def test_calculate_balance_velocity(self, feature_processor, sample_data):
        """ Test calculation of balance change velocity """
        features = feature_processor.calculate_balance_velocity(sample_data)

        assert 'balance_velocity' in features.columns
        assert len(features) == len(sample_data)
        assert not features['balance_velocity'].isnull().any()

    def test_calculate_liquidity_stress(self, feature_processor, sample_data):
        """ Test calculation of liquidity stress indicators """
        features = feature_processor.calculate_liquidity_stress(sample_data)

        assert 'liquidity_stress' in features.columns
        assert all(features['liquidity_stress'].between(0, 1))


