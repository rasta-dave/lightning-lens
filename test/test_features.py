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
    
    

