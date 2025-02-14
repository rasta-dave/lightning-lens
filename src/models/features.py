import pandas as pd
import numpy as np
from typing import List
from datetime import datetime

class FeatureProcessor:
    """ Process raw channel metrics into features for ML model """

    REQUIRED_COLUMNS = [
        'timestamp', 'channel_id', 'capacity',
        'local_balance', 'remote_balance', 'balance_ratio'
    ]

    def __init__(self):
        """ Initialize feature processor """
        pass

    def validate_input(self, data: pd.DataFrame):
        """ Validate input data has required columns and format """
        if data.empty:
            raise ValueError('Input DataFrame is empty')
        
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            raise ValueError(f'Missing required columns: {missing_cols}')
        
    def calculate_balance_velocity(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Calculate rate of change in local balance """
        df = data.copy()
        df = df.sort_values('timestamp')

        # Calculate balance changes ...
        df['balance_velocity'] = df.groupby('channel_id')['timestamp'].diff().dt.total_seconds() / 3600

        # Calculate time differences in hours ...
        df['time_diff'] = df.groupby('channel_id')['timestamp'].diff().dt.total_seconds() / 3600

        # Calculate velocity (balance change per hour)
        df['balance_velocity'] = df['balance_velocity'] / df['time_diff']

        #Fill first entry (which will be NaN) with 0 ...
        df['balance_velocity'] = df['balance_velocity'].fillna(0)

        # Drop temporary column ...
        df = df.drop('time_diff', axis=1)

        return df