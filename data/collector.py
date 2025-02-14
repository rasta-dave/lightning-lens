from datetime import datetime
from src.utils.lnd_client import LndClient
import pandas as pd
from typing import Dict, List

class DataCollector:
    def __init__(self, node_client: LndClient):
        self.client = node_client

    def collect_channel_metrics(self):
        """ Collect current channel metrics """
        try:
            metrics = {
                'timestamp': datetime.now(),
                'channels': self.client.get_channel_balances(),
                'forwarding': self.client.get_forwarding_history()
            }
            return metrics
        except Exception as e:
            raise Exception(f'Error collecting channel metrics: {e}')
        
    def process_metrics(self, raw_metrics: Dict) -> pd.DataFrame:
        """ Process raw metrics into a structured format """
        channels_data = []

        for channel in raw_metrics['channels']:
            channel_data = {
                'timestamp': raw_metrics['timestamp'],
                'channel_id': channel['channel_id'],
                'capacity': channel['capacity'],
                'local_balance': channel['local_balance'],
                'remote_balance': channel['remote_balance'],
                'remote_pubkey': channel['remote_pubkey'],
                'balance_ratio': channel['local_balance'] / channel['capacity'],
            }
            channels_data.append(channel_data)

        return pd.DataFrame(channels_data)
    
    def save_metrics(self, metrics_df: pd.DataFrame, filepath: str):
        """ Save metrics to CSV file """
        metrics_df.to_csv(filepath, index=False)