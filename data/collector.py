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