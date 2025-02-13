import pytest
from datetime import datetime, timedelta
from src.utils.lnd_client import LndClient
from src.utils.config import load_config

class TestLndClient:
    @pytest.fixture
    def config(self):
        """ Load test configuration """
        return load_config("configs/test_config.yaml")
    
    @pytest.fixture
    def alice_client(self, config):
        """ Create a client instance for Alice's node """
        return LndClient("alice")
    
    @pytest.fixture
    def bob_client(self, config):
        """ Create a client instance for Bob's node """
        return LndClient("bob")

def test_lnd_client_initialization():
    client = LndClient('alice')
    assert client is not None