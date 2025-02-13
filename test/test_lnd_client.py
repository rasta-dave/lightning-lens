import pytest
from datetime import datetime, timedelta
from src.utils.lnd_client import LndClient
from src.utils.config import load_config
from unittest.mock import MagicMock

# Fixtures moved here
@pytest.fixture
def config():
    """ Load test configuration """
    return load_config("configs/test_config.yaml")

@pytest.fixture
def alice_client(config):
    """ Create a client instance for Alice's node """
    return LndClient("alice")

@pytest.fixture
def bob_client(config):
    """ Create a client instance for Bob's node """
    return LndClient("bob")

# Tests
def test_lnd_client_initialization():
    client = LndClient('alice')
    assert client is not None

def test_get_info(alice_client):
    # Create a mock response directly without using mocker
    mock_response = MagicMock()
    mock_response.identity_pubkey = 'test_pubkey'
    mock_response.alias = 'test_alias'
    mock_response.num_peers = 5
    mock_response.num_active_channels = 3
    mock_response.block_height = 100
    
    # Mock the stub's GetInfo method directly
    alice_client.stub.GetInfo = MagicMock(return_value=mock_response)
    
    # Call the method
    info = alice_client.get_info()
    
    # Assertions
    assert info['pubkey'] == 'test_pubkey'
    assert info['alias'] == 'test_alias'
    assert info['num_peers'] == 5
    assert info['num_active_channels'] == 3
    assert info['blockheight'] == 100

def test_invalid_node_name():
    with pytest.raises(KeyError):
        LndClient('invalid_node')

def test_get_info_mocked():
    client = LndClient('alice')
    client.stub.GetInfo = MagicMock(return_value=MagicMock(
        identity_pubkey='test_pubkey',
        alias='test_alias',
        num_peers=5,
        num_active_channels=3,
        block_height=100
    ))
    info = client.get_info()
    assert info['pubkey'] == 'test_pubkey'

def test_polar_connection(alice_client):
    """Test basic connection to Polar node"""
    try:
        info = alice_client.get_info()
        assert info is not None
        assert 'pubkey' in info
        assert 'alias' in info
        print(f"Successfully connected to node: {info['alias']}")
    except Exception as e:
        pytest.fail(f"Failed to connect to Polar node: {e}")