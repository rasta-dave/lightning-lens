import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from src.utils.lnd_client import LndClient
from src.utils.config import load_config
from src.proto import lightning_pb2 as ln

class TestLndClient:
    @pytest.fixture
    def mock_config(self):
        return {
            'nodes': {
                'alice': {
                    'rpc_server': 'http://localhost:10001',
                    'tls_cert_path': '~/.polar/networks/1/volumes/lnd/alice/tls.cert',
                    'macaroon_path': '~/.polar/networks/1/volumes/lnd/alice/data/chain(bitcoin/regtest/admin.macaroon'
                }
            }
        }

    def test_successful_initialization(self, mock_config):
        """ Test successful client initialization """
        # Creating mock file contents ...
        mock_cert = b'mock certificate content'
        mock_macaroon = b'mock macaroon content'

        # Create a mock file handler that retunrs different content for different files ...
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__ = MagicMock()
        mock_file.read.side_effect = [mock_cert, mock_macaroon]

        with patch('src.utils.lnd_client.load_config', return_value=mock_config), \
            patch('builtins.open', MagicMock(return_value=mock_file)), \
            patch('src.utils.lnd_client.grpc.ssl_channel_credentials'), \
            patch('src.utils.lnd_client.grpc.metadata_call_credentials'), \
            patch('src.utils.lnd_client.grpc.composite_channel_credentials'), \
            patch('src.utils.lnd_client.grpc.secure_channel'), \
            patch('src.utils.lnd_client.lnrpc.LightningStub'):

            client = LndClient('alice')
            assert client is not None
            assert client.node_config == mock_config['nodes']['alice']

    def test_invalid_node_name(self):
        with pytest.raises(KeyError) as exc_info:
            LndClient('invalid_node')
        assert 'invalid_node' in str(exc_info.value)

    @patch('src.utils.lnd_client.grpc.ssl_channel_credentials')
    @patch('src.utils.lnd_client.grpc.metadata_call_credentials')
    def test_initialization_with_invalid_cert(self, mock_meta_creds, mock_ssl_creds):
        mock_ssl_creds.side_effect = Exception('Invalid certificate')
        with pytest.raises(Exception) as exc_info:
            LndClient('alice')
        assert 'Invalid certificate' in str(exc_info.value)
