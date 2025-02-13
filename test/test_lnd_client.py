import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open
from src.utils.lnd_client import LndClient
from src.proto import lightning_pb2 as ln
import grpc

class TestLndClient:
    @pytest.fixture
    def mock_config(self):
        """ Provide a mock config for testing """
        return {
            'nodes': {
                'alice': {
                    'rpc_server': 'http://localhost:10001',
                    'tls_cert_path': '~/.polar/networks/1/volumes/lnd/alice/tls.cert',
                    'macaroon_path': '~/.polar/networks/1/volumes/lnd/alice/data/chain(bitcoin/regtest/admin.macaroon'
                }
            }
        }

    @pytest.fixture
    def mock_client(self):
        """ Create a mock LND client """
        with patch('src.utils.lnd_client.load_config'), \
            patch('builtins.open', mock_open(read_data=b'mock_cert_data')), \
            patch('src.utils.lnd_client.grpc.ssl_channel_credentials'), \
            patch('src.utils.lnd_client.grpc.metadata_call_credentials'), \
            patch('src.utils.lnd_client.grpc.composite_channel_credentials'), \
            patch('src.utils.lnd_client.grpc.secure_channel'), \
            patch('src.utils.lnd_client.lnrpc.LightningStub'):
            client = LndClient('alice')
            client.stub = MagicMock()
            return client


    def test_successful_initialization(self, mock_config):
        """ Test successful client initialization """
        # Creating mock file contents ...
        mock_cert = b'mock certificate content'
        mock_macaroon = b'mock macaroon content'

        # Create a mock file handler that returns different content for different files ...
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

    def test_get_info_success(self, mock_client):
        """ Test succesful get_info call """
        # Preparing mock response ...
        mock_response = MagicMock()
        mock_response.identity_pubkey = 'test_pubkey'
        mock_response.alias = 'test_alias'
        mock_response.num_peers = 5
        mock_response.num_active_channels = 3
        mock_response.block_height = 100

        mock_client.stub.GetInfo.return_value = mock_response

        info = mock_client.get_info()

        assert info['pubkey'] == 'test_pubkey'
        assert info['alias'] == 'test_alias'
        assert info['num_peers'] == 5
        assert info['num_active_channels'] == 3
        assert info['blockheight'] == 100

    def test_get_info_failure(self, mock_client):
        """ Test get_info with RPC failure """
        mock_client.stub.GetInfo.side_effect = grpc.RpcError('RPC Error')

        with pytest.raises(Exception) as exc_info:
            mock_client.get_info()
        assert 'Error fetching node info' in str(exc_info.value)

    def test_get_channel_balances_success(self, mock_client):
        """ Test successful get_channel_balances call """
        # Preparing the mock channel ...
        mock_channel = MagicMock()
        mock_channel.chan_id = '123456'
        mock_channel.capacity = 1000000
        mock_channel.local_balance = 500000
        mock_channel.remote_balance = 500000
        mock_channel.remote_pubkey = 'remote_pubkey'

        # Preparing the mock response ...
        mock_response = MagicMock()
        mock_response.channels = [mock_channel]

        mock_client.stub.ListChannels.return_value = mock_response

        channels = mock_client.get_channel_balances()

        assert len(channels) == 1
        assert channels[0]['channel_id'] == '123456'
        assert channels[0]['capacity'] == 1000000
        assert channels[0]['local_balance'] == 500000
        assert channels[0]['remote_balance'] == 500000
        assert channels[0]['remote_pubkey'] == 'remote_pubkey'
