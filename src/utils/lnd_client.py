import os
import codecs
import grpc
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from ..proto import lightning_pb2 as ln
from ..proto import lightning_pb2_grpc as lnrpc

from .config import load_config


class LndClient:
    """ ☋ Client for interacting with LND nodes ☊ """

    def __init__(self, node_name: str, config_path: str = "configs/test_config.yaml"):
        self.config = load_config(config_path)
        self.node_config = self.config['nodes'][node_name]
        self.stub = self._create_stub()

    def _create_stub(self) -> lnrpc.LightningStub:
        """ Create a gRPC stub for LND communication ⛵ """

        cert_path = os.path.expanduser(self.node_config['tls_cert_path'])
        cert = open(cert_path, 'rb').read()

        macaroon_path = os.path.expanduser(self.node_config['macaroon_path'])
        with open(macaroon_path, 'rb') as f:
            macaroon_bytes = f.read()
        macaroon = codecs.encode(macaroon_bytes, 'hex')

        cert_creds = grpc.ssl_channel_credentials(cert)
        auth_creds = grpc.metadta_call_credentials(
            lambda _, cb: cb([('macaroon', macaroon)], None)
        )
        combined_creds = grpc.composite_channel_credentials(cert_creds, auth_creds)

        channel = grpc.secure_channel(
            self.node_config['rpc_server'],
            combined_creds
        )
        return lnrpc.LightningStub(channel)
    
        def get_info(self) -> Dict:
            """ Get basic info about the node """
            response = self.stub.GetInfo(ln.GetInfoRequest())
            return {
                'pubkey': response.identity_pubkey,
                'alias': response.alias,
                'num_peers': response.num_peers,
                'num_active_channels': response.num_active_channels,
                'blockheight': response.block_height
            }
        
        def get_channel_balances(self) -> List[Dict]:
            """ Get balance information for all channels """
            response = self.stub.ListChannels(ln.ListChannelsRequest())
            channels = []
            for channel in response.channels:
                channels.append({
                    'channel_id': channel.chan_id,
                    'capacity': channel.capacity,
                    'local_balance': channel.local_balance,
                    'remote_balance': channel.remote_balance,
                    
                })
