import os
import time
from src.utils.lnd_client import LndClient

def verify_node_connection(node_name, config):
    """Verify we can connect to an existing node"""
    try:
        # Check if files exist first
        cert_path = config['nodes'][node_name]['tls_cert_path']
        macaroon_path = config['nodes'][node_name]['macaroon_path']
        
        if not os.path.exists(cert_path):
            print(f"TLS cert not found at: {cert_path}")
            return False
            
        if not os.path.exists(macaroon_path):
            print(f"Macaroon not found at: {macaroon_path}")
            return False
            
        # Try connecting
        client = LndClient(node_name)
        info = client.get_info()
        print(f"Successfully connected to {node_name}: {info['alias']}")
        return True
    except Exception as e:
        print(f"Failed to connect to {node_name}: {e}")
        return False

def init_nodes():
    """Verify connection to existing nodes"""
    print("Verifying connections to existing nodes...")
    
    from src.utils.config import load_config
    config = load_config("configs/nodes.yaml")
    
    # Only check Alice and Bob for now
    nodes = ['alice', 'bob']  # Removed 'carol' until her node is ready
    all_connected = True
    
    for node in nodes:
        if not verify_node_connection(node, config):
            all_connected = False
    
    if all_connected:
        print("\nSuccessfully connected to all nodes!")
        print("Ready to start monitoring network activity.")
    
    return all_connected

if __name__ == "__main__":
    init_nodes() 