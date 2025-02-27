import utils.lnd_client as LndClient
import utils.config as config

def setup_test_network():
    # Create channels between nodes
    for connection in config['network']['topology']:
        from_node = connection['from']
        for to_node in connection['to']:
            # Open channel
            amount = connection['capacity']
            client = LndClient(from_node)
            client.open_channel(to_node, amount)
            
    # TODO: Implement channel balancing
    # balance_channels()  # Commented out until implemented 