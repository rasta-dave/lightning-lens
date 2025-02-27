import random
import time
from src.utils.lnd_client import LndClient

def simulate_network_activity():
    nodes = ['alice', 'bob', 'carol', 'dave', 'eve']
    clients = {name: LndClient(name) for name in nodes}
    
    while True:
        # Random payment between random nodes
        sender = random.choice(nodes)
        receiver = random.choice([n for n in nodes if n != sender])
        
        amount = random.randint(1000, 100000)  # Random payment size
        try:
            # Create invoice
            invoice = clients[receiver].create_invoice(amount, "Test payment")
            # Pay invoice
            clients[sender].pay_invoice(invoice)
            
            print(f"Payment: {sender} -> {receiver}: {amount} sats")
        except Exception as e:
            print(f"Payment failed: {e}")
            
        # Random delay between payments
        time.sleep(random.uniform(1, 10)) 