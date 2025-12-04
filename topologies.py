import math

def create_ring_topology(clients):
    """
    Creates a ring topology where each client is connected to two neighbors.

    Args:
        clients (list): A list of client objects.

    Returns:
        dict: A dictionary where keys are client indices and values are lists of their neighbor client objects.
    """
    num_clients = len(clients)
    topology = {}
    if num_clients <= 1:
        return {client.idx: [] for client in clients}

    for i in range(num_clients):
        left_neighbor_idx = (i - 1 + num_clients) % num_clients
        right_neighbor_idx = (i + 1) % num_clients
        
        current_client = clients[i]
        left_neighbor = clients[left_neighbor_idx]
        right_neighbor = clients[right_neighbor_idx]
        
        topology[current_client.idx] = [left_neighbor, right_neighbor]
        
    return topology

def create_full_topology(clients):
    """
    Creates a fully connected topology where each client is connected to all other clients.

    Args:
        clients (list): A list of client objects.

    Returns:
        dict: A dictionary where keys are client indices and values are lists of their neighbor client objects.
    """
    topology = {}
    for client in clients:
        neighbors = [c for c in clients if c.idx != client.idx]
        topology[client.idx] = neighbors
    return topology

def create_star_topology(clients, center_idx=0):
    """
    Creates a star topology with a central client connected to all others.

    Args:
        clients (list): A list of client objects.
        center_idx (int): The index of the client to be the center of the star.

    Returns:
        dict: A dictionary where keys are client indices and values are lists of their neighbor client objects.
    """
    num_clients = len(clients)
    if num_clients == 0:
        return {}
    
    topology = {client.idx: [] for client in clients}
    center_client = clients[center_idx]

    for client in clients:
        if client.idx != center_client.idx:
            # Peripheral nodes connect to the center
            topology[client.idx].append(center_client)
            # Center node connects to all peripheral nodes
            topology[center_client.idx].append(client)
            
    return topology

def create_grid_topology(clients):
    """
    Creates a 2D grid topology.

    Args:
        clients (list): A list of client objects.

    Returns:
        dict: A dictionary where keys are client indices and values are lists of their neighbor client objects.
    """
    num_clients = len(clients)
    if num_clients == 0:
        return {}

    # Find the grid dimensions that are as close to a square as possible
    cols = int(math.sqrt(num_clients))
    while num_clients % cols != 0:
        cols -= 1
    rows = num_clients // cols

    client_map = {client.idx: client for client in clients}
    idx_matrix = [[clients[r * cols + c].idx for c in range(cols)] for r in range(rows)]
    
    topology = {client.idx: [] for client in clients}

    for r in range(rows):
        for c in range(cols):
            current_idx = idx_matrix[r][c]
            neighbors = []
            # Up
            if r > 0:
                neighbors.append(client_map[idx_matrix[r - 1][c]])
            # Down
            if r < rows - 1:
                neighbors.append(client_map[idx_matrix[r + 1][c]])
            # Left
            if c > 0:
                neighbors.append(client_map[idx_matrix[r][c - 1]])
            # Right
            if c < cols - 1:
                neighbors.append(client_map[idx_matrix[r][c + 1]])
            
            topology[current_idx] = neighbors
            
    return topology
