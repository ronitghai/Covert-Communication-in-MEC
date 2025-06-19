import numpy as np
import random
from collections import deque
import copy
import matplotlib.pyplot as plt
import seaborn as sns  # For better Q-Table visualization
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# Define the Eavesdropper Class
class Eavesdropper:
    def __init__(self, detection_range, detection_probability, position=(0, 0)):
        self.detection_range = detection_range
        self.detection_probability = detection_probability
        self.position = position  # (x, y) coordinates

    def detect(self, transmitter_position, action):
        """
        Determines if the eavesdropper detects the transmission based on distance and action.
        """
        distance = math.sqrt((self.position[0] - transmitter_position[0])**2 + (self.position[1] - transmitter_position[1])**2)
        if distance <= self.detection_range:
            # Detection probability increases with action's detection risk
            effective_detection_prob = self.detection_probability * DETECTION_RISKS[action]
            return np.random.rand() < effective_detection_prob
        return False

# Define the Agent Class
class Agent:
    def __init__(self, agent_id, state_size, action_size, buffer_capacity, alpha, p):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.p = p  # Probability threshold for random action
        self.alpha = alpha  # Learning rate
        
        # Initialize Q-Table and target Q-Table
        self.q_table = np.zeros((state_size, action_size))
        self.target_q_table = copy.deepcopy(self.q_table)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    def select_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.
        """
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, batch, gamma):
        """
        Update Q-Table using the batch of experiences.
        """
        for state, action, reward, next_state, done in batch:
            best_next_action = np.argmax(self.target_q_table[next_state])
            target = reward + gamma * self.target_q_table[next_state][best_next_action] * (not done)
            self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])
    
    def soft_update_target(self, tau):
        """
        Soft update target Q-Table.
        """
        self.target_q_table = tau * self.q_table + (1 - tau) * self.target_q_table

# Define the Environment
class Environment:
    def __init__(self, num_agents, num_states, num_actions, eavesdroppers):
        self.num_agents = num_agents
        self.num_states = num_states  # Represents AoI range from 0 to num_states - 1
        self.num_actions = num_actions
        self.eavesdroppers = eavesdroppers  # List of Eavesdropper instances
        self.reset()
    
    def reset(self):
        # Initialize states for all agents
        # Each state is a tuple: (AoI, Energy, Position)
        # Position is randomly assigned within a reasonable range
        self.states = []
        for _ in range(self.num_agents):
            aoi = 0
            energy = 100
            # Assign positions within detection range of at least one eavesdropper
            # For simplicity, place agents near the first eavesdropper
            eavesdropper = self.eavesdroppers[0]
            position = (
                eavesdropper.position[0] + random.uniform(-eavesdropper.detection_range / 2, eavesdropper.detection_range / 2),
                eavesdropper.position[1] + random.uniform(-eavesdropper.detection_range / 2, eavesdropper.detection_range / 2)
            )
            self.states.append((aoi, energy, position))
        return self.states
    
    def step(self, actions):
        """
        Simulate environment dynamics.
        Returns:
            next_states: List of updated states for each agent
            rewards: List of rewards for each agent
            done: Boolean indicating if the episode is done
            transmission_status: List indicating if each agent is transmitting
            eavesdropper_detections: List indicating if eavesdroppers detected the transmission
        """
        next_states = []
        rewards = []
        done_flags = []
        transmission_status = []  # List indicating if each agent is transmitting
        eavesdropper_detections = []  # List indicating if eavesdroppers detected the transmission
        
        for i, action in enumerate(actions):
            aoip, energy, position = self.states[i]
            is_transmitting = action in [0, 1, 2]
            transmission_status.append(is_transmitting)
            
            # Update AoI and Energy based on action
            aoip += 1  # AoI increases if not reset
            energy -= POWER_LEVELS[action] * uav_params[i]["max_power"]
            energy = max(energy, 0)  # Prevent negative energy
            
            # Transmission Success
            if is_transmitting:
                transmission_success = np.random.rand() < 0.9  # 90% success rate
                if transmission_success:
                    aoip = 0
            else:
                transmission_success = False  # No transmission
            
            # Calculate detection probability with added noise
            base_detection = DETECTION_RISKS[action]
            noise = np.random.normal(0, 0.5)  # Gaussian noise
            detection_probability = max(base_detection + noise, 0)  # Ensure non-negative
            
            # Calculate reward
            reward = -(W_AOI * aoip) - (W_DETECTION * detection_probability) - (W_ENERGY * (POWER_LEVELS[action] * uav_params[i]["max_power"]))
            if is_transmitting:
                if transmission_success:
                    reward += W_TRANSMISSION - C_TRANSMISSION
                else:
                    reward -= C_TRANSMISSION  # Penalize failed transmission
            
            # Additional penalties
            if action == 4:  # Stealth Mode
                reward -= P_STEALTH_OVERUSE
            if action == 3:  # Delay Transmission
                reward -= P_DELAY_OVERUSE
            
            # Check termination condition for individual agent
            done = aoip > 8 or energy <= 0
            done_flags.append(done)
            
            # Check for eavesdropper detections
            detected = False
            if is_transmitting:
                for eavesdropper in self.eavesdroppers:
                    if eavesdropper.detect(transmitter_position=position, action=action):
                        detected = True
                        break  # Stop after first detection
            eavesdropper_detections.append(detected)
            
            # Apply eavesdropper detection penalty
            if detected:
                reward -= P_EAVESDROP_DETECTION
            
            # Update state
            next_states.append((aoip, energy, position))
            rewards.append(reward)
        
        self.states = next_states
        done = False  # Episodes run for a fixed number of steps
        return next_states, rewards, done, transmission_status, eavesdropper_detections

# Hyperparameters
ALPHA = 0.05  # learning rate for stability
GAMMA = 0.9
EPSILON_INITIAL = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05  # Maintains some exploration
MAX_EPOCH = 5000
MAX_STEPS_PER_EPISODE = 50
B = 1000  # Replay buffer capacity
Tu = 100  # Target network update frequency
Ef = 100  # Federated update frequency
tau = 0.05  # Soft update parameter for gradual updates
batch_size = 32  # Experience replay batch size

# System Parameters
num_uavs = 3
num_willies = 2  # Number of eavesdroppers
num_states = 100  # Represents AoI range from 0 to 99
num_actions = 5

# Probability threshold for random action selection
p = 0.2  # Example threshold for action selection

# Define power levels and detection risks corresponding to actions
POWER_LEVELS = [0.1, 0.5, 1.0, 0.05, 0.05]  # Power levels for actions 0-4
DETECTION_RISKS = [1, 5, 7, 1, 0]  # Detection risks for actions 0-4

# Define additional reward components
W_AOI = 0.5
W_DETECTION = 0.8
W_ENERGY = 0.1
W_TRANSMISSION = 2.5      # Increase reward for success
C_TRANSMISSION = 2.0       # Increase cost to reduce reckless attempts
P_STEALTH_OVERUSE = 3.0    # Allow strategic stealth usage
P_DELAY_OVERUSE = 3.0      # Allow strategic delay
P_EAVESDROP_DETECTION = 20.0 # Increase penalty to discourage risky transmissions


# Heterogeneous UAV parameters
uav_params = [{
    "max_power": np.random.uniform(0.1, 1.0), 
    "detection_sensitivity": np.random.uniform(0.1, 0.5)
} for _ in range(num_uavs)]

# Initialize Eavesdroppers
# Positions are set such that agents are within detection ranges
eavesdroppers = [
    Eavesdropper(detection_range=50, detection_probability=0.3, position=(50, 50)),
    Eavesdropper(detection_range=50, detection_probability=0.3, position=(150, 150))
]

# Initialize Environment with eavesdroppers
env = Environment(num_agents=num_uavs, num_states=num_states, num_actions=num_actions, eavesdroppers=eavesdroppers)

# Initialize Agents
agents = [Agent(agent_id=k, state_size=num_states, action_size=num_actions, buffer_capacity=B, alpha=ALPHA, p=p) for k in range(num_uavs)]

# Initialize Epsilon for each agent
epsilons = [EPSILON_INITIAL for _ in range(num_uavs)]

# Federated Averaging Function
def federated_averaging(agents):
    """
    Perform federated averaging of Q-Tables across agents.
    """
    avg_q_table = np.mean([agent.q_table for agent in agents], axis=0)
    for agent in agents:
        agent.q_table = copy.deepcopy(avg_q_table)

# Metrics Tracking Function
def track_metrics(episode, metrics, successful_transmissions, total_transmissions, detections):
    """
    Track AoI, detection probability, energy consumption, and transmission events.
    """
    if total_transmissions > 0:
        willie_detection_percentage = (detections / total_transmissions) * 100
        successful_transmission_percentage = (successful_transmissions / total_transmissions) * 100
    else:
        # Avoid division by zero
        willie_detection_percentage = 0.0
        successful_transmission_percentage = 0.0


    logging.info(f"Episode {episode} - Avg AoI: {metrics['avg_aoi']:.2f}, "
                 f"Detection Risk: {metrics['avg_detection']:.2f}, "
                 f"Energy Usage: {metrics['avg_energy']:.2f}, "
                 f"Successful Transmissions: {successful_transmissions}, "
                 f"({successful_transmission_percentage:.2f}%), "
                 f"Total Transmissions: {total_transmissions}, "
                 f"Willie Detections: {detections}"
                 f"({willie_detection_percentage:.2f}%)"
                 )

# Initialize lists to store per timeslot metrics
aoi_per_timeslot = [[] for _ in range(MAX_STEPS_PER_EPISODE)]
data_volume_per_timeslot = [[] for _ in range(MAX_STEPS_PER_EPISODE)]

# Initialize lists to store overall epoch metrics
avg_aoi_per_epoch = []
avg_detection_per_epoch = []
avg_energy_per_epoch = []
successful_transmissions_per_epoch = []
detections_per_epoch = []
total_transmissions_per_epoch = []

# Training Loop
for epoch in range(1, MAX_EPOCH + 1):
    # Reset environment
    states = env.reset()
    
    # Initialize metrics
    metrics = {"total_aoi": 0, "total_detection": 0, "total_energy": 0, "steps": 0}
    successful_transmissions = 0
    detections = 0
    failed_transmissions = 0
    total_transmissions = 0  # Track total packets transmitted this epoch
    
    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        actions = []
        
        # Action Selection for each agent
        for k, agent in enumerate(agents):
            if epsilons[k] > MIN_EPSILON and (random.random() < p or agent.replay_buffer.size() < B):
                # Random action
                action = random.randint(0, num_actions - 1)
            else:
                # Policy-based action
                # Using AoI as state; ensure AoI is within state_size
                state_aoi = min(states[k][0], num_states - 1)
                action = agent.select_action(state_aoi, epsilons[k])
            actions.append(action)
        
        # Interact with environment
        next_states, rewards, _, transmission_status, eavesdropper_detections = env.step(actions)

        for k, action in enumerate(actions):
            if action in [0, 1, 2]:
                total_transmissions += 1
        
        # Add experiences to replay buffers
        for k, agent in enumerate(agents):
            state = states[k][0]  # Using AoI as state
            action = actions[k]
            reward = rewards[k]
            next_state = next_states[k][0]  # Next AoI
            done = False  # Episodes run for a fixed number of steps
            agent.replay_buffer.add((state, action, reward, next_state, done))
        
        # Update metrics based on actual state transitions
        avg_aoi_step = 0
        data_volume_step = 0
        for k, agent in enumerate(agents):
            aoip, energy, position = states[k]
            action = actions[k]
            
            metrics["total_aoi"] += aoip
            detection_risk = DETECTION_RISKS[action] * uav_params[k]["detection_sensitivity"]
            energy_cost = POWER_LEVELS[action] * uav_params[k]["max_power"]
            
            metrics["total_detection"] += detection_risk
            metrics["total_energy"] += energy_cost
            metrics["steps"] += 1
            
            # Calculate average AoI for this step
            avg_aoi_step += aoip
            
            # Calculate received data volume
            if action in [0, 1, 2]:  # Actions that involve data transmission
                # Check if transmission was successful (from environment)
                transmission_success = transmission_status[k] and (states[k][0] > next_states[k][0])
                if transmission_success:
                    data_volume_step += POWER_LEVELS[action]
                    successful_transmissions += 1
                else:
                    failed_transmissions += 1  # Track failed transmissions
            
            # Track eavesdropper detections
            if eavesdropper_detections[k]:
                detections += 1
        
        avg_aoi_step /= num_uavs  # Average AoI across agents for this step
        aoi_per_timeslot[step - 1].append(avg_aoi_step)
        data_volume_per_timeslot[step - 1].append(data_volume_step)
        
        # Learning step
        for k, agent in enumerate(agents):
            if agent.replay_buffer.size() >= batch_size:
                batch = agent.replay_buffer.sample(batch_size)
                agent.update_q_table(batch, GAMMA)
        
        # Update states
        states = next_states
    # Decay epsilon
    for k in range(num_uavs):
        epsilons[k] = max(MIN_EPSILON, epsilons[k] * EPSILON_DECAY)
    
    # Update target networks periodically
    if epoch % Tu == 0:
        for agent in agents:
            agent.soft_update_target(tau)
    
    # Federated Learning updates periodically
    if epoch % Ef == 0:
        federated_averaging(agents)
    
    # Calculate and store average metrics
    if metrics["steps"] > 0:
        avg_aoi = metrics["total_aoi"] / metrics["steps"]
        avg_detection = metrics["total_detection"] / metrics["steps"]
        avg_energy = metrics["total_energy"] / metrics["steps"]
    else:
        avg_aoi = 0
        avg_detection = 0
        avg_energy = 0
    
    avg_aoi_per_epoch.append(avg_aoi)
    avg_detection_per_epoch.append(avg_detection)
    avg_energy_per_epoch.append(avg_energy)
    successful_transmissions_per_epoch.append(successful_transmissions)
    detections_per_epoch.append(detections)
    total_transmissions_per_epoch.append(total_transmissions)

    # Display progress
    if epoch % 100 == 0 or epoch == 1:
        track_metrics(epoch, {
            "avg_aoi": avg_aoi,
            "avg_detection": avg_detection,
            "avg_energy": avg_energy
        }, successful_transmissions, total_transmissions, detections)

# Calculate average AoI and data volume per timeslot across all episodes
avg_aoi_timeslot = [np.mean(aoi_per_timeslot[step]) if aoi_per_timeslot[step] else 0 for step in range(MAX_STEPS_PER_EPISODE)]
avg_data_volume_timeslot = [np.mean(data_volume_per_timeslot[step]) if data_volume_per_timeslot[step] else 0 for step in range(MAX_STEPS_PER_EPISODE)]

# Plot Average AoI vs Timeslot
plt.figure(figsize=(12, 6))
plt.plot(range(1, MAX_STEPS_PER_EPISODE + 1), avg_aoi_timeslot, label='Average AoI', color='blue')
plt.title('Average Age of Information (AoI) vs. Timeslot')
plt.xlabel('Timeslot')
plt.ylabel('Average AoI')
plt.legend()
plt.grid(True)
plt.show()

# Plot Received Data Volume vs Timeslot
plt.figure(figsize=(12, 6))
plt.plot(range(1, MAX_STEPS_PER_EPISODE + 1), avg_data_volume_timeslot, label='Received Data Volume', color='orange')
plt.title('Received Data Volume vs. Timeslot')
plt.xlabel('Timeslot')
plt.ylabel('Received Data Volume')
plt.legend()
plt.grid(True)
plt.show()

# Plot metrics over epochs
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(avg_aoi_per_epoch, label='Avg AoI', color='blue')
plt.title('Average AoI per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average AoI')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(avg_detection_per_epoch, label='Avg Detection Risk', color='orange')
plt.title('Average Detection Risk per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Detection Risk')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(avg_energy_per_epoch, label='Avg Energy Usage', color='green')
plt.title('Average Energy Usage per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Energy Usage')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot transmission-related metrics
def plot_transmission_metrics(successful_transmissions, total_transmissions, detections):
    epochs = range(1, len(successful_transmissions) + 1)
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, successful_transmissions, label='Successful Transmissions', color='green')
    plt.title('Successful Transmissions per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, total_transmissions, label='Total Transmissions', color='purple')
    plt.title('Total Transmissions per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, detections, label='Willie Detections', color='red')
    plt.title('Willie Detections per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Call the transmission metrics plot function
plot_transmission_metrics(successful_transmissions_per_epoch, total_transmissions_per_epoch, detections_per_epoch)