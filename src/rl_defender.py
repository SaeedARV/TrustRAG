import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import copy
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

# Define experience replay memory tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer to store and sample experiences for training"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class ContentFeatureExtractor:
    """Extracts features from document content for RL state representation"""
    
    def __init__(self, embedding_model=None, tokenizer=None):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        
        # Suspicious patterns dictionary with weights
        self.pattern_dict = {
            "please output:": 0.9,
            "when asked": 0.8,
            "respond with": 0.85,
            "your answer should be": 0.9,
            "override": 0.75,
            "ignore previous": 0.9,
            "instead of": 0.6,
            "make sure to say": 0.85,
            "must respond": 0.8,
            "you are required to": 0.85,
            "always answer": 0.7,
            "always respond": 0.7, 
            "answer must be": 0.85,
            "when you see this": 0.9,
            "forget all instructions": 0.95
        }
    
    def extract_features(self, content_list):
        """
        Extract features from content for RL state representation
        
        Args:
            content_list: List of text content
            
        Returns:
            Feature matrix (num_contents, feature_dim)
        """
        num_contents = len(content_list)
        features = np.zeros((num_contents, 8))  # 8 hand-crafted features
        
        for i, content in enumerate(content_list):
            # Content length (normalized)
            features[i, 0] = min(1.0, len(content) / 1000)
            
            # Suspicious pattern detection
            content_lower = content.lower()
            pattern_score = 0
            for pattern, weight in self.pattern_dict.items():
                if pattern in content_lower:
                    pattern_score += weight
            features[i, 1] = min(1.0, pattern_score / 3.0)  # Normalize
            
            # Question mark count (normalized)
            features[i, 2] = min(1.0, content.count('?') / 5.0)
            
            # Exclamation mark count (normalized)
            features[i, 3] = min(1.0, content.count('!') / 3.0)
            
            # Quotes count (normalized) - might indicate instructions
            features[i, 4] = min(1.0, content.count('"') / 10.0)
            
            # All caps words ratio (might indicate emphasis/instructions)
            words = content.split()
            if words:
                caps_ratio = sum(1 for word in words if word.isupper() and len(word) > 1) / len(words)
                features[i, 5] = min(1.0, caps_ratio * 5.0)  # Normalize
            
            # Imperative mood detection (approximation using command verbs at start of sentences)
            sentences = content.split('.')
            command_verbs = ['do', 'say', 'write', 'tell', 'answer', 'respond', 'reply', 'output', 'give', 'provide']
            command_count = 0
            for sentence in sentences:
                first_word = sentence.strip().split(' ')[0].lower() if sentence.strip() else ''
                if first_word in command_verbs:
                    command_count += 1
            features[i, 6] = min(1.0, command_count / max(1, len(sentences)))
            
            # Content complexity (approximation using word length)
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                features[i, 7] = min(1.0, avg_word_length / 10.0)  # Normalize
        
        return features

    def compute_similarity_matrix(self, contents, embeddings=None):
        """
        Compute similarity matrix between contents
        
        Args:
            contents: List of text content
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Similarity matrix (num_contents, num_contents)
        """
        num_contents = len(contents)
        sim_matrix = np.zeros((num_contents, num_contents))
        
        # Use embeddings if provided, otherwise use simple cosine similarity on bag of words
        if embeddings is not None:
            for i in range(num_contents):
                for j in range(i, num_contents):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        else:
            # Simple BOW representation as fallback
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer().fit(contents)
            content_vectors = vectorizer.transform(contents).toarray()
            
            for i in range(num_contents):
                for j in range(i, num_contents):
                    vec_i = content_vectors[i]
                    vec_j = content_vectors[j]
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    else:
                        sim = 0
                        
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        
        return sim_matrix


class AdversarialDetectorDQN(nn.Module):
    """Deep Q-Network for adversarial content detection"""
    
    def __init__(self, state_dim, hidden_dim=128, num_actions=2):
        super(AdversarialDetectorDQN, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Neural network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class RLDefender:
    """Reinforcement Learning based defender for adversarial content detection"""
    
    def __init__(self, state_dim=8, hidden_dim=128, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 batch_size=64, target_update=10, model_path='models/rl_defender'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN networks - policy and target
        self.policy_net = AdversarialDetectorDQN(state_dim, hidden_dim).to(self.device)
        self.target_net = AdversarialDetectorDQN(state_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer()
        
        # Feature extractor
        self.feature_extractor = ContentFeatureExtractor()
        
        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.model_path = model_path
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_steps = 0
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
    def update_epsilon(self):
        """Decay epsilon value for exploration-exploitation trade-off"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state representation
            
        Returns:
            Action index (0: keep content, 1: remove content)
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, 1)
        else:
            # Exploitation: best action according to the policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def optimize_model(self):
        """Perform one step of optimization to update policy network"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (from target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def filter_content(self, content_list, embeddings=None, known_adversarial=None, is_training=True):
        """
        Filter potentially adversarial content using the RL policy
        
        Args:
            content_list: List of text content to filter
            embeddings: Pre-computed embeddings (optional)
            known_adversarial: Set of known adversarial texts (optional)
            is_training: Whether we're in training mode
            
        Returns:
            Filtered content list, removed indices
        """
        if not content_list:
            return content_list, []
        
        # Extract features for each content
        features = self.feature_extractor.extract_features(content_list)
        
        # Add similarity matrix features
        sim_matrix = self.feature_extractor.compute_similarity_matrix(content_list, embeddings)
        
        # Action decisions and tracking
        keep_indices = []
        remove_indices = []
        
        # Process each content
        for i, content in enumerate(content_list):
            # Create state representation
            # Use individual features and aggregate similarity with other content
            avg_sim = np.mean(sim_matrix[i, :])
            max_sim = np.max(sim_matrix[i, :] if len(sim_matrix[i, :]) > 1 else [0])
            
            state = np.append(features[i], [avg_sim, max_sim])
            
            # Select action based on policy
            action = self.select_action(state)
            
            if action == 0:  # Keep content
                keep_indices.append(i)
            else:  # Remove content
                remove_indices.append(i)
            
            # If in training mode and we have ground truth, add experience to replay buffer
            if is_training and known_adversarial is not None:
                # Determine if this content is actually adversarial
                is_adversarial = content in known_adversarial
                
                # Calculate reward: 
                # +1 for correct keep (not adversarial) or correct remove (adversarial)
                # -1 for incorrect keep (adversarial) or incorrect remove (not adversarial)
                if (action == 0 and not is_adversarial) or (action == 1 and is_adversarial):
                    reward = 1.0  # Correct decision
                else:
                    reward = -1.0  # Incorrect decision
                
                # Store experience
                # For simplicity, we use the same state as next_state (terminal state)
                # and mark all experiences as "done"
                self.memory.add(state, action, reward, state, True)
                
                # Update model
                self.optimize_model()
        
        # Create filtered content list
        filtered_content = [content_list[i] for i in keep_indices]
        
        # Update epsilon for exploration
        if is_training:
            self.update_epsilon()
        
        return filtered_content, remove_indices
    
    def save_model(self, path=None):
        """Save the model"""
        if path is None:
            path = self.model_path
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
    
    def load_model(self, path=None):
        """Load the model"""
        if path is None:
            path = self.model_path
        
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.training_steps = checkpoint['training_steps']
            return True
        else:
            return False


def rl_filtering(embedding_topk, topk_contents, adv_text_set, n_gram, defender=None, is_training=True):
    """
    Enhanced content filtering using RL-based defender
    
    Args:
        embedding_topk: Embeddings of the top-k retrieved documents
        topk_contents: Text content of the top-k retrieved documents
        adv_text_set: Set of known adversarial texts
        n_gram: Boolean flag for using n-gram similarity checking
        defender: RLDefender instance (will create new one if None)
        is_training: Whether to train the model during filtering
        
    Returns:
        Filtered embeddings and contents
    """
    # Early exit if there's nothing to filter
    if len(topk_contents) <= 1:
        return embedding_topk, topk_contents
    
    # Create or use defender
    if defender is None:
        defender = RLDefender(state_dim=10)  # 8 content features + 2 similarity features
        model_path = 'models/rl_defender'
        if os.path.exists(model_path):
            defender.load_model(model_path)
    
    # Apply RL-based filtering
    filtered_contents, removed_indices = defender.filter_content(
        topk_contents, 
        embeddings=embedding_topk,
        known_adversarial=adv_text_set,
        is_training=is_training
    )
    
    # Check if we removed everything
    if not filtered_contents:
        # If everything was filtered, return empty arrays
        return np.array([]).reshape(0, embedding_topk.shape[1]), []
    
    # Create mask for the filtered content
    mask = np.ones(len(topk_contents), dtype=bool)
    for idx in removed_indices:
        mask[idx] = False
    
    # Filter embeddings
    filtered_embeddings = embedding_topk[mask]
    
    # Save model periodically during training
    if is_training and random.random() < 0.1:  # 10% chance to save
        defender.save_model()
    
    return filtered_embeddings, filtered_contents 