import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    learning_rate: float = 0.0003
    batch_size: int = 2048
    n_epochs: int = 10
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_function_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_layers(observations)
        
        action_mean = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        
        return action_mean, value.squeeze(-1)
        
    def get_action_and_value(self, observations: torch.Tensor, 
                           actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value = self.forward(observations)
        
        action_std = torch.exp(self.log_std)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        if actions is None:
            actions = action_dist.sample()
            
        log_probs = action_dist.log_prob(actions).sum(axis=-1)
        entropy = action_dist.entropy().sum(axis=-1)
        
        return actions, log_probs, entropy, value


class PPOAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: PPOConfig):
        self.config = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor_critic = ActorCritic(observation_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
        
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            if deterministic:
                action_mean, value = self.actor_critic(obs_tensor)
                action = action_mean
                log_prob = torch.zeros(1)
                entropy = torch.zeros(1)
            else:
                action, log_prob, entropy, value = self.actor_critic.get_action_and_value(obs_tensor)
                
            return action.cpu().numpy().flatten(), log_prob.item(), value.item()
            
    def store_transition(self, observation: np.ndarray, action: np.ndarray, 
                        log_prob: float, reward: float, value: float, done: bool) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
                
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            
        returns = advantages + values[:-1]
        
        return advantages, returns
        
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        if len(self.observations) < self.config.batch_size:
            return {}
            
        advantages, returns = self.compute_gae(next_value)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        observations = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        batch_size = len(observations)
        indices = np.arange(batch_size)
        
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                _, new_log_probs, entropy, values = self.actor_critic.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values, batch_returns)
                
                entropy_loss = -entropy.mean()
                
                total_loss = (policy_loss + 
                            self.config.value_function_coef * value_loss + 
                            self.config.entropy_coef * entropy_loss)
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                
        num_updates = self.config.n_epochs * (batch_size // self.config.batch_size)
        
        self.clear_buffer()
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "explained_variance": self._explained_variance(returns, returns_tensor.detach().cpu().numpy())
        }
        
    def _explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        var_y = np.var(y_true)
        return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
        
    def clear_buffer(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        
    def save(self, filepath: str) -> None:
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        
    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def set_training_mode(self, training: bool = True) -> None:
        if training:
            self.actor_critic.train()
        else:
            self.actor_critic.eval()
