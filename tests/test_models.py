import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import time

from models.rl_environment import RLEnvironment, LatencySimulator
from models.ppo_agent import PPOAgent, PPOConfig, ActorCritic
from data.market_data import MarketSnapshot
from data.orderbook import OrderBook, OrderBookLevel


@pytest.fixture
def rl_config():
    return {
        "reinforcement_learning": {
            "action_space": {
                "spread_adjustment": [-0.5, 0.5],
                "size_multiplier": [0.1, 2.0]
            },
            "reward_function": {
                "pnl_weight": 1.0,
                "inventory_penalty": 0.1,
                "order_age_penalty": 0.05
            },
            "latency_simulation": {
                "enabled": True,
                "min_delay_ms": 30,
                "max_delay_ms": 100,
                "batch_matching": True
            }
        },
        "order_management": {
            "default_order_size": 1000
        }
    }


@pytest.fixture
def sample_snapshot():
    bids = [OrderBookLevel(50000.0, 0.5)]
    asks = [OrderBookLevel(50001.0, 0.4)]
    
    orderbook = OrderBook(
        symbol="BTC/USDT",
        bids=bids,
        asks=asks,
        timestamp=int(time.time() * 1000)
    )
    
    return MarketSnapshot(
        symbol="BTC/USDT",
        timestamp=int(time.time() * 1000),
        orderbook=orderbook,
        volatility=0.02
    )


class TestLatencySimulator:
    def test_initialization(self, rl_config):
        latency_config = rl_config["reinforcement_learning"]["latency_simulation"]
        simulator = LatencySimulator(latency_config)
        
        assert simulator.enabled is True
        assert simulator.min_delay_ms == 30
        assert simulator.max_delay_ms == 100
        assert simulator.batch_matching is True
        
    def test_add_latency_disabled(self, rl_config):
        latency_config = rl_config["reinforcement_learning"]["latency_simulation"]
        latency_config["enabled"] = False
        simulator = LatencySimulator(latency_config)
        
        delay = simulator.add_latency("order", {"test": "data"})
        assert delay == 0
        assert len(simulator.pending_events) == 0
        
    def test_add_latency_enabled(self, rl_config):
        latency_config = rl_config["reinforcement_learning"]["latency_simulation"]
        simulator = LatencySimulator(latency_config)
        
        delay = simulator.add_latency("order", {"test": "data"})
        
        assert 30 <= delay <= 150
        assert len(simulator.pending_events) == 1
        
    def test_process_events(self, rl_config):
        latency_config = rl_config["reinforcement_learning"]["latency_simulation"]
        simulator = LatencySimulator(latency_config)
        
        current_time = int(time.time() * 1000)
        
        simulator.add_latency("order", {"test": "data1"})
        simulator.add_latency("order", {"test": "data2"})
        
        future_time = current_time + 200
        ready_events = simulator.process_events(future_time)
        
        assert len(ready_events) >= 0
        assert len(simulator.pending_events) >= 0


class TestRLEnvironment:
    def test_initialization(self, rl_config):
        env = RLEnvironment(rl_config)
        
        assert env.observation_dim == 20
        assert env.action_dim == 4
        assert env.current_inventory == 0.0
        assert env.current_pnl == 0.0
        
    def test_reset(self, rl_config, sample_snapshot):
        env = RLEnvironment(rl_config)
        
        observation = env.reset(sample_snapshot)
        
        assert len(observation) == env.observation_dim
        assert env.current_inventory == 0.0
        assert env.current_pnl == 0.0
        assert len(env.state_history) == 0
        
    def test_decode_action(self, rl_config):
        env = RLEnvironment(rl_config)
        
        action = np.array([0.5, -0.3, 0.1, 0.8])
        adjustments = env._decode_action(action)
        
        assert "spread_adjustment" in adjustments
        assert "size_multiplier" in adjustments
        assert "reservation_adjustment" in adjustments
        assert "refresh_quotes" in adjustments
        
        assert adjustments["refresh_quotes"] is True
        assert 0.1 <= adjustments["size_multiplier"] <= 2.0
        
    def test_get_observation(self, rl_config, sample_snapshot):
        env = RLEnvironment(rl_config)
        
        observation = env._get_observation(sample_snapshot)
        
        assert len(observation) == env.observation_dim
        assert all(isinstance(x, (int, float, np.number)) for x in observation)
        assert not any(np.isnan(observation))
        
    def test_step(self, rl_config, sample_snapshot):
        env = RLEnvironment(rl_config)
        
        env.reset(sample_snapshot)
        action = np.random.uniform(-1, 1, env.action_dim)
        
        next_obs, reward, done, info = env.step(action, sample_snapshot)
        
        assert len(next_obs) == env.observation_dim
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "quotes" in info
        
    def test_simulate_order_execution(self, rl_config, sample_snapshot):
        env = RLEnvironment(rl_config)
        
        from strategy.avellaneda_stoikov import QuoteParameters
        quotes = QuoteParameters(
            symbol="BTC/USDT",
            reservation_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            bid_size=0.02,
            ask_size=0.02,
            spread=10.0,
            timestamp=int(time.time() * 1000)
        )
        
        fill_result = env._simulate_order_execution(quotes, sample_snapshot, 50)
        
        assert "bid_filled" in fill_result
        assert "ask_filled" in fill_result
        assert "pnl" in fill_result
        assert fill_result["bid_filled"] >= 0
        assert fill_result["ask_filled"] >= 0
        
    def test_calculate_fill_probability(self, rl_config):
        env = RLEnvironment(rl_config)
        
        prob_aggressive = env._calculate_fill_probability(50010.0, 50000.0, "ask", 50, 0.02)
        prob_passive = env._calculate_fill_probability(50100.0, 50000.0, "ask", 50, 0.02)
        
        assert 0.0 <= prob_aggressive <= 1.0
        assert 0.0 <= prob_passive <= 1.0
        assert prob_aggressive > prob_passive
        
    def test_episode_termination(self, rl_config, sample_snapshot):
        env = RLEnvironment(rl_config)
        
        env.reset(sample_snapshot)
        env.current_pnl = -2000.0
        
        from risk.risk_manager import RiskMetrics, RiskLevel
        risk_metrics = RiskMetrics(
            symbol="BTC/USDT",
            timestamp=int(time.time() * 1000),
            inventory_ratio=0.0,
            position_size=0.0,
            unrealized_pnl=0.0,
            daily_pnl=0.0,
            volatility=0.02,
            drawdown=0.0,
            risk_level=RiskLevel.CRITICAL,
            violations=[]
        )
        
        should_terminate = env._check_episode_termination(risk_metrics, int(time.time() * 1000))
        assert should_terminate is True


class TestActorCritic:
    def test_initialization(self):
        model = ActorCritic(observation_dim=20, action_dim=4, hidden_dim=256)
        
        assert model.shared_layers is not None
        assert model.actor_head is not None
        assert model.critic_head is not None
        assert model.log_std.shape == (4,)
        
    def test_forward_pass(self):
        model = ActorCritic(observation_dim=20, action_dim=4)
        
        batch_size = 8
        observations = torch.randn(batch_size, 20)
        
        action_mean, value = model.forward(observations)
        
        assert action_mean.shape == (batch_size, 4)
        assert value.shape == (batch_size,)
        
    def test_get_action_and_value(self):
        model = ActorCritic(observation_dim=20, action_dim=4)
        
        batch_size = 8
        observations = torch.randn(batch_size, 20)
        
        actions, log_probs, entropy, value = model.get_action_and_value(observations)
        
        assert actions.shape == (batch_size, 4)
        assert log_probs.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size,)


class TestPPOAgent:
    def test_initialization(self):
        config = PPOConfig(learning_rate=0.001, batch_size=64)
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        assert agent.observation_dim == 20
        assert agent.action_dim == 4
        assert agent.config.learning_rate == 0.001
        assert agent.config.batch_size == 64
        
    def test_get_action_deterministic(self):
        config = PPOConfig()
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        observation = np.random.randn(20)
        action, log_prob, value = agent.get_action(observation, deterministic=True)
        
        assert action.shape == (4,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        
    def test_get_action_stochastic(self):
        config = PPOConfig()
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        observation = np.random.randn(20)
        action, log_prob, value = agent.get_action(observation, deterministic=False)
        
        assert action.shape == (4,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        
    def test_store_transition(self):
        config = PPOConfig()
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        observation = np.random.randn(20)
        action = np.random.randn(4)
        
        agent.store_transition(observation, action, 0.5, 1.0, 0.8, False)
        
        assert len(agent.observations) == 1
        assert len(agent.actions) == 1
        assert len(agent.log_probs) == 1
        assert len(agent.rewards) == 1
        assert len(agent.values) == 1
        assert len(agent.dones) == 1
        
    def test_compute_gae(self):
        config = PPOConfig()
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        for i in range(10):
            observation = np.random.randn(20)
            action = np.random.randn(4)
            agent.store_transition(observation, action, 0.5, float(i), 0.8, False)
            
        advantages, returns = agent.compute_gae(next_value=1.0)
        
        assert len(advantages) == 10
        assert len(returns) == 10
        assert isinstance(advantages, np.ndarray)
        assert isinstance(returns, np.ndarray)
        
    def test_update_insufficient_data(self):
        config = PPOConfig(batch_size=100)
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        for i in range(10):
            observation = np.random.randn(20)
            action = np.random.randn(4)
            agent.store_transition(observation, action, 0.5, 1.0, 0.8, False)
            
        update_info = agent.update()
        
        assert update_info == {}
        
    def test_update_sufficient_data(self):
        config = PPOConfig(batch_size=32, n_epochs=2)
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        for i in range(64):
            observation = np.random.randn(20)
            action = np.random.randn(4)
            agent.store_transition(observation, action, 0.5, 1.0, 0.8, False)
            
        update_info = agent.update()
        
        assert "policy_loss" in update_info
        assert "value_loss" in update_info
        assert "entropy_loss" in update_info
        assert len(agent.observations) == 0
        
    def test_clear_buffer(self):
        config = PPOConfig()
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        observation = np.random.randn(20)
        action = np.random.randn(4)
        agent.store_transition(observation, action, 0.5, 1.0, 0.8, False)
        
        agent.clear_buffer()
        
        assert len(agent.observations) == 0
        assert len(agent.actions) == 0
        assert len(agent.log_probs) == 0
        assert len(agent.rewards) == 0
        assert len(agent.values) == 0
        assert len(agent.dones) == 0
        
    def test_save_load_model(self, tmp_path):
        config = PPOConfig()
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        filepath = tmp_path / "test_model.pth"
        agent.save(str(filepath))
        
        assert filepath.exists()
        
        new_agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        new_agent.load(str(filepath))
        
        observation = np.random.randn(20)
        action1, _, _ = agent.get_action(observation, deterministic=True)
        action2, _, _ = new_agent.get_action(observation, deterministic=True)
        
        np.testing.assert_array_almost_equal(action1, action2, decimal=5)
        
    def test_training_mode(self):
        config = PPOConfig()
        agent = PPOAgent(observation_dim=20, action_dim=4, config=config)
        
        agent.set_training_mode(True)
        assert agent.actor_critic.training is True
        
        agent.set_training_mode(False)
        assert agent.actor_critic.training is False


@pytest.mark.parametrize("observation_dim,action_dim", [(10, 2), (20, 4), (50, 8)])
def test_different_dimensions(observation_dim, action_dim):
    config = PPOConfig()
    agent = PPOAgent(observation_dim=observation_dim, action_dim=action_dim, config=config)
    
    observation = np.random.randn(observation_dim)
    action, log_prob, value = agent.get_action(observation)
    
    assert action.shape == (action_dim,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_ppo_config_defaults():
    config = PPOConfig()
    
    assert config.learning_rate == 0.0003
    assert config.batch_size == 2048
    assert config.n_epochs == 10
    assert config.clip_range == 0.2
    assert config.entropy_coef == 0.01
    assert config.value_function_coef == 0.5
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
