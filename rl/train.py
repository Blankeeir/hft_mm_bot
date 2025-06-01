import yaml, argparse
import gymnasium as gym
from stable_baselines3 import PPO
from rl.env import MMEnv
from utils.logger import get_logger

log = get_logger("train")

def train(cfg_path, pair):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    sim = None  # TODO: implement backtest simulator
    env = MMEnv(sim, cfg)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_mm_" + pair)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    parser.add_argument("--pair", required=True)
    args = parser.parse_args()
    train(args.config, args.pair)
