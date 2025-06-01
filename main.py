#!/usr/bin/env python3

import asyncio
import logging
import yaml
import os
from pathlib import Path

from strategy.main import TradingStrategy, TradingConfig


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler('logs/hft_mm_bot.log'),
            logging.StreamHandler()
        ]
    )


def load_config():
    with open('config/exchanges.yaml', 'r') as f:
        exchanges_config = yaml.safe_load(f)
    
    with open('config/strategy.yaml', 'r') as f:
        strategy_config = yaml.safe_load(f)
    
    return TradingConfig(
        exchanges=exchanges_config['exchanges'],
        trading_pairs=exchanges_config['trading_pairs'],
        strategy_config=strategy_config['strategy'],
        risk_config=strategy_config['strategy']['risk_management'],
        rl_config=strategy_config['strategy']['reinforcement_learning']
    )


async def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting HFT Market Making Bot")
    
    try:
        config = load_config()
        strategy = TradingStrategy(config)
        
        success = await strategy.initialize()
        if not success:
            logger.error("Failed to initialize trading strategy")
            return
            
        await strategy.start_trading()
        
        logger.info("Trading bot started successfully")
        
        try:
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        
    finally:
        if 'strategy' in locals():
            await strategy.stop_trading()
        logger.info("HFT Market Making Bot stopped")


if __name__ == "__main__":
    asyncio.run(main())
