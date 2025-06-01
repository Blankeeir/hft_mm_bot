# HFT Market Making Bot

High-frequency market making trading bot with AI smart prediction for Coinbase International, OKX, and Binance.

## Features
- RL-augmented Avellaneda-Stoikov strategy
- Micro-price anchoring and volatility forecasting
- HAR/TimeMixer models for volatility prediction
- FIX API integration with WebSocket fallback
- Advanced risk management and position tracking

## Structure
- `connectors/` - Exchange API integrations
- `strategy/` - Trading strategy implementations  
- `data/` - Market data processing and storage
- `risk/` - Risk management modules
- `models/` - ML/RL model implementations
- `tests/` - Test suites
- `config/` - Configuration files

