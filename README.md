# High-Frequency Market Making Bot with RL-Augmented Avellaneda-Stoikov Strategy

[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen)](https://github.com/Blankeeir/hft_mm_bot)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A comprehensive high-frequency market making trading bot implementing the research-backed RL-augmented Avellaneda-Stoikov strategy with micro-price anchoring and advanced volatility forecasting for Coinbase International, OKX, and Binance exchanges.

## 🚀 Key Features

### Advanced Trading Strategy
- **RL-Augmented Avellaneda-Stoikov**: Mathematical implementation with PPO reinforcement learning
- **Micro-Price Anchoring**: Reduces adverse selection by ~12% through queue-weighted pricing
- **Order Flow Imbalance (OFI)**: Real-time market direction prediction over 1s/5s windows
- **Dynamic Risk Management**: Inventory-based position sizing with volatility scaling

### Multi-Exchange Support
- **Binance**: FIX API + WebSocket (VIP 9+ compatible)
- **Coinbase International**: Institutional API integration
- **OKX**: Institutional account support
- **Unified Interface**: Seamless multi-exchange operations

### AI-Powered Forecasting
- **HAR Model**: Daily/weekly/monthly realized volatility combinations
- **TimeMixer Model**: MLP-based multiscale volatility forecasting (1-12 days)
- **Real-time Updates**: Sub-hourly parameter refresh for dynamic adaptation

### High-Performance Architecture
- **Async/Await**: Non-blocking I/O for millisecond-level responsiveness
- **Low Latency**: Optimized for co-location and institutional trading
- **Real-time Data**: L2/L3 orderbook processing with QuestDB integration
- **Comprehensive Testing**: 142 tests covering all components

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Memory**: 8GB+ RAM recommended
- **Storage**: SSD for time-series data
- **Network**: Low-latency connection (co-location recommended)

### Exchange Account Requirements
- **Binance**: VIP 9+ account with FIX API permissions (`FIX_API` or `FIX_API_READ_ONLY`)
- **Coinbase**: International institutional account
- **OKX**: Institutional account tier with API trading permissions

## 🛠️ Installation & Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/Blankeeir/hft_mm_bot.git
cd hft_mm_bot

# Switch to the implementation branch
git checkout devin/1748778219-hft-market-making-bot
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
```

Edit `.env` with your exchange credentials:
```env
# Binance Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_FIX_ENABLED=true

# Coinbase Configuration
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET_KEY=your_coinbase_secret_key
COINBASE_PASSPHRASE=your_coinbase_passphrase

# OKX Configuration
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_PASSPHRASE=your_okx_passphrase

# Trading Configuration
TRADING_MODE=paper  # paper or live
LOG_LEVEL=INFO
```

### Step 5: Verify Installation
```bash
# Run all tests to verify setup
make test

# Should show: 142 passed, 1 skipped
```

## 🚦 Quick Start Guide

### 1. Paper Trading Mode (Recommended First)
```bash
python main.py --mode paper --config config/strategy.yaml --symbols BTC-USD,ETH-USD
```

### 2. Live Trading (Production)
⚠️ **WARNING**: Only use live mode with small amounts initially
```bash
python main.py --mode live --config config/strategy.yaml --symbols BTC-USD
```

### 3. Monitor Performance
```bash
# View real-time logs
tail -f logs/trading.log

# Check performance metrics
python -c "
from strategy.main import TradingStrategy
# Performance monitoring code here
"
```

## 📊 Configuration Guide

### Strategy Parameters (`config/strategy.yaml`)
```yaml
avellaneda_stoikov:
  risk_aversion: 0.1          # Risk aversion parameter (γ) - Higher = more conservative
  order_arrival_intensity: 1.5 # Order arrival rate (A) - Higher = tighter spreads
  time_horizon: 86400         # Trading horizon in seconds (24 hours)

order_management:
  default_order_size: 1000    # Base order size in USD
  min_spread_bps: 1          # Minimum spread (1 basis point = 0.01%)
  max_spread_bps: 100        # Maximum spread (100 basis points = 1%)
  refresh_interval_ms: 200    # How often to update quotes (milliseconds)

risk_management:
  max_inventory_usd: 50000   # Maximum position size per symbol
  max_drawdown_pct: 5.0      # Stop trading if losses exceed 5%
  volatility_threshold: 0.05  # Circuit breaker for high volatility (5%)

reinforcement_learning:
  learning_rate: 0.0003      # PPO learning rate
  batch_size: 2048          # Training batch size
  training_enabled: true     # Enable RL training mode
```

### Exchange Configuration (`config/exchanges.yaml`)
```yaml
binance:
  enabled: true
  use_fix_api: true          # Use FIX API if available
  testnet: false            # Set to true for testing
  fee_structure:
    maker_fee: 0.000075      # 0.75 bps maker fee
    taker_fee: 0.0013        # 13 bps taker fee

coinbase:
  enabled: true
  sandbox: false            # Set to true for testing
  fee_structure:
    maker_fee: -0.000007     # -0.7 bps (rebate for institutional)
    taker_fee: 0.000075      # 7.5 bps taker fee

okx:
  enabled: true
  demo: false              # Set to true for testing
  fee_structure:
    maker_fee: -0.00001      # -1 bps (rebate for institutional)
    taker_fee: 0.00015       # 15 bps taker fee
```

## 🎯 Understanding the Strategy

### Mathematical Foundation
The bot implements the Avellaneda-Stoikov model enhanced with reinforcement learning:

**Reservation Price Formula:**
```
r(t) = pm(t) - I(t) × γ × σ² × (T-t)
```

**Optimal Spread Formula:**
```
δ = ½[γσ²(T-t) + (1/γk)ln(1+kγ)]
```

Where:
- `pm(t)`: Micro-price (queue-weighted mid-price)
- `I(t)`: Current inventory position
- `γ`: Risk aversion parameter (configurable)
- `σ`: Volatility forecast from HAR/TimeMixer models
- `T-t`: Time remaining in trading session

### Key Strategy Components

1. **Micro-Price Calculation**: Uses order book depth to calculate fair value
2. **Inventory Management**: Automatically skews quotes to reduce inventory risk
3. **Volatility Forecasting**: HAR and TimeMixer models predict future volatility
4. **RL Enhancement**: PPO agent learns optimal parameter adjustments
5. **Risk Controls**: Multiple layers of risk management and circuit breakers

## 📈 Performance Monitoring

### Real-time Monitoring
```bash
# View live performance dashboard
python -m monitoring.dashboard

# Check system health
python -m monitoring.health_check

# Generate performance report
python -m monitoring.performance_report --period 24h
```

### Key Metrics to Watch
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.5)
- **Fill Rate**: Percentage of orders filled (target: >80%)
- **Inventory Turnover**: How quickly positions are managed
- **Spread Capture**: Percentage of bid-ask spread captured
- **Adverse Selection**: Losses from informed traders

## 🛡️ Risk Management Features

### Automatic Risk Controls
- **Position Limits**: Maximum inventory per symbol
- **Drawdown Protection**: Automatic trading halt on losses
- **Volatility Circuit Breakers**: Stop trading during market stress
- **Order Age Limits**: Cancel stale quotes automatically
- **Market Manipulation Detection**: Identify suspicious patterns

### Manual Risk Controls
```python
# Emergency stop all trading
python -c "
from strategy.main import TradingStrategy
strategy = TradingStrategy(config)
strategy.emergency_stop()
"

# Check current positions
python -c "
from risk.risk_manager import RiskManager
risk_manager = RiskManager(config)
print(risk_manager.get_portfolio_summary())
"
```

## 🔧 Advanced Usage

### Custom Strategy Development
```python
from strategy.avellaneda_stoikov import AvellanedaStoikovEngine

class CustomStrategy(AvellanedaStoikovEngine):
    def calculate_optimal_quotes(self, snapshot, volatility_forecast=None):
        # Add your custom logic here
        quotes = super().calculate_optimal_quotes(snapshot, volatility_forecast)
        
        # Example: Adjust spreads based on time of day
        import datetime
        hour = datetime.datetime.now().hour
        if 9 <= hour <= 16:  # Market hours
            quotes.spread *= 0.8  # Tighter spreads
        
        return quotes
```

### Backtesting Historical Data
```bash
# Download historical data
python -m data.download --symbol BTC-USD --start 2024-01-01 --end 2024-12-31

# Run backtest
python -m backtesting.runner \
    --data data/historical/BTC-USD.csv \
    --config config/strategy.yaml \
    --output results/backtest_results.json
```

### Model Training
```bash
# Train RL model on historical data
python -m models.train \
    --data-path data/historical/ \
    --model-output models/trained/ \
    --epochs 1000 \
    --symbol BTC-USD
```

## 🧪 Testing & Validation

### Run Test Suite
```bash
# All tests (should show 142 passed, 1 skipped)
make test

# Specific test categories
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-connectors    # Exchange connector tests
make test-strategy      # Strategy logic tests
```

### Paper Trading Validation
```bash
# Run paper trading for 24 hours
python main.py --mode paper --duration 86400 --log-level DEBUG

# Analyze paper trading results
python -m analysis.paper_trading_report --log-file logs/trading.log
```

## 🚀 Production Deployment

### Pre-deployment Checklist
- [ ] All tests passing (`make test`)
- [ ] Exchange API credentials configured and tested
- [ ] Risk limits properly set for your account size
- [ ] Monitoring and alerting systems configured
- [ ] Backup and recovery procedures in place
- [ ] Network latency optimized (co-location recommended)

### Docker Deployment
```bash
# Build container
docker build -t hft-mm-bot .

# Run with environment file
docker run --env-file .env --name hft-bot hft-mm-bot

# View logs
docker logs -f hft-bot
```

### Systemd Service (Linux)
```bash
# Create service file
sudo tee /etc/systemd/system/hft-mm-bot.service > /dev/null <<EOF
[Unit]
Description=HFT Market Making Bot
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/opt/hft_mm_bot
ExecStart=/opt/hft_mm_bot/venv/bin/python main.py --mode live
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable hft-mm-bot
sudo systemctl start hft-mm-bot
```

## 📊 Performance Optimization

### Latency Optimization
```python
# config/performance.yaml
performance:
  cpu_affinity: [0, 1]      # Pin to specific CPU cores
  memory_lock: true         # Lock memory to prevent swapping
  network_buffer_size: 65536 # Increase network buffer
  gc_disable: true          # Disable garbage collection during trading
```

### Network Configuration
```bash
# Optimize network settings (Linux)
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' >> /etc/sysctl.conf
sysctl -p
```

## 🔍 Troubleshooting

### Common Issues

**1. Connection Errors**
```bash
# Check network connectivity
ping api.binance.com
ping api.exchange.coinbase.com
ping www.okx.com

# Verify API credentials
python -c "
from connectors.binance import BinanceConnector
connector = BinanceConnector(config)
print(connector.test_connection())
"
```

**2. High Latency**
```bash
# Measure latency to exchanges
python -m monitoring.latency_test

# Expected results:
# Binance: <50ms
# Coinbase: <100ms
# OKX: <100ms
```

**3. Risk Violations**
```bash
# Check current risk status
python -c "
from risk.risk_manager import RiskManager
rm = RiskManager(config)
print(rm.get_risk_summary())
"
```

**4. Model Training Issues**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Train with CPU if GPU unavailable
python -m models.train --device cpu
```

### Debug Mode
```bash
# Run with maximum logging
python main.py --mode paper --log-level DEBUG --debug

# Enable profiling
python -m cProfile -o profile.stats main.py --mode paper
```

## 📞 Support & Resources

### Documentation
- **API Reference**: See `docs/api/` directory
- **Strategy Guide**: See `docs/strategy/` directory
- **Risk Management**: See `docs/risk/` directory

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/Blankeeir/hft_mm_bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Blankeeir/hft_mm_bot/discussions)
- **Email**: xusiyi2005@gmail.com

### Community
- Join our Discord server for real-time support
- Follow development updates on Twitter
- Contribute to the project on GitHub

## ⚠️ Important Disclaimers

### Risk Warning
- **High Risk**: Cryptocurrency trading involves substantial risk of loss
- **No Guarantees**: Past performance does not guarantee future results
- **Capital Risk**: Never trade with money you cannot afford to lose
- **Market Risk**: Crypto markets are highly volatile and unpredictable

### Legal Compliance
- Ensure compliance with local regulations
- Some jurisdictions restrict algorithmic trading
- Consult with legal and financial advisors
- Understand tax implications in your jurisdiction

### Technical Risks
- **Software Bugs**: Thoroughly test before live trading
- **Network Issues**: Ensure reliable internet connection
- **Exchange Risks**: Exchanges may experience downtime or issues
- **Model Risk**: AI models may perform poorly in new market conditions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research paper: "A Comprehensive Research Plan for Profitable Cryptocurrency Market Making"
- Avellaneda & Stoikov original market making framework
- PPO algorithm implementation based on OpenAI research
- Exchange API documentation and developer communities

---

**🎯 Ready to Start Trading?**

1. **Test First**: Always start with paper trading
2. **Start Small**: Use minimal position sizes initially
3. **Monitor Closely**: Watch performance and risk metrics
4. **Scale Gradually**: Increase size only after proven performance
5. **Stay Informed**: Keep up with market conditions and updates

**Built with ❤️ for institutional-grade cryptocurrency market making**

