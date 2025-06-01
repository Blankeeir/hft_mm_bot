import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
import time

logger = logging.getLogger(__name__)


@dataclass
class VolatilityForecast:
    symbol: str
    timestamp: int
    har_forecast: float
    timemixer_forecast: float
    combined_forecast: float
    confidence: float
    horizon_hours: int


class HARModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.daily_window = config.get("daily_window", 1)
        self.weekly_window = config.get("weekly_window", 7)
        self.monthly_window = config.get("monthly_window", 30)
        
        self.coefficients = None
        self.is_fitted = False
        
    def prepare_features(self, realized_volatilities: List[float]) -> np.ndarray:
        if len(realized_volatilities) < self.monthly_window:
            return None
            
        rv_array = np.array(realized_volatilities)
        
        rv_daily = rv_array[-self.daily_window:].mean() if self.daily_window > 0 else 0
        rv_weekly = rv_array[-self.weekly_window:].mean() if len(rv_array) >= self.weekly_window else rv_daily
        rv_monthly = rv_array[-self.monthly_window:].mean() if len(rv_array) >= self.monthly_window else rv_weekly
        
        return np.array([1.0, rv_daily, rv_weekly, rv_monthly])
        
    def fit(self, price_data: List[Tuple[int, float]]) -> bool:
        try:
            if len(price_data) < self.monthly_window + 1:
                return False
                
            prices = np.array([price for _, price in price_data])
            returns = np.diff(np.log(prices))
            
            realized_vols = []
            for i in range(len(returns)):
                if i == 0:
                    realized_vols.append(abs(returns[i]))
                else:
                    window_returns = returns[max(0, i-23):i+1]
                    realized_vols.append(np.sqrt(np.sum(window_returns**2)))
                    
            X = []
            y = []
            
            for i in range(self.monthly_window, len(realized_vols)):
                features = self.prepare_features(realized_vols[:i])
                if features is not None:
                    X.append(features)
                    y.append(realized_vols[i])
                    
            if len(X) < 10:
                return False
                
            X = np.array(X)
            y = np.array(y)
            
            self.coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
            self.is_fitted = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error fitting HAR model: {e}")
            return False
            
    def predict(self, realized_volatilities: List[float]) -> float:
        if not self.is_fitted or self.coefficients is None:
            return 0.0
            
        features = self.prepare_features(realized_volatilities)
        if features is None:
            return 0.0
            
        return max(0.0, np.dot(features, self.coefficients))


class TimeMixerModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.input_dim = len(config.get("input_features", ["ohlcv", "volume", "volatility"]))
        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_layers = config.get("num_layers", 4)
        self.forecast_horizon = config.get("forecast_horizon", 12)
        self.sequence_length = 24
        
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        self.mixing_layers = nn.ModuleList([
            MixingBlock(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        self.output_projection = nn.Linear(self.hidden_dim, self.forecast_horizon)
        self.dropout = nn.Dropout(0.1)
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        
        x = self.input_projection(x)
        
        for mixing_layer in self.mixing_layers:
            x = mixing_layer(x)
            x = self.dropout(x)
            
        x = x.mean(dim=1)
        output = self.output_projection(x)
        
        return output
        
    def prepare_sequences(self, data: List[Tuple[int, Dict[str, float]]]) -> Tuple[np.ndarray, np.ndarray]:
        if len(data) < self.sequence_length + self.forecast_horizon:
            return None, None
            
        features = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            seq_features = []
            
            for j in range(i, i + self.sequence_length):
                timestamp, feature_dict = data[j]
                feature_vector = [
                    feature_dict.get("price", 0.0),
                    feature_dict.get("volume", 0.0),
                    feature_dict.get("volatility", 0.0),
                ]
                seq_features.append(feature_vector)
                
            target_vols = []
            for j in range(i + self.sequence_length, i + self.sequence_length + self.forecast_horizon):
                _, target_dict = data[j]
                target_vols.append(target_dict.get("volatility", 0.0))
                
            features.append(seq_features)
            targets.append(target_vols)
            
        return np.array(features), np.array(targets)
        
    def fit(self, data: List[Tuple[int, Dict[str, float]]], epochs: int = 100) -> bool:
        try:
            X, y = self.prepare_sequences(data)
            if X is None or y is None:
                return False
                
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
            
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            self.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.forward(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logger.debug(f"TimeMixer epoch {epoch}, loss: {loss.item():.6f}")
                    
            self.is_fitted = True
            return True
            
        except Exception as e:
            logger.error(f"Error fitting TimeMixer model: {e}")
            return False
            
    def predict(self, recent_data: List[Tuple[int, Dict[str, float]]]) -> List[float]:
        if not self.is_fitted or len(recent_data) < self.sequence_length:
            return [0.0] * self.forecast_horizon
            
        try:
            features = []
            for i in range(-self.sequence_length, 0):
                timestamp, feature_dict = recent_data[i]
                feature_vector = [
                    feature_dict.get("price", 0.0),
                    feature_dict.get("volume", 0.0),
                    feature_dict.get("volatility", 0.0),
                ]
                features.append(feature_vector)
                
            X = np.array([features])
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
            
            X_tensor = torch.FloatTensor(X)
            
            self.eval()
            with torch.no_grad():
                output = self.forward(X_tensor)
                predictions = output.squeeze().numpy()
                
            return [max(0.0, pred) for pred in predictions]
            
        except Exception as e:
            logger.error(f"Error in TimeMixer prediction: {e}")
            return [0.0] * self.forecast_horizon


class MixingBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.temporal_mixing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.feature_mixing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        temporal_out = x.permute(0, 2, 1)  # (batch, hidden, seq)
        temporal_out = temporal_out.reshape(batch_size * hidden_dim, seq_len)
        temporal_linear = nn.Linear(seq_len, seq_len).to(x.device)
        temporal_out = temporal_linear(temporal_out)
        temporal_out = temporal_out.reshape(batch_size, hidden_dim, seq_len)
        temporal_out = temporal_out.permute(0, 2, 1)  # back to (batch, seq, hidden)
        x = self.norm1(x + temporal_out)
        
        feature_out = self.feature_mixing(x)
        x = self.norm2(x + feature_out)
        
        return x


class VolatilityForecaster:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.har_config = config.get("har_model", {})
        self.timemixer_config = config.get("timemixer_model", {})
        
        self.har_models: Dict[str, HARModel] = {}
        self.timemixer_models: Dict[str, TimeMixerModel] = {}
        
        self.price_history: Dict[str, List[Tuple[int, float]]] = {}
        self.feature_history: Dict[str, List[Tuple[int, Dict[str, float]]]] = {}
        
        self.last_update: Dict[str, int] = {}
        self.update_interval = 3600000
        
    def add_price_data(self, symbol: str, timestamp: int, price: float, volume: float = 0.0) -> None:
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.feature_history[symbol] = []
            
        self.price_history[symbol].append((timestamp, price))
        
        if len(self.price_history[symbol]) > 1:
            prev_price = self.price_history[symbol][-2][1]
            returns = np.log(price / prev_price) if prev_price > 0 else 0.0
            volatility = abs(returns)
        else:
            volatility = 0.0
            
        features = {
            "price": price,
            "volume": volume,
            "volatility": volatility,
        }
        
        self.feature_history[symbol].append((timestamp, features))
        
        max_history = 10000
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.feature_history[symbol] = self.feature_history[symbol][-max_history:]
            
    def should_retrain(self, symbol: str) -> bool:
        current_time = int(time.time() * 1000)
        last_update = self.last_update.get(symbol, 0)
        
        return (current_time - last_update) > self.update_interval
        
    def train_models(self, symbol: str) -> bool:
        if symbol not in self.price_history or len(self.price_history[symbol]) < 100:
            return False
            
        try:
            if symbol not in self.har_models:
                self.har_models[symbol] = HARModel(self.har_config)
                
            if symbol not in self.timemixer_models:
                self.timemixer_models[symbol] = TimeMixerModel(self.timemixer_config)
                
            har_success = self.har_models[symbol].fit(self.price_history[symbol])
            timemixer_success = self.timemixer_models[symbol].fit(self.feature_history[symbol])
            
            if har_success or timemixer_success:
                self.last_update[symbol] = int(time.time() * 1000)
                logger.info(f"Volatility models trained for {symbol}: HAR={har_success}, TimeMixer={timemixer_success}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error training volatility models for {symbol}: {e}")
            return False
            
    def forecast_volatility(self, symbol: str, horizon_hours: int = 1) -> Optional[VolatilityForecast]:
        if symbol not in self.price_history:
            return None
            
        if self.should_retrain(symbol):
            self.train_models(symbol)
            
        current_time = int(time.time() * 1000)
        
        har_forecast = 0.0
        timemixer_forecast = 0.0
        
        if symbol in self.har_models and self.har_models[symbol].is_fitted:
            prices = [price for _, price in self.price_history[symbol]]
            if len(prices) >= 2:
                returns = np.diff(np.log(prices))
                realized_vols = [abs(ret) for ret in returns[-30:]]
                har_forecast = self.har_models[symbol].predict(realized_vols)
                
        if symbol in self.timemixer_models and self.timemixer_models[symbol].is_fitted:
            if len(self.feature_history[symbol]) >= 24:
                predictions = self.timemixer_models[symbol].predict(self.feature_history[symbol])
                if predictions and horizon_hours <= len(predictions):
                    timemixer_forecast = predictions[horizon_hours - 1]
                    
        if har_forecast == 0.0 and timemixer_forecast == 0.0:
            return None
            
        if har_forecast > 0 and timemixer_forecast > 0:
            combined_forecast = 0.6 * har_forecast + 0.4 * timemixer_forecast
            confidence = 0.8
        elif har_forecast > 0:
            combined_forecast = har_forecast
            confidence = 0.6
        else:
            combined_forecast = timemixer_forecast
            confidence = 0.5
            
        return VolatilityForecast(
            symbol=symbol,
            timestamp=current_time,
            har_forecast=har_forecast,
            timemixer_forecast=timemixer_forecast,
            combined_forecast=combined_forecast,
            confidence=confidence,
            horizon_hours=horizon_hours
        )
        
    def get_current_volatility(self, symbol: str, window_hours: int = 1) -> float:
        if symbol not in self.price_history:
            return 0.0
            
        current_time = int(time.time() * 1000)
        start_time = current_time - (window_hours * 3600000)
        
        recent_prices = [price for timestamp, price in self.price_history[symbol] 
                        if timestamp >= start_time]
        
        if len(recent_prices) < 2:
            return 0.0
            
        returns = np.diff(np.log(recent_prices))
        return float(np.std(returns) * np.sqrt(len(returns)))
