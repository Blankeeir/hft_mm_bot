import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import time

from data.volatility import VolatilityForecaster, HARModel, TimeMixerModel, VolatilityForecast


@pytest.fixture
def volatility_config():
    return {
        "har_model": {
            "daily_window": 1,
            "weekly_window": 7,
            "monthly_window": 30
        },
        "timemixer_model": {
            "forecast_horizon": 12,
            "input_features": ["ohlcv", "volume", "volatility"],
            "hidden_dim": 128,
            "num_layers": 4
        }
    }


@pytest.fixture
def sample_price_data():
    base_price = 50000.0
    current_time = int(time.time() * 1000)
    
    price_data = []
    for i in range(100):
        timestamp = current_time - (99 - i) * 3600000
        price = base_price * (1 + np.random.normal(0, 0.02))
        price_data.append((timestamp, price))
        
    return price_data


@pytest.fixture
def sample_feature_data():
    current_time = int(time.time() * 1000)
    
    feature_data = []
    for i in range(100):
        timestamp = current_time - (99 - i) * 3600000
        features = {
            "price": 50000.0 * (1 + np.random.normal(0, 0.02)),
            "volume": np.random.exponential(1000),
            "volatility": abs(np.random.normal(0, 0.02))
        }
        feature_data.append((timestamp, features))
        
    return feature_data


class TestHARModel:
    def test_initialization(self, volatility_config):
        model = HARModel(volatility_config["har_model"])
        assert model.daily_window == 1
        assert model.weekly_window == 7
        assert model.monthly_window == 30
        assert model.is_fitted is False
        
    def test_prepare_features(self, volatility_config):
        model = HARModel(volatility_config["har_model"])
        
        realized_vols = [0.01, 0.02, 0.015, 0.025, 0.018] * 10
        features = model.prepare_features(realized_vols)
        
        assert features is not None
        assert len(features) == 4
        assert features[0] == 1.0
        
    def test_fit_success(self, volatility_config, sample_price_data):
        model = HARModel(volatility_config["har_model"])
        
        success = model.fit(sample_price_data)
        
        assert success is True
        assert model.is_fitted is True
        assert model.coefficients is not None
        
    def test_fit_insufficient_data(self, volatility_config):
        model = HARModel(volatility_config["har_model"])
        
        short_data = [(int(time.time() * 1000), 50000.0)] * 5
        success = model.fit(short_data)
        
        assert success is False
        assert model.is_fitted is False
        
    def test_predict(self, volatility_config, sample_price_data):
        model = HARModel(volatility_config["har_model"])
        model.fit(sample_price_data)
        
        realized_vols = [0.01, 0.02, 0.015] * 15
        prediction = model.predict(realized_vols)
        
        assert isinstance(prediction, float)
        assert prediction >= 0
        
    def test_predict_not_fitted(self, volatility_config):
        model = HARModel(volatility_config["har_model"])
        
        realized_vols = [0.01, 0.02, 0.015]
        prediction = model.predict(realized_vols)
        
        assert prediction == 0.0


class TestTimeMixerModel:
    def test_initialization(self, volatility_config):
        model = TimeMixerModel(volatility_config["timemixer_model"])
        assert model.input_dim == 3
        assert model.hidden_dim == 128
        assert model.num_layers == 4
        assert model.forecast_horizon == 12
        
    def test_forward_pass(self, volatility_config):
        model = TimeMixerModel(volatility_config["timemixer_model"])
        
        batch_size = 4
        seq_len = 24
        features = 3
        
        x = torch.randn(batch_size, seq_len, features)
        output = model.forward(x)
        
        assert output.shape == (batch_size, 12)
        
    def test_prepare_sequences(self, volatility_config, sample_feature_data):
        model = TimeMixerModel(volatility_config["timemixer_model"])
        
        X, y = model.prepare_sequences(sample_feature_data)
        
        assert X is not None
        assert y is not None
        assert X.shape[1] == 24
        assert X.shape[2] == 3
        assert y.shape[1] == 12
        
    def test_prepare_sequences_insufficient_data(self, volatility_config):
        model = TimeMixerModel(volatility_config["timemixer_model"])
        
        short_data = [(int(time.time() * 1000), {"price": 50000, "volume": 1000, "volatility": 0.01})] * 10
        X, y = model.prepare_sequences(short_data)
        
        assert X is None
        assert y is None
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fit_with_gpu(self, volatility_config, sample_feature_data):
        model = TimeMixerModel(volatility_config["timemixer_model"])
        
        success = model.fit(sample_feature_data, epochs=5)
        
        assert success is True
        assert model.is_fitted is True
        
    def test_fit_cpu(self, volatility_config, sample_feature_data):
        model = TimeMixerModel(volatility_config["timemixer_model"])
        
        with patch('torch.cuda.is_available', return_value=False):
            success = model.fit(sample_feature_data, epochs=5)
            
        assert success is True
        assert model.is_fitted is True
        
    def test_predict(self, volatility_config, sample_feature_data):
        model = TimeMixerModel(volatility_config["timemixer_model"])
        model.fit(sample_feature_data, epochs=5)
        
        recent_data = sample_feature_data[-30:]
        predictions = model.predict(recent_data)
        
        assert len(predictions) == 12
        assert all(pred >= 0 for pred in predictions)
        
    def test_predict_not_fitted(self, volatility_config, sample_feature_data):
        model = TimeMixerModel(volatility_config["timemixer_model"])
        
        recent_data = sample_feature_data[-30:]
        predictions = model.predict(recent_data)
        
        assert predictions == [0.0] * 12


class TestVolatilityForecaster:
    def test_initialization(self, volatility_config):
        forecaster = VolatilityForecaster(volatility_config)
        assert forecaster.config == volatility_config
        assert forecaster.har_config == volatility_config["har_model"]
        assert forecaster.timemixer_config == volatility_config["timemixer_model"]
        
    def test_add_price_data(self, volatility_config):
        forecaster = VolatilityForecaster(volatility_config)
        
        current_time = int(time.time() * 1000)
        forecaster.add_price_data("BTC/USDT", current_time, 50000.0, 1000.0)
        
        assert "BTC/USDT" in forecaster.price_history
        assert "BTC/USDT" in forecaster.feature_history
        assert len(forecaster.price_history["BTC/USDT"]) == 1
        assert len(forecaster.feature_history["BTC/USDT"]) == 1
        
    def test_add_multiple_price_data(self, volatility_config, sample_price_data):
        forecaster = VolatilityForecaster(volatility_config)
        
        for timestamp, price in sample_price_data:
            forecaster.add_price_data("BTC/USDT", timestamp, price, 1000.0)
            
        assert len(forecaster.price_history["BTC/USDT"]) == len(sample_price_data)
        assert len(forecaster.feature_history["BTC/USDT"]) == len(sample_price_data)
        
    def test_should_retrain(self, volatility_config):
        forecaster = VolatilityForecaster(volatility_config)
        
        current_time = int(time.time() * 1000)
        old_time = current_time - forecaster.update_interval - 1000
        
        forecaster.last_update["BTC/USDT"] = old_time
        assert forecaster.should_retrain("BTC/USDT") is True
        
        forecaster.last_update["BTC/USDT"] = current_time
        assert forecaster.should_retrain("BTC/USDT") is False
        
    def test_train_models_insufficient_data(self, volatility_config):
        forecaster = VolatilityForecaster(volatility_config)
        
        short_data = [(int(time.time() * 1000), 50000.0)] * 10
        for timestamp, price in short_data:
            forecaster.add_price_data("BTC/USDT", timestamp, price, 1000.0)
            
        success = forecaster.train_models("BTC/USDT")
        assert success is False
        
    def test_train_models_success(self, volatility_config, sample_price_data):
        forecaster = VolatilityForecaster(volatility_config)
        
        for timestamp, price in sample_price_data:
            forecaster.add_price_data("BTC/USDT", timestamp, price, 1000.0)
            
        success = forecaster.train_models("BTC/USDT")
        assert success is True
        assert "BTC/USDT" in forecaster.har_models
        assert "BTC/USDT" in forecaster.timemixer_models
        
    def test_forecast_volatility_no_data(self, volatility_config):
        forecaster = VolatilityForecaster(volatility_config)
        
        forecast = forecaster.forecast_volatility("BTC/USDT")
        assert forecast is None
        
    def test_forecast_volatility_success(self, volatility_config, sample_price_data):
        forecaster = VolatilityForecaster(volatility_config)
        
        for timestamp, price in sample_price_data:
            forecaster.add_price_data("BTC/USDT", timestamp, price, 1000.0)
            
        forecaster.train_models("BTC/USDT")
        forecast = forecaster.forecast_volatility("BTC/USDT")
        
        assert forecast is not None
        assert isinstance(forecast, VolatilityForecast)
        assert forecast.symbol == "BTC/USDT"
        assert forecast.combined_forecast >= 0
        assert 0 <= forecast.confidence <= 1
        
    def test_get_current_volatility(self, volatility_config, sample_price_data):
        forecaster = VolatilityForecaster(volatility_config)
        
        for timestamp, price in sample_price_data:
            forecaster.add_price_data("BTC/USDT", timestamp, price, 1000.0)
            
        volatility = forecaster.get_current_volatility("BTC/USDT")
        
        assert isinstance(volatility, float)
        assert volatility >= 0
        
    def test_get_current_volatility_no_data(self, volatility_config):
        forecaster = VolatilityForecaster(volatility_config)
        
        volatility = forecaster.get_current_volatility("BTC/USDT")
        assert volatility == 0.0


@pytest.mark.parametrize("horizon_hours", [1, 6, 12, 24])
def test_forecast_different_horizons(volatility_config, sample_price_data, horizon_hours):
    forecaster = VolatilityForecaster(volatility_config)
    
    for timestamp, price in sample_price_data:
        forecaster.add_price_data("BTC/USDT", timestamp, price, 1000.0)
        
    forecaster.train_models("BTC/USDT")
    forecast = forecaster.forecast_volatility("BTC/USDT", horizon_hours)
    
    if forecast:
        assert forecast.horizon_hours == horizon_hours
        assert forecast.combined_forecast >= 0


def test_volatility_forecast_dataclass():
    forecast = VolatilityForecast(
        symbol="BTC/USDT",
        timestamp=int(time.time() * 1000),
        har_forecast=0.02,
        timemixer_forecast=0.025,
        combined_forecast=0.022,
        confidence=0.8,
        horizon_hours=1
    )
    
    assert forecast.symbol == "BTC/USDT"
    assert forecast.har_forecast == 0.02
    assert forecast.timemixer_forecast == 0.025
    assert forecast.combined_forecast == 0.022
    assert forecast.confidence == 0.8
    assert forecast.horizon_hours == 1
