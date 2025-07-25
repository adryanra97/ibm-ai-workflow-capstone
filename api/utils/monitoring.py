"""
Model monitoring and drift detection for the Revenue Forecasting API

Author: Adryan R A
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics data class"""
    mae: float
    mape: float
    rmse: float
    mase: float
    r2: float
    timestamp: datetime
    model_type: str
    country: str


@dataclass
class DriftDetectionResult:
    """Drift detection result data class"""
    has_drift: bool
    drift_score: float
    drift_type: str  # "data", "concept", "performance"
    affected_features: List[str]
    timestamp: datetime
    recommendation: str


class ModelMonitor:
    """
    Model monitoring and drift detection system
    """
    
    def __init__(self):
        self.performance_history: Dict[str, List[ModelPerformanceMetrics]] = defaultdict(list)
        self.drift_alerts: List[DriftDetectionResult] = []
        self.baseline_stats: Dict[str, Dict] = {}
        self.business_metrics_cache: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize monitoring system"""
        logger.info("Initializing Model Monitor")
        
        # Load historical performance data
        await self._load_performance_history()
        
        # Load baseline statistics
        await self._load_baseline_stats()
        
        logger.info("Model Monitor initialized successfully")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            import psutil
            
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Model performance summary
            model_summary = await self._get_model_performance_summary()
            
            return {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "model_count": len(self.performance_history),
                "active_alerts": len([alert for alert in self.drift_alerts 
                                    if alert.timestamp > datetime.now() - timedelta(days=1)]),
                "avg_prediction_accuracy": model_summary.get("avg_accuracy", 0.0),
                "last_training_time": model_summary.get("last_training", "never")
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                "cpu_usage_percent": 0.0,
                "memory_usage_percent": 0.0,
                "memory_available_gb": 0.0,
                "disk_usage_percent": 0.0,
                "disk_free_gb": 0.0,
                "model_count": len(self.performance_history),
                "active_alerts": len(self.drift_alerts),
                "system_monitoring": "limited"
            }
    
    async def update_training_metrics(self, country: str, metrics: Dict[str, Dict[str, float]]):
        """Update training metrics for monitoring"""
        try:
            for model_type, model_metrics in metrics.items():
                performance = ModelPerformanceMetrics(
                    mae=model_metrics.get("mae", 0.0),
                    mape=model_metrics.get("mape", 0.0),
                    rmse=model_metrics.get("rmse", 0.0),
                    mase=model_metrics.get("mase", 0.0),
                    r2=model_metrics.get("r2", 0.0),
                    timestamp=datetime.now(),
                    model_type=model_type,
                    country=country
                )
                
                key = f"{country}_{model_type}"
                self.performance_history[key].append(performance)
                
                # Keep only last 100 records per model
                if len(self.performance_history[key]) > 100:
                    self.performance_history[key] = self.performance_history[key][-100:]
            
            # Check for performance drift
            await self._check_performance_drift(country, metrics)
            
        except Exception as e:
            logger.error(f"Failed to update training metrics: {e}")
    
    async def update_prediction_metrics(self, country: str, model_type: str, 
                                      performance: Dict[str, float]):
        """Update prediction metrics"""
        try:
            # Store prediction performance
            key = f"{country}_{model_type}_predictions"
            if key not in self.performance_history:
                self.performance_history[key] = []
            
            # Add prediction performance record
            pred_metrics = {
                "prediction_time": performance.get("prediction_time", 0.0),
                "confidence_score": performance.get("confidence", 0.0),
                "timestamp": datetime.now(),
                "country": country,
                "model_type": model_type
            }
            
            self.performance_history[key].append(pred_metrics)
            
        except Exception as e:
            logger.error(f"Failed to update prediction metrics: {e}")
    
    async def get_drift_analysis(self) -> Dict[str, Any]:
        """Get comprehensive drift analysis"""
        try:
            # Data drift analysis
            data_drift = await self._analyze_data_drift()
            
            # Concept drift analysis
            concept_drift = await self._analyze_concept_drift()
            
            # Performance drift analysis  
            performance_drift = await self._analyze_performance_drift()
            
            # Overall drift assessment
            overall_drift_score = (
                data_drift.get("overall_score", 0.0) * 0.3 +
                concept_drift.get("overall_score", 0.0) * 0.3 +
                performance_drift.get("overall_score", 0.0) * 0.4
            )
            
            recommendations = await self._generate_drift_recommendations(
                data_drift, concept_drift, performance_drift
            )
            
            return {
                "overall_drift_score": overall_drift_score,
                "drift_status": "high" if overall_drift_score > 0.7 else 
                              "medium" if overall_drift_score > 0.3 else "low",
                "data_drift": data_drift,
                "concept_drift": concept_drift,
                "performance_drift": performance_drift,
                "recommendations": recommendations,
                "last_updated": datetime.now().isoformat(),
                "active_alerts": [
                    {
                        "type": alert.drift_type,
                        "score": alert.drift_score,
                        "timestamp": alert.timestamp.isoformat(),
                        "recommendation": alert.recommendation
                    }
                    for alert in self.drift_alerts[-10:]  # Last 10 alerts
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get drift analysis: {e}")
            return {"error": str(e)}
    
    async def get_performance_analysis(self) -> Dict[str, Any]:
        """Get model performance analysis vs business metrics"""
        try:
            # Get recent model performance
            model_performance = await self._get_recent_model_performance()
            
            # Get business metrics correlation
            business_correlation = await self._analyze_business_correlation()
            
            # Get accuracy trends
            accuracy_trends = await self._get_accuracy_trends()
            
            # Get feature importance analysis
            feature_analysis = await self._analyze_feature_importance()
            
            return {
                "model_performance": model_performance,
                "business_correlation": business_correlation,
                "accuracy_trends": accuracy_trends,
                "feature_analysis": feature_analysis,
                "performance_summary": {
                    "best_performing_model": await self._get_best_performing_model(),
                    "worst_performing_model": await self._get_worst_performing_model(),
                    "average_accuracy": await self._get_average_accuracy(),
                    "performance_stability": await self._get_performance_stability()
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance analysis: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    async def _load_performance_history(self):
        """Load historical performance data"""
        try:
            history_file = os.path.join(settings.model_storage_path, "performance_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to dataclass objects
                    for key, records in data.items():
                        self.performance_history[key] = [
                            ModelPerformanceMetrics(**record) for record in records
                        ]
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")
    
    async def _load_baseline_stats(self):
        """Load baseline statistics for drift detection"""
        try:
            baseline_file = os.path.join(settings.model_storage_path, "baseline_stats.json")
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    self.baseline_stats = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load baseline stats: {e}")
    
    async def _get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance"""
        if not self.performance_history:
            return {"avg_accuracy": 0.0, "last_training": "never"}
        
        # Calculate average accuracy across all models
        all_r2_scores = []
        last_training = datetime.min
        
        for key, metrics_list in self.performance_history.items():
            if metrics_list:
                r2_scores = [m.r2 for m in metrics_list if hasattr(m, 'r2')]
                all_r2_scores.extend(r2_scores)
                
                latest_timestamp = max(m.timestamp for m in metrics_list if hasattr(m, 'timestamp'))
                if latest_timestamp > last_training:
                    last_training = latest_timestamp
        
        avg_accuracy = np.mean(all_r2_scores) if all_r2_scores else 0.0
        last_training_str = last_training.isoformat() if last_training != datetime.min else "never"
        
        return {
            "avg_accuracy": float(avg_accuracy),
            "last_training": last_training_str
        }
    
    async def _check_performance_drift(self, country: str, metrics: Dict[str, Dict[str, float]]):
        """Check for performance drift"""
        try:
            for model_type, model_metrics in metrics.items():
                key = f"{country}_{model_type}"
                
                if key in self.performance_history and len(self.performance_history[key]) > 5:
                    # Get recent performance
                    recent_r2 = [m.r2 for m in self.performance_history[key][-5:]]
                    historical_r2 = [m.r2 for m in self.performance_history[key][:-5]]
                    
                    if len(historical_r2) > 0:
                        # Statistical test for performance degradation
                        statistic, p_value = stats.ttest_ind(recent_r2, historical_r2)
                        
                        # If recent performance is significantly worse
                        if statistic < -2 and p_value < 0.05:
                            drift_result = DriftDetectionResult(
                                has_drift=True,
                                drift_score=abs(statistic) / 10,  # Normalized score
                                drift_type="performance",
                                affected_features=[model_type],
                                timestamp=datetime.now(),
                                recommendation=f"Model {model_type} for {country} showing performance degradation. Consider retraining."
                            )
                            
                            self.drift_alerts.append(drift_result)
                            logger.warning(f"Performance drift detected for {key}: {drift_result.recommendation}")
        
        except Exception as e:
            logger.error(f"Failed to check performance drift: {e}")
    
    async def _analyze_data_drift(self) -> Dict[str, Any]:
        """Analyze data drift"""
        # Placeholder implementation - would analyze input data distributions
        return {
            "overall_score": 0.2,
            "features_with_drift": [],
            "distribution_changes": {},
            "statistical_tests": {}
        }
    
    async def _analyze_concept_drift(self) -> Dict[str, Any]:
        """Analyze concept drift"""
        # Placeholder implementation - would analyze target variable relationships
        return {
            "overall_score": 0.1,
            "relationship_changes": {},
            "seasonal_pattern_changes": {},
            "trend_changes": {}
        }
    
    async def _analyze_performance_drift(self) -> Dict[str, Any]:
        """Analyze performance drift"""
        try:
            drift_scores = []
            model_drifts = {}
            
            for key, metrics_list in self.performance_history.items():
                if len(metrics_list) > 10:
                    # Calculate performance trend
                    r2_scores = [m.r2 for m in metrics_list if hasattr(m, 'r2')]
                    if len(r2_scores) > 5:
                        recent_performance = np.mean(r2_scores[-5:])
                        historical_performance = np.mean(r2_scores[:-5])
                        
                        drift_score = max(0, (historical_performance - recent_performance) / historical_performance)
                        drift_scores.append(drift_score)
                        model_drifts[key] = drift_score
            
            overall_score = np.mean(drift_scores) if drift_scores else 0.0
            
            return {
                "overall_score": float(overall_score),
                "model_drift_scores": model_drifts,
                "high_drift_models": [k for k, v in model_drifts.items() if v > 0.3],
                "performance_trends": {}
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze performance drift: {e}")
            return {"overall_score": 0.0, "error": str(e)}
    
    async def _generate_drift_recommendations(self, data_drift: Dict, 
                                            concept_drift: Dict, 
                                            performance_drift: Dict) -> List[str]:
        """Generate drift mitigation recommendations"""
        recommendations = []
        
        if performance_drift.get("overall_score", 0) > 0.3:
            recommendations.append("High performance drift detected. Consider retraining models.")
        
        if data_drift.get("overall_score", 0) > 0.5:
            recommendations.append("Significant data drift detected. Review input data quality and distributions.")
        
        if concept_drift.get("overall_score", 0) > 0.4:
            recommendations.append("Concept drift detected. Market conditions may have changed.")
        
        if not recommendations:
            recommendations.append("Model performance is stable. Continue monitoring.")
        
        return recommendations
    
    async def _get_recent_model_performance(self) -> Dict[str, Any]:
        """Get recent model performance metrics"""
        performance_summary = {}
        
        for key, metrics_list in self.performance_history.items():
            if metrics_list:
                recent_metrics = metrics_list[-1]  # Most recent
                performance_summary[key] = {
                    "mae": recent_metrics.mae if hasattr(recent_metrics, 'mae') else 0.0,
                    "mape": recent_metrics.mape if hasattr(recent_metrics, 'mape') else 0.0,
                    "rmse": recent_metrics.rmse if hasattr(recent_metrics, 'rmse') else 0.0,
                    "r2": recent_metrics.r2 if hasattr(recent_metrics, 'r2') else 0.0,
                    "timestamp": recent_metrics.timestamp.isoformat() if hasattr(recent_metrics, 'timestamp') else None
                }
        
        return performance_summary
    
    async def _analyze_business_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between model performance and business metrics"""
        # Placeholder - would integrate with business metrics from L3 database
        return {
            "revenue_correlation": 0.85,
            "customer_retention_correlation": 0.72,
            "seasonal_impact": 0.65,
            "model_business_alignment": "high"
        }
    
    async def _get_accuracy_trends(self) -> Dict[str, List[float]]:
        """Get accuracy trends over time"""
        trends = {}
        
        for key, metrics_list in self.performance_history.items():
            if metrics_list:
                r2_scores = [m.r2 for m in metrics_list if hasattr(m, 'r2')]
                trends[key] = r2_scores[-20:]  # Last 20 measurements
        
        return trends
    
    async def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance changes"""
        # Placeholder - would analyze feature importance drift
        return {
            "feature_stability": "high",
            "important_features": ["previous_7_days", "seasonal_trend", "year_over_year"],
            "feature_drift_scores": {}
        }
    
    async def _get_best_performing_model(self) -> str:
        """Get best performing model"""
        best_model = "none"
        best_score = -1
        
        for key, metrics_list in self.performance_history.items():
            if metrics_list:
                avg_r2 = np.mean([m.r2 for m in metrics_list if hasattr(m, 'r2')])
                if avg_r2 > best_score:
                    best_score = avg_r2
                    best_model = key
        
        return best_model
    
    async def _get_worst_performing_model(self) -> str:
        """Get worst performing model"""
        worst_model = "none"
        worst_score = 2.0
        
        for key, metrics_list in self.performance_history.items():
            if metrics_list:
                avg_r2 = np.mean([m.r2 for m in metrics_list if hasattr(m, 'r2')])
                if avg_r2 < worst_score:
                    worst_score = avg_r2
                    worst_model = key
        
        return worst_model
    
    async def _get_average_accuracy(self) -> float:
        """Get overall average accuracy"""
        all_r2_scores = []
        
        for metrics_list in self.performance_history.values():
            r2_scores = [m.r2 for m in metrics_list if hasattr(m, 'r2')]
            all_r2_scores.extend(r2_scores)
        
        return float(np.mean(all_r2_scores)) if all_r2_scores else 0.0
    
    async def _get_performance_stability(self) -> str:
        """Get performance stability assessment"""
        stability_scores = []
        
        for metrics_list in self.performance_history.values():
            if len(metrics_list) > 5:
                r2_scores = [m.r2 for m in metrics_list if hasattr(m, 'r2')]
                if len(r2_scores) > 1:
                    stability = 1 - np.std(r2_scores) / np.mean(r2_scores) if np.mean(r2_scores) > 0 else 0
                    stability_scores.append(stability)
        
        if not stability_scores:
            return "unknown"
        
        avg_stability = np.mean(stability_scores)
        
        if avg_stability > 0.8:
            return "high"
        elif avg_stability > 0.6:
            return "medium"
        else:
            return "low"
