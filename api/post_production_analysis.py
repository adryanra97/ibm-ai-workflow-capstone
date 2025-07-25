"""
Post-production analysis script for investigating the relationship 
between model performance and business metrics
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import json
from pathlib import Path

from services.data_service import DataService
from services.model_service import ModelService
from services.log_service import LogService
from utils.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PostProductionAnalyzer:
    """
    Analyze the relationship between model performance and business metrics
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        self.settings = get_settings()
        self.data_service = DataService()
        self.model_service = ModelService()
        self.log_service = LogService()
        
        # Analysis parameters
        self.analysis_period_days = 90
        self.forecast_horizons = [7, 14, 30, 60]
        self.countries = ["United States", "Canada", "United Kingdom", "Australia"]
        self.model_types = ["prophet", "arima"]
        
        # Output directory
        self.output_dir = Path("analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.analysis_results = {}
        
    async def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive post-production analysis
        
        Returns:
            Complete analysis results
        """
        logger.info("Starting post-production analysis")
        
        analysis_results = {
            'analysis_metadata': {
                'start_time': datetime.now().isoformat(),
                'analysis_period_days': self.analysis_period_days,
                'countries_analyzed': self.countries,
                'model_types': self.model_types,
                'forecast_horizons': self.forecast_horizons
            },
            'model_performance_analysis': {},
            'business_impact_analysis': {},
            'drift_analysis': {},
            'comparative_analysis': {},
            'recommendations': []
        }
        
        try:
            # 1. Model Performance Analysis
            logger.info("Analyzing model performance...")
            analysis_results['model_performance_analysis'] = await self._analyze_model_performance()
            
            # 2. Business Impact Analysis
            logger.info("Analyzing business impact...")
            analysis_results['business_impact_analysis'] = await self._analyze_business_impact()
            
            # 3. Model Drift Analysis
            logger.info("Analyzing model drift...")
            analysis_results['drift_analysis'] = await self._analyze_model_drift()
            
            # 4. Comparative Analysis
            logger.info("Running comparative analysis...")
            analysis_results['comparative_analysis'] = await self._compare_models_performance()
            
            # 5. Generate Recommendations
            logger.info("Generating recommendations...")
            analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
            
            # 6. Create Visualizations
            logger.info("Creating visualizations...")
            await self._create_visualizations(analysis_results)
            
            # 7. Save Results
            await self._save_analysis_results(analysis_results)
            
            analysis_results['analysis_metadata']['end_time'] = datetime.now().isoformat()
            analysis_results['analysis_metadata']['status'] = 'completed'
            
            logger.info("Post-production analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis_results['analysis_metadata']['status'] = 'failed'
            analysis_results['analysis_metadata']['error'] = str(e)
            return analysis_results
    
    async def _analyze_model_performance(self) -> Dict[str, Any]:
        """
        Analyze model performance across different metrics and time periods
        """
        performance_analysis = {
            'accuracy_metrics': {},
            'forecast_horizon_analysis': {},
            'country_performance': {},
            'model_comparison': {}
        }
        
        # Analyze each country and model combination
        for country in self.countries:
            performance_analysis['country_performance'][country] = {}
            
            for model_type in self.model_types:
                try:
                    # Get model predictions and actual values
                    predictions_data = await self._get_historical_predictions(country, model_type)
                    actual_data = await self._get_actual_values(country)
                    
                    if predictions_data and actual_data:
                        # Calculate accuracy metrics
                        metrics = self._calculate_accuracy_metrics(predictions_data, actual_data)
                        performance_analysis['country_performance'][country][model_type] = metrics
                        
                        # Analyze by forecast horizon
                        horizon_metrics = self._analyze_by_forecast_horizon(predictions_data, actual_data)
                        
                        key = f"{country}_{model_type}"
                        performance_analysis['forecast_horizon_analysis'][key] = horizon_metrics
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {model_type} for {country}: {e}")
                    performance_analysis['country_performance'][country][model_type] = {'error': str(e)}
        
        # Calculate aggregate metrics
        performance_analysis['accuracy_metrics'] = self._calculate_aggregate_metrics(
            performance_analysis['country_performance']
        )
        
        return performance_analysis
    
    async def _analyze_business_impact(self) -> Dict[str, Any]:
        """
        Analyze the business impact of model predictions
        """
        business_analysis = {
            'revenue_correlation': {},
            'prediction_value': {},
            'cost_benefit_analysis': {},
            'business_metrics_impact': {}
        }
        
        for country in self.countries:
            try:
                # Get business metrics
                business_metrics = await self.data_service.get_business_metrics(
                    country=country,
                    days_back=self.analysis_period_days
                )
                
                # Get model predictions for the same period
                prediction_data = await self._get_recent_predictions(country)
                
                if business_metrics and prediction_data:
                    # Analyze correlation between predictions and business outcomes
                    correlation_analysis = self._analyze_prediction_business_correlation(
                        prediction_data, business_metrics
                    )
                    business_analysis['revenue_correlation'][country] = correlation_analysis
                    
                    # Calculate prediction value (how much the predictions helped)
                    prediction_value = self._calculate_prediction_value(
                        prediction_data, business_metrics
                    )
                    business_analysis['prediction_value'][country] = prediction_value
                    
                    # Cost-benefit analysis
                    cost_benefit = self._calculate_cost_benefit(
                        prediction_data, business_metrics
                    )
                    business_analysis['cost_benefit_analysis'][country] = cost_benefit
                    
            except Exception as e:
                logger.warning(f"Failed business impact analysis for {country}: {e}")
                business_analysis['revenue_correlation'][country] = {'error': str(e)}
        
        return business_analysis
    
    async def _analyze_model_drift(self) -> Dict[str, Any]:
        """
        Analyze model drift over time
        """
        drift_analysis = {
            'performance_drift': {},
            'data_drift': {},
            'concept_drift': {},
            'drift_alerts': []
        }
        
        for country in self.countries:
            for model_type in self.model_types:
                try:
                    # Get model performance over time
                    performance_timeline = await self._get_model_performance_timeline(country, model_type)
                    
                    if performance_timeline:
                        # Analyze performance drift
                        perf_drift = self._detect_performance_drift(performance_timeline)
                        drift_analysis['performance_drift'][f"{country}_{model_type}"] = perf_drift
                        
                        # Analyze data drift
                        data_drift = await self._detect_data_drift(country, model_type)
                        drift_analysis['data_drift'][f"{country}_{model_type}"] = data_drift
                        
                        # Check for concept drift
                        concept_drift = self._detect_concept_drift(performance_timeline)
                        drift_analysis['concept_drift'][f"{country}_{model_type}"] = concept_drift
                        
                        # Generate alerts for significant drift
                        if perf_drift.get('significant_drift', False):
                            drift_analysis['drift_alerts'].append({
                                'country': country,
                                'model_type': model_type,
                                'drift_type': 'performance',
                                'severity': perf_drift.get('severity', 'medium'),
                                'details': perf_drift
                            })
                        
                except Exception as e:
                    logger.warning(f"Failed drift analysis for {model_type} in {country}: {e}")
        
        return drift_analysis
    
    async def _compare_models_performance(self) -> Dict[str, Any]:
        """
        Compare performance between different models
        """
        comparison_analysis = {
            'model_rankings': {},
            'statistical_significance': {},
            'contextual_performance': {},
            'ensemble_opportunities': {}
        }
        
        for country in self.countries:
            try:
                # Get performance data for all models
                model_performances = {}
                for model_type in self.model_types:
                    predictions = await self._get_historical_predictions(country, model_type)
                    actual = await self._get_actual_values(country)
                    
                    if predictions and actual:
                        metrics = self._calculate_accuracy_metrics(predictions, actual)
                        model_performances[model_type] = metrics
                
                if len(model_performances) >= 2:
                    # Rank models by performance
                    rankings = self._rank_models(model_performances)
                    comparison_analysis['model_rankings'][country] = rankings
                    
                    # Test statistical significance of differences
                    significance_tests = self._test_statistical_significance(model_performances)
                    comparison_analysis['statistical_significance'][country] = significance_tests
                    
                    # Analyze contextual performance (when each model performs better)
                    contextual = await self._analyze_contextual_performance(country, model_performances)
                    comparison_analysis['contextual_performance'][country] = contextual
                    
                    # Identify ensemble opportunities
                    ensemble_ops = self._identify_ensemble_opportunities(model_performances)
                    comparison_analysis['ensemble_opportunities'][country] = ensemble_ops
                    
            except Exception as e:
                logger.warning(f"Failed model comparison for {country}: {e}")
                comparison_analysis['model_rankings'][country] = {'error': str(e)}
        
        return comparison_analysis
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on analysis results
        """
        recommendations = []
        
        # Model Performance Recommendations
        perf_analysis = analysis_results.get('model_performance_analysis', {})
        country_perf = perf_analysis.get('country_performance', {})
        
        for country, models in country_perf.items():
            for model_type, metrics in models.items():
                if isinstance(metrics, dict) and 'mae' in metrics:
                    # High error recommendation
                    if metrics['mae'] > 500:  # Threshold for high error
                        recommendations.append({
                            'type': 'model_performance',
                            'priority': 'high',
                            'country': country,
                            'model': model_type,
                            'issue': 'High prediction error',
                            'recommendation': f'Consider retraining {model_type} model for {country} or adjusting hyperparameters',
                            'details': f"MAE: {metrics['mae']:.2f}, RMSE: {metrics.get('rmse', 0):.2f}"
                        })
                    
                    # Low R² recommendation
                    if metrics.get('r2', 0) < 0.7:
                        recommendations.append({
                            'type': 'model_performance',
                            'priority': 'medium',
                            'country': country,
                            'model': model_type,
                            'issue': 'Low model fit',
                            'recommendation': f'Consider adding more features or using a different model architecture for {country}',
                            'details': f"R²: {metrics.get('r2', 0):.3f}"
                        })
        
        # Drift Recommendations
        drift_analysis = analysis_results.get('drift_analysis', {})
        drift_alerts = drift_analysis.get('drift_alerts', [])
        
        for alert in drift_alerts:
            recommendations.append({
                'type': 'model_drift',
                'priority': alert.get('severity', 'medium'),
                'country': alert['country'],
                'model': alert['model_type'],
                'issue': f'{alert["drift_type"].title()} drift detected',
                'recommendation': f'Retrain {alert["model_type"]} model for {alert["country"]} with recent data',
                'details': alert.get('details', {})
            })
        
        # Model Comparison Recommendations
        comparison_analysis = analysis_results.get('comparative_analysis', {})
        rankings = comparison_analysis.get('model_rankings', {})
        
        for country, ranking_data in rankings.items():
            if isinstance(ranking_data, dict) and 'rankings' in ranking_data:
                best_model = ranking_data['rankings'][0] if ranking_data['rankings'] else None
                if best_model:
                    recommendations.append({
                        'type': 'model_selection',
                        'priority': 'low',
                        'country': country,
                        'model': 'all',
                        'issue': 'Model selection optimization',
                        'recommendation': f'Consider using {best_model["model"]} as primary model for {country}',
                        'details': f"Best performing model with score: {best_model.get('score', 0):.3f}"
                    })
        
        # Business Impact Recommendations
        business_analysis = analysis_results.get('business_impact_analysis', {})
        revenue_correlation = business_analysis.get('revenue_correlation', {})
        
        for country, correlation_data in revenue_correlation.items():
            if isinstance(correlation_data, dict) and 'correlation_strength' in correlation_data:
                if correlation_data['correlation_strength'] == 'weak':
                    recommendations.append({
                        'type': 'business_impact',
                        'priority': 'medium',
                        'country': country,
                        'model': 'all',
                        'issue': 'Weak correlation with business metrics',
                        'recommendation': f'Investigate model features and business KPI alignment for {country}',
                        'details': correlation_data
                    })
        
        # Sort recommendations by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    async def _create_visualizations(self, analysis_results: Dict[str, Any]) -> None:
        """
        Create comprehensive visualizations of the analysis results
        """
        # 1. Model Performance Comparison
        await self._plot_model_performance_comparison(analysis_results)
        
        # 2. Forecast Accuracy by Horizon
        await self._plot_forecast_accuracy_by_horizon(analysis_results)
        
        # 3. Business Impact Correlation
        await self._plot_business_impact_correlation(analysis_results)
        
        # 4. Model Drift Over Time
        await self._plot_model_drift(analysis_results)
        
        # 5. Country-wise Performance
        await self._plot_country_performance(analysis_results)
        
        # 6. Error Distribution Analysis
        await self._plot_error_distributions(analysis_results)
    
    async def _plot_model_performance_comparison(self, analysis_results: Dict[str, Any]) -> None:
        """Create model performance comparison plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Comparison Across Countries', fontsize=16)
            
            country_perf = analysis_results.get('model_performance_analysis', {}).get('country_performance', {})
            
            # Prepare data for visualization
            data_for_plot = []
            for country, models in country_perf.items():
                for model_type, metrics in models.items():
                    if isinstance(metrics, dict) and 'mae' in metrics:
                        data_for_plot.append({
                            'Country': country,
                            'Model': model_type,
                            'MAE': metrics.get('mae', 0),
                            'RMSE': metrics.get('rmse', 0),
                            'MAPE': metrics.get('mape', 0),
                            'R²': metrics.get('r2', 0)
                        })
            
            if data_for_plot:
                df = pd.DataFrame(data_for_plot)
                
                # MAE comparison
                sns.barplot(data=df, x='Country', y='MAE', hue='Model', ax=axes[0,0])
                axes[0,0].set_title('Mean Absolute Error by Country')
                axes[0,0].tick_params(axis='x', rotation=45)
                
                # RMSE comparison
                sns.barplot(data=df, x='Country', y='RMSE', hue='Model', ax=axes[0,1])
                axes[0,1].set_title('Root Mean Square Error by Country')
                axes[0,1].tick_params(axis='x', rotation=45)
                
                # MAPE comparison
                sns.barplot(data=df, x='Country', y='MAPE', hue='Model', ax=axes[1,0])
                axes[1,0].set_title('Mean Absolute Percentage Error by Country')
                axes[1,0].tick_params(axis='x', rotation=45)
                
                # R² comparison
                sns.barplot(data=df, x='Country', y='R²', hue='Model', ax=axes[1,1])
                axes[1,1].set_title('R² Score by Country')
                axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create performance comparison plot: {e}")
    
    async def _plot_forecast_accuracy_by_horizon(self, analysis_results: Dict[str, Any]) -> None:
        """Create forecast accuracy by horizon plots"""
        try:
            horizon_analysis = analysis_results.get('model_performance_analysis', {}).get('forecast_horizon_analysis', {})
            
            if not horizon_analysis:
                return
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data
            data_for_plot = []
            for key, metrics in horizon_analysis.items():
                if isinstance(metrics, dict):
                    country, model = key.split('_', 1)
                    for horizon, accuracy in metrics.items():
                        if isinstance(accuracy, (int, float)):
                            data_for_plot.append({
                                'Country': country,
                                'Model': model,
                                'Horizon': int(horizon),
                                'Accuracy': accuracy
                            })
            
            if data_for_plot:
                df = pd.DataFrame(data_for_plot)
                
                # Create line plot
                for model in df['Model'].unique():
                    model_data = df[df['Model'] == model]
                    for country in model_data['Country'].unique():
                        country_data = model_data[model_data['Country'] == country]
                        ax.plot(country_data['Horizon'], country_data['Accuracy'], 
                               marker='o', label=f'{model} - {country}')
                
                ax.set_xlabel('Forecast Horizon (days)')
                ax.set_ylabel('Accuracy Score')
                ax.set_title('Forecast Accuracy by Horizon')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'forecast_accuracy_by_horizon.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create horizon accuracy plot: {e}")
    
    async def _plot_business_impact_correlation(self, analysis_results: Dict[str, Any]) -> None:
        """Create business impact correlation plots"""
        try:
            business_analysis = analysis_results.get('business_impact_analysis', {})
            revenue_correlation = business_analysis.get('revenue_correlation', {})
            
            if not revenue_correlation:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Business Impact Analysis', fontsize=16)
            
            # Prepare correlation data
            correlation_data = []
            for country, corr_data in revenue_correlation.items():
                if isinstance(corr_data, dict) and 'correlation_coefficient' in corr_data:
                    correlation_data.append({
                        'Country': country,
                        'Correlation': corr_data['correlation_coefficient'],
                        'Prediction_Accuracy': corr_data.get('prediction_accuracy', 0),
                        'Business_Value': corr_data.get('business_value_score', 0)
                    })
            
            if correlation_data:
                df = pd.DataFrame(correlation_data)
                
                # Correlation by country
                axes[0,0].bar(df['Country'], df['Correlation'])
                axes[0,0].set_title('Prediction-Revenue Correlation by Country')
                axes[0,0].tick_params(axis='x', rotation=45)
                axes[0,0].set_ylabel('Correlation Coefficient')
                
                # Prediction accuracy vs business value
                axes[0,1].scatter(df['Prediction_Accuracy'], df['Business_Value'])
                axes[0,1].set_xlabel('Prediction Accuracy')
                axes[0,1].set_ylabel('Business Value Score')
                axes[0,1].set_title('Accuracy vs Business Value')
                
                # Add country labels to scatter plot
                for i, country in enumerate(df['Country']):
                    axes[0,1].annotate(country, (df['Prediction_Accuracy'][i], df['Business_Value'][i]))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'business_impact_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create business impact plot: {e}")
    
    # Additional helper methods would be implemented here...
    # For brevity, I'm including key structural methods
    
    async def _get_historical_predictions(self, country: str, model_type: str) -> Optional[pd.DataFrame]:
        """Get historical predictions for a country and model"""
        # This would query the log database for historical predictions
        # Implementation would depend on the specific log structure
        return None
    
    async def _get_actual_values(self, country: str) -> Optional[pd.DataFrame]:
        """Get actual revenue values for comparison"""
        try:
            return await self.data_service.get_training_data(
                country=country,
                start_date=datetime.now().date() - timedelta(days=self.analysis_period_days),
                end_date=datetime.now().date()
            )
        except Exception as e:
            logger.error(f"Failed to get actual values for {country}: {e}")
            return None
    
    def _calculate_accuracy_metrics(self, predictions: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
        """Calculate accuracy metrics between predictions and actual values"""
        # Implementation would align predictions with actual values and calculate metrics
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'mape': 0.0,
            'r2': 0.0
        }
    
    async def _save_analysis_results(self, analysis_results: Dict[str, Any]) -> None:
        """Save analysis results to files"""
        try:
            # Save JSON results
            output_file = self.output_dir / f"post_production_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            # Save summary report
            summary_file = self.output_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_file, 'w') as f:
                f.write("POST-PRODUCTION ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Analysis Period: {self.analysis_period_days} days\n")
                f.write(f"Countries Analyzed: {', '.join(self.countries)}\n")
                f.write(f"Model Types: {', '.join(self.model_types)}\n\n")
                
                # Write recommendations
                recommendations = analysis_results.get('recommendations', [])
                f.write(f"RECOMMENDATIONS ({len(recommendations)} total):\n")
                f.write("-" * 30 + "\n\n")
                
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec['issue']} ({rec['priority']} priority)\n")
                    f.write(f"   Country: {rec['country']}, Model: {rec['model']}\n")
                    f.write(f"   Recommendation: {rec['recommendation']}\n\n")
            
            logger.info(f"Analysis results saved to {output_file}")
            logger.info(f"Analysis summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")


async def main():
    """
    Main function to run the post-production analysis
    """
    analyzer = PostProductionAnalyzer()
    
    try:
        results = await analyzer.run_full_analysis()
        
        print("\n" + "=" * 60)
        print("POST-PRODUCTION ANALYSIS COMPLETED")
        print("=" * 60)
        
        print(f"\nAnalysis Status: {results['analysis_metadata']['status']}")
        print(f"Countries Analyzed: {len(results['analysis_metadata']['countries_analyzed'])}")
        print(f"Models Analyzed: {len(results['analysis_metadata']['model_types'])}")
        
        recommendations = results.get('recommendations', [])
        print(f"\nRecommendations Generated: {len(recommendations)}")
        
        if recommendations:
            print("\nTop Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"{i}. {rec['issue']} ({rec['priority']} priority)")
                print(f"   {rec['recommendation']}")
        
        print(f"\nDetailed results saved to: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
