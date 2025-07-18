"""
Health Data Processing Utilities
Handles data cleaning, preprocessing, and feature engineering for health monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class HealthDataProcessor:
    """
    Comprehensive data processing for health monitoring system
    """
    
    def __init__(self):
        self.normal_ranges = {
            'heart_rate': (60, 100),
            'blood_oxygen': (95, 100),
            'temperature': (97.0, 99.5),
            'steps': (0, 50000),
            'sleep_quality': (0, 10),
            'stress_level': (0, 10)
        }
        
        self.feature_columns = [
            'heart_rate', 'blood_oxygen', 'temperature',
            'steps', 'sleep_quality', 'stress_level'
        ]
    
    def preprocess_dataset(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Preprocess entire dataset for training
        
        Args:
            raw_data: List of raw health data dictionaries
            
        Returns:
            Cleaned and processed DataFrame
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            
            # Clean data
            df_cleaned = self._clean_data(df)
            
            # Handle missing values
            df_imputed = self._handle_missing_values(df_cleaned)
            
            # Remove outliers
            df_no_outliers = self._remove_outliers(df_imputed)
            
            # Feature engineering
            df_engineered = self._engineer_features(df_no_outliers)
            
            logger.info(f"Processed {len(df)} records, {len(df_engineered)} remaining after cleaning")
            
            return df_engineered
            
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise
    
    def preprocess_single_datapoint(self, health_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Preprocess single health data point
        
        Args:
            health_data: Single health data record
            
        Returns:
            Processed health data
        """
        try:
            # Validate and clean single data point
            cleaned_data = self._validate_single_datapoint(health_data)
            
            # Add derived features
            processed_data = self._add_derived_features(cleaned_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing single datapoint: {str(e)}")
            return self._get_default_datapoint()
    
    def prepare_for_visualization(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Prepare data for frontend visualization
        
        Args:
            data: List of health data records
            
        Returns:
            Data formatted for charts and visualizations
        """
        try:
            df = pd.DataFrame(data)
            
            if df.empty:
                return self._get_empty_visualization_data()
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Prepare time series data
            time_series = {}
            for metric in self.feature_columns:
                if metric in df.columns:
                    time_series[metric] = [
                        {
                            'timestamp': row['timestamp'].isoformat(),
                            'value': float(row[metric]) if pd.notna(row[metric]) else None
                        }
                        for _, row in df.iterrows()
                    ]
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(df)
            
            # Identify trends
            trends = self._calculate_trends(df)
            
            return {
                'time_series': time_series,
                'summary_stats': summary_stats,
                'trends': trends,
                'data_quality': self._assess_data_quality(df)
            }
            
        except Exception as e:
            logger.error(f"Error preparing data for visualization: {str(e)}")
            return self._get_empty_visualization_data()
    
    def generate_summary_report(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary health report
        
        Args:
            data: Historical health data
            
        Returns:
            Summary report with key insights
        """
        try:
            df = pd.DataFrame(data)
            
            if df.empty:
                return self._get_empty_report()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate averages and ranges
            metrics_summary = {}
            for metric in self.feature_columns:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if not values.empty:
                        metrics_summary[metric] = {
                            'average': float(values.mean()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'std': float(values.std()),
                            'trend': self._calculate_metric_trend(values),
                            'normal_range': self.normal_ranges.get(metric, (0, 100)),
                            'within_normal': self._calculate_normal_percentage(values, metric)
                        }
            
            # Identify key insights
            insights = self._generate_insights(df)
            
            # Calculate health score
            health_score = self._calculate_health_score(df)
            
            return {
                'period': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat(),
                    'days': (df['timestamp'].max() - df['timestamp'].min()).days
                },
                'metrics_summary': metrics_summary,
                'health_score': health_score,
                'insights': insights,
                'recommendations': self._generate_report_recommendations(df)
            }
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return self._get_empty_report()
    
    def generate_detailed_report(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Generate detailed health report with advanced analytics
        
        Args:
            data: Historical health data
            
        Returns:
            Detailed report with comprehensive analysis
        """
        try:
            # Start with summary report
            report = self.generate_summary_report(data)
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add detailed analytics
                report['detailed_analytics'] = {
                    'correlation_analysis': self._calculate_correlations(df),
                    'pattern_analysis': self._analyze_patterns(df),
                    'anomaly_frequency': self._analyze_anomaly_frequency(df),
                    'circadian_patterns': self._analyze_circadian_patterns(df),
                    'weekly_patterns': self._analyze_weekly_patterns(df)
                }
                
                # Add predictive insights
                report['predictive_insights'] = self._generate_predictive_insights(df)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {str(e)}")
            return self.generate_summary_report(data)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data by removing invalid entries"""
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Validate numeric columns
        for column in self.feature_columns:
            if column in df.columns:
                # Remove non-numeric values
                df[column] = pd.to_numeric(df[column], errors='coerce')
                
                # Remove values outside reasonable ranges
                if column in self.normal_ranges:
                    min_val, max_val = self.normal_ranges[column]
                    # Allow some flexibility for outlier detection later
                    extended_min = min_val * 0.5
                    extended_max = max_val * 2.0
                    df = df[(df[column] >= extended_min) | (df[column] <= extended_max) | df[column].isna()]
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using forward fill and interpolation"""
        df_filled = df.copy()
        
        # Sort by timestamp for proper forward fill
        df_filled = df_filled.sort_values('timestamp')
        
        # Forward fill missing values
        for column in self.feature_columns:
            if column in df_filled.columns:
                df_filled[column] = df_filled[column].fillna(method='ffill')
        
        # Fill remaining missing values with median
        for column in self.feature_columns:
            if column in df_filled.columns:
                median_value = df_filled[column].median()
                df_filled[column] = df_filled[column].fillna(median_value)
        
        return df_filled
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        df_no_outliers = df.copy()
        
        for column in self.feature_columns:
            if column in df_no_outliers.columns:
                Q1 = df_no_outliers[column].quantile(0.25)
                Q3 = df_no_outliers[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                df_no_outliers = df_no_outliers[
                    (df_no_outliers[column] >= lower_bound) & 
                    (df_no_outliers[column] <= upper_bound)
                ]
        
        return df_no_outliers
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from raw data"""
        df_engineered = df.copy()
        
        # Ensure timestamp is datetime
        df_engineered['timestamp'] = pd.to_datetime(df_engineered['timestamp'])
        
        # Time-based features
        df_engineered['hour'] = df_engineered['timestamp'].dt.hour
        df_engineered['day_of_week'] = df_engineered['timestamp'].dt.dayofweek
        df_engineered['is_weekend'] = df_engineered['day_of_week'].isin([5, 6])
        
        # Rolling averages
        for column in self.feature_columns:
            if column in df_engineered.columns:
                df_engineered[f'{column}_rolling_mean'] = df_engineered[column].rolling(window=5, min_periods=1).mean()
                df_engineered[f'{column}_rolling_std'] = df_engineered[column].rolling(window=5, min_periods=1).std()
        
        # Heart rate variability (simplified)
        if 'heart_rate' in df_engineered.columns:
            df_engineered['heart_rate_variability'] = df_engineered['heart_rate'].rolling(window=3).std()
        
        # Activity level based on steps
        if 'steps' in df_engineered.columns:
            df_engineered['activity_level'] = pd.cut(
                df_engineered['steps'],
                bins=[0, 3000, 7000, 12000, float('inf')],
                labels=['sedentary', 'lightly_active', 'moderately_active', 'very_active']
            )
        
        return df_engineered
    
    def _validate_single_datapoint(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Validate and clean single data point"""
        cleaned_data = {}
        
        for column in self.feature_columns:
            value = data.get(column)
            
            if value is not None:
                try:
                    # Convert to float
                    float_value = float(value)
                    
                    # Validate range
                    if column in self.normal_ranges:
                        min_val, max_val = self.normal_ranges[column]
                        # Allow some flexibility
                        if min_val * 0.5 <= float_value <= max_val * 2.0:
                            cleaned_data[column] = float_value
                        else:
                            # Use median value for out-of-range data
                            cleaned_data[column] = (min_val + max_val) / 2
                    else:
                        cleaned_data[column] = float_value
                        
                except (ValueError, TypeError):
                    # Use default value for invalid data
                    if column in self.normal_ranges:
                        min_val, max_val = self.normal_ranges[column]
                        cleaned_data[column] = (min_val + max_val) / 2
            else:
                # Use default value for missing data
                if column in self.normal_ranges:
                    min_val, max_val = self.normal_ranges[column]
                    cleaned_data[column] = (min_val + max_val) / 2
        
        return cleaned_data
    
    def _add_derived_features(self, data: Dict[str, float]) -> Dict[str, float]:
        """Add derived features to single data point"""
        enhanced_data = data.copy()
        
        # Add timestamp if not present
        if 'timestamp' not in enhanced_data:
            enhanced_data['timestamp'] = datetime.now().timestamp()
        
        # Add time-based features
        current_time = datetime.now()
        enhanced_data['hour'] = current_time.hour
        enhanced_data['day_of_week'] = current_time.weekday()
        enhanced_data['is_weekend'] = current_time.weekday() >= 5
        
        # Calculate activity level
        steps = enhanced_data.get('steps', 0)
        if steps < 3000:
            enhanced_data['activity_level'] = 0  # sedentary
        elif steps < 7000:
            enhanced_data['activity_level'] = 1  # lightly active
        elif steps < 12000:
            enhanced_data['activity_level'] = 2  # moderately active
        else:
            enhanced_data['activity_level'] = 3  # very active
        
        return enhanced_data
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each metric"""
        stats = {}
        
        for column in self.feature_columns:
            if column in df.columns:
                values = df[column].dropna()
                if not values.empty:
                    stats[column] = {
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'count': int(len(values))
                    }
        
        return stats
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Calculate trends for each metric"""
        trends = {}
        
        for column in self.feature_columns:
            if column in df.columns and len(df) > 1:
                values = df[column].dropna()
                if len(values) > 1:
                    # Simple trend calculation
                    first_half = values[:len(values)//2].mean()
                    second_half = values[len(values)//2:].mean()
                    
                    if second_half > first_half * 1.05:
                        trends[column] = 'increasing'
                    elif second_half < first_half * 0.95:
                        trends[column] = 'decreasing'
                    else:
                        trends[column] = 'stable'
                else:
                    trends[column] = 'insufficient_data'
        
        return trends
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        total_records = len(df)
        
        quality_metrics = {
            'total_records': total_records,
            'completeness': {},
            'consistency': {},
            'validity': {}
        }
        
        for column in self.feature_columns:
            if column in df.columns:
                non_null_count = df[column].notna().sum()
                quality_metrics['completeness'][column] = float(non_null_count / total_records)
                
                # Check validity (within reasonable ranges)
                if column in self.normal_ranges:
                    min_val, max_val = self.normal_ranges[column]
                    valid_count = ((df[column] >= min_val * 0.5) & 
                                 (df[column] <= max_val * 2.0)).sum()
                    quality_metrics['validity'][column] = float(valid_count / non_null_count) if non_null_count > 0 else 0
        
        return quality_metrics
    
    def _get_default_datapoint(self) -> Dict[str, float]:
        """Get default data point when processing fails"""
        return {
            'heart_rate': 72.0,
            'blood_oxygen': 98.0,
            'temperature': 98.6,
            'steps': 8000.0,
            'sleep_quality': 7.0,
            'stress_level': 3.0,
            'timestamp': datetime.now().timestamp()
        }
    
    def _get_empty_visualization_data(self) -> Dict[str, Any]:
        """Get empty visualization data structure"""
        return {
            'time_series': {},
            'summary_stats': {},
            'trends': {},
            'data_quality': {'total_records': 0}
        }
    
    def _get_empty_report(self) -> Dict[str, Any]:
        """Get empty report structure"""
        return {
            'period': {'start': None, 'end': None, 'days': 0},
            'metrics_summary': {},
            'health_score': 0,
            'insights': [],
            'recommendations': []
        }
    
    def _calculate_metric_trend(self, values: pd.Series) -> str:
        """Calculate trend for a specific metric"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_normal_percentage(self, values: pd.Series, metric: str) -> float:
        """Calculate percentage of values within normal range"""
        if metric not in self.normal_ranges:
            return 100.0
        
        min_val, max_val = self.normal_ranges[metric]
        within_normal = ((values >= min_val) & (values <= max_val)).sum()
        
        return float(within_normal / len(values) * 100) if len(values) > 0 else 0.0
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate key insights from the data"""
        insights = []
        
        # Add insights based on data analysis
        for column in self.feature_columns:
            if column in df.columns:
                values = df[column].dropna()
                if not values.empty:
                    avg_value = values.mean()
                    
                    if column == 'steps':
                        if avg_value < 6000:
                            insights.append(f"Your average daily steps ({avg_value:.0f}) is below recommended levels")
                        elif avg_value > 12000:
                            insights.append(f"Excellent activity level with {avg_value:.0f} average daily steps")
                    
                    elif column == 'sleep_quality':
                        if avg_value < 6:
                            insights.append(f"Sleep quality ({avg_value:.1f}/10) could be improved")
                        elif avg_value > 8:
                            insights.append(f"Excellent sleep quality ({avg_value:.1f}/10)")
                    
                    elif column == 'stress_level':
                        if avg_value > 6:
                            insights.append(f"Stress levels ({avg_value:.1f}/10) are elevated")
                        elif avg_value < 3:
                            insights.append(f"Good stress management ({avg_value:.1f}/10)")
        
        return insights
    
    def _calculate_health_score(self, df: pd.DataFrame) -> float:
        """Calculate overall health score (0-100)"""
        scores = []
        
        for column in self.feature_columns:
            if column in df.columns and column in self.normal_ranges:
                values = df[column].dropna()
                if not values.empty:
                    min_val, max_val = self.normal_ranges[column]
                    within_normal_pct = self._calculate_normal_percentage(values, column)
                    scores.append(within_normal_pct)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _generate_report_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on report data"""
        recommendations = []
        
        # Analyze each metric and provide recommendations
        for column in self.feature_columns:
            if column in df.columns:
                values = df[column].dropna()
                if not values.empty:
                    avg_value = values.mean()
                    normal_pct = self._calculate_normal_percentage(values, column)
                    
                    if normal_pct < 80:  # Less than 80% within normal range
                        if column == 'heart_rate':
                            recommendations.append("Consider cardiovascular health monitoring")
                        elif column == 'blood_oxygen':
                            recommendations.append("Monitor respiratory health closely")
                        elif column == 'steps':
                            recommendations.append("Increase daily physical activity")
                        elif column == 'sleep_quality':
                            recommendations.append("Focus on improving sleep hygiene")
                        elif column == 'stress_level':
                            recommendations.append("Implement stress management techniques")
        
        return recommendations
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between metrics"""
        correlations = {}
        
        numeric_columns = [col for col in self.feature_columns if col in df.columns]
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            
            # Extract interesting correlations
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i < j:  # Avoid duplicates
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.3:  # Only significant correlations
                            correlations[f"{col1}_vs_{col2}"] = float(corr_value)
        
        return correlations
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in the data"""
        patterns = {}
        
        if 'hour' in df.columns:
            # Hourly patterns
            hourly_avg = df.groupby('hour')[self.feature_columns].mean()
            patterns['hourly_patterns'] = hourly_avg.to_dict()
        
        if 'day_of_week' in df.columns:
            # Weekly patterns
            weekly_avg = df.groupby('day_of_week')[self.feature_columns].mean()
            patterns['weekly_patterns'] = weekly_avg.to_dict()
        
        return patterns
    
    def _analyze_anomaly_frequency(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze frequency of anomalies"""
        anomaly_counts = {}
        
        for column in self.feature_columns:
            if column in df.columns and column in self.normal_ranges:
                min_val, max_val = self.normal_ranges[column]
                anomalies = ((df[column] < min_val) | (df[column] > max_val)).sum()
                anomaly_counts[column] = int(anomalies)
        
        return anomaly_counts
    
    def _analyze_circadian_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze circadian patterns in health metrics"""
        if 'hour' not in df.columns:
            return {}
        
        circadian_patterns = {}
        
        for column in self.feature_columns:
            if column in df.columns:
                hourly_avg = df.groupby('hour')[column].mean()
                
                # Find peak and trough hours
                peak_hour = hourly_avg.idxmax()
                trough_hour = hourly_avg.idxmin()
                
                circadian_patterns[column] = {
                    'peak_hour': int(peak_hour),
                    'trough_hour': int(trough_hour),
                    'peak_value': float(hourly_avg[peak_hour]),
                    'trough_value': float(hourly_avg[trough_hour])
                }
        
        return circadian_patterns
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly patterns in health metrics"""
        if 'day_of_week' not in df.columns:
            return {}
        
        weekly_patterns = {}
        
        for column in self.feature_columns:
            if column in df.columns:
                daily_avg = df.groupby('day_of_week')[column].mean()
                
                weekly_patterns[column] = {
                    'weekday_avg': float(daily_avg[:5].mean()),  # Mon-Fri
                    'weekend_avg': float(daily_avg[5:].mean()),  # Sat-Sun
                    'daily_averages': daily_avg.to_dict()
                }
        
        return weekly_patterns
    
    def _generate_predictive_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate predictive insights based on trends"""
        insights = []
        
        for column in self.feature_columns:
            if column in df.columns and len(df) > 7:  # Need sufficient data
                values = df[column].dropna()
                if len(values) > 7:
                    # Simple trend prediction
                    recent_trend = self._calculate_metric_trend(values.tail(7))
                    
                    if recent_trend == 'increasing':
                        if column == 'stress_level':
                            insights.append(f"Stress levels are trending upward - consider stress management")
                        elif column == 'heart_rate':
                            insights.append(f"Heart rate is trending higher - monitor cardiovascular health")
                    elif recent_trend == 'decreasing':
                        if column == 'steps':
                            insights.append(f"Activity levels are declining - increase daily movement")
                        elif column == 'sleep_quality':
                            insights.append(f"Sleep quality is declining - review sleep habits")
        
        return insights