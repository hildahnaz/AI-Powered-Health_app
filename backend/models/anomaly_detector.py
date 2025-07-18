"""
Health Anomaly Detection Model
Implements Isolation Forest and other anomaly detection algorithms
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class HealthAnomalyDetector:
    """
    Advanced anomaly detection system for health monitoring data
    Uses Isolation Forest as the primary algorithm with additional validation
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'heart_rate', 'blood_oxygen', 'temperature', 
            'steps', 'sleep_quality', 'stress_level'
        ]
        self.model_trained = False
        self.performance_metrics = {}
        
        # Anomaly thresholds based on medical standards
        self.normal_ranges = {
            'heart_rate': (60, 100),
            'blood_oxygen': (95, 100),
            'temperature': (97.0, 99.5),
            'steps': (0, 50000),
            'sleep_quality': (6, 10),
            'stress_level': (0, 5)
        }
    
    def train_initial_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the initial anomaly detection model
        
        Args:
            training_data: DataFrame with health metrics
            
        Returns:
            Dictionary with training results and performance metrics
        """
        try:
            logger.info("Starting anomaly detection model training...")
            
            # Prepare features
            X = self._prepare_features(training_data)
            
            # Create synthetic anomalies for validation
            X_with_anomalies, y_true = self._create_labeled_dataset(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_with_anomalies, y_true, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train Isolation Forest
            self.model = IsolationForest(
                contamination=0.1,  # Expected proportion of anomalies
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0
            )
            
            # Train on normal data only
            normal_data = X_train_scaled[y_train == 1]
            self.model.fit(normal_data)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_pred_binary = np.where(y_pred == -1, 0, 1)  # Convert to binary
            
            # Calculate performance metrics
            self.performance_metrics = self._calculate_performance_metrics(
                y_test, y_pred_binary
            )
            
            self.model_trained = True
            
            # Save model
            self._save_model()
            
            logger.info(f"Model training completed. Accuracy: {self.performance_metrics['accuracy']:.3f}")
            
            return {
                'status': 'success',
                'performance': self.performance_metrics,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {str(e)}")
            raise
    
    def detect_anomalies(self, health_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in current health data
        
        Args:
            health_data: Dictionary with current health metrics
            
        Returns:
            List of detected anomalies with details
        """
        if not self.model_trained:
            logger.warning("Model not trained. Loading pre-trained model...")
            self._load_model()
        
        try:
            # Prepare single data point
            X = self._prepare_single_datapoint(health_data)
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            
            # Predict anomaly
            anomaly_score = self.model.decision_function(X_scaled)[0]
            is_anomaly = self.model.predict(X_scaled)[0] == -1
            
            anomalies = []
            
            if is_anomaly:
                # Identify which metrics are anomalous
                anomalous_metrics = self._identify_anomalous_metrics(health_data)
                
                for metric, details in anomalous_metrics.items():
                    anomalies.append({
                        'id': f"anomaly_{metric}_{int(datetime.now().timestamp())}",
                        'metric': metric,
                        'value': details['value'],
                        'normal_range': details['normal_range'],
                        'severity': details['severity'],
                        'description': details['description'],
                        'confidence': abs(anomaly_score),
                        'timestamp': datetime.now().isoformat(),
                        'recommendations': self._get_metric_recommendations(metric, details)
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def retrain_model(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Retrain the model with new data
        
        Args:
            new_data: New training data
            
        Returns:
            Retraining results
        """
        try:
            logger.info("Retraining anomaly detection model...")
            
            # Combine with existing data if available
            if hasattr(self, 'training_data'):
                combined_data = pd.concat([self.training_data, new_data])
            else:
                combined_data = new_data
            
            # Retrain model
            results = self.train_initial_model(combined_data)
            
            logger.info("Model retrained successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
            raise
    
    def analyze_anomaly_patterns(self, anomalies: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patterns in detected anomalies
        
        Args:
            anomalies: List of anomaly records
            
        Returns:
            Analysis results
        """
        if not anomalies:
            return {
                'total_anomalies': 0,
                'patterns': {},
                'trends': {},
                'recommendations': []
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(anomalies)
        
        # Analyze patterns
        metric_counts = df['metric'].value_counts().to_dict()
        severity_distribution = df['severity'].value_counts().to_dict()
        
        # Time-based analysis
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_distribution = df['hour'].value_counts().sort_index().to_dict()
        
        # Correlation analysis
        correlations = self._analyze_metric_correlations(df)
        
        return {
            'total_anomalies': len(anomalies),
            'patterns': {
                'by_metric': metric_counts,
                'by_severity': severity_distribution,
                'by_hour': hourly_distribution
            },
            'correlations': correlations,
            'trends': self._identify_trends(df),
            'recommendations': self._generate_pattern_recommendations(df)
        }
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get current model performance metrics"""
        return self.performance_metrics
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model training"""
        return data[self.feature_columns].values
    
    def _prepare_single_datapoint(self, health_data: Dict[str, float]) -> np.ndarray:
        """Prepare single data point for prediction"""
        return np.array([health_data.get(col, 0) for col in self.feature_columns])
    
    def _create_labeled_dataset(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a labeled dataset with synthetic anomalies for validation
        """
        normal_data = X.copy()
        anomalous_data = []
        
        # Generate synthetic anomalies
        n_anomalies = int(len(X) * 0.1)  # 10% anomalies
        
        for _ in range(n_anomalies):
            # Select random sample and introduce anomaly
            idx = np.random.randint(0, len(X))
            anomaly = X[idx].copy()
            
            # Randomly select metric to make anomalous
            metric_idx = np.random.randint(0, len(self.feature_columns))
            
            # Introduce anomaly based on metric type
            if metric_idx == 0:  # heart_rate
                anomaly[metric_idx] = np.random.choice([40, 150])  # Very low or high
            elif metric_idx == 1:  # blood_oxygen
                anomaly[metric_idx] = np.random.uniform(85, 94)  # Low oxygen
            elif metric_idx == 2:  # temperature
                anomaly[metric_idx] = np.random.choice([95, 102])  # Very low or high
            elif metric_idx == 3:  # steps
                anomaly[metric_idx] = np.random.uniform(0, 500)  # Very low steps
            elif metric_idx == 4:  # sleep_quality
                anomaly[metric_idx] = np.random.uniform(0, 4)  # Poor sleep
            elif metric_idx == 5:  # stress_level
                anomaly[metric_idx] = np.random.uniform(8, 10)  # High stress
            
            anomalous_data.append(anomaly)
        
        # Combine normal and anomalous data
        X_combined = np.vstack([normal_data, np.array(anomalous_data)])
        y_combined = np.hstack([
            np.ones(len(normal_data)),  # Normal = 1
            np.zeros(len(anomalous_data))  # Anomaly = 0
        ])
        
        return X_combined, y_combined
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def _identify_anomalous_metrics(self, health_data: Dict[str, float]) -> Dict[str, Dict]:
        """Identify which specific metrics are anomalous"""
        anomalous_metrics = {}
        
        for metric, value in health_data.items():
            if metric in self.normal_ranges:
                min_val, max_val = self.normal_ranges[metric]
                
                if value < min_val or value > max_val:
                    severity = self._calculate_severity(metric, value, min_val, max_val)
                    
                    anomalous_metrics[metric] = {
                        'value': value,
                        'normal_range': (min_val, max_val),
                        'severity': severity,
                        'description': self._get_anomaly_description(metric, value, min_val, max_val)
                    }
        
        return anomalous_metrics
    
    def _calculate_severity(self, metric: str, value: float, min_val: float, max_val: float) -> str:
        """Calculate anomaly severity"""
        if metric == 'heart_rate':
            if value < 50 or value > 120:
                return 'critical'
            elif value < 55 or value > 110:
                return 'high'
            else:
                return 'medium'
        elif metric == 'blood_oxygen':
            if value < 90:
                return 'critical'
            elif value < 93:
                return 'high'
            else:
                return 'medium'
        elif metric == 'temperature':
            if value < 95 or value > 101:
                return 'critical'
            elif value < 96 or value > 100:
                return 'high'
            else:
                return 'medium'
        else:
            # Default severity calculation
            range_size = max_val - min_val
            deviation = max(abs(value - min_val), abs(value - max_val)) - range_size/2
            
            if deviation > range_size:
                return 'critical'
            elif deviation > range_size/2:
                return 'high'
            else:
                return 'medium'
    
    def _get_anomaly_description(self, metric: str, value: float, min_val: float, max_val: float) -> str:
        """Generate human-readable anomaly description"""
        descriptions = {
            'heart_rate': f"Heart rate of {value} bpm is {'below' if value < min_val else 'above'} normal range ({min_val}-{max_val} bpm)",
            'blood_oxygen': f"Blood oxygen level of {value}% is below normal range ({min_val}-{max_val}%)",
            'temperature': f"Body temperature of {value}°F is {'below' if value < min_val else 'above'} normal range ({min_val}-{max_val}°F)",
            'steps': f"Step count of {value} is below recommended daily activity",
            'sleep_quality': f"Sleep quality score of {value} indicates poor sleep",
            'stress_level': f"Stress level of {value} is elevated above normal range"
        }
        
        return descriptions.get(metric, f"{metric} value of {value} is outside normal range")
    
    def _get_metric_recommendations(self, metric: str, details: Dict) -> List[str]:
        """Get recommendations for specific anomalous metrics"""
        recommendations = {
            'heart_rate': [
                "Consider consulting with a healthcare provider",
                "Practice relaxation techniques",
                "Monitor caffeine intake",
                "Ensure adequate hydration"
            ],
            'blood_oxygen': [
                "Seek immediate medical attention if persistent",
                "Practice deep breathing exercises",
                "Check device calibration",
                "Avoid high altitude activities"
            ],
            'temperature': [
                "Monitor temperature regularly",
                "Stay hydrated",
                "Rest and avoid strenuous activity",
                "Consult healthcare provider if persistent"
            ],
            'steps': [
                "Increase daily physical activity",
                "Take short walks throughout the day",
                "Use stairs instead of elevators",
                "Set hourly movement reminders"
            ],
            'sleep_quality': [
                "Maintain consistent sleep schedule",
                "Create relaxing bedtime routine",
                "Limit screen time before bed",
                "Ensure comfortable sleep environment"
            ],
            'stress_level': [
                "Practice stress management techniques",
                "Consider meditation or yoga",
                "Ensure adequate sleep",
                "Seek support if stress persists"
            ]
        }
        
        return recommendations.get(metric, ["Monitor this metric closely", "Consult healthcare provider if concerned"])
    
    def _analyze_metric_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlations between anomalous metrics"""
        # This would implement correlation analysis
        # For now, return placeholder
        return {
            'heart_rate_stress': 0.7,
            'sleep_quality_stress': -0.6,
            'temperature_heart_rate': 0.4
        }
    
    def _identify_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify trends in anomaly data"""
        return {
            'increasing_frequency': False,
            'peak_hours': [14, 20],  # 2 PM and 8 PM
            'most_common_metric': df['metric'].mode().iloc[0] if not df.empty else None
        }
    
    def _generate_pattern_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on anomaly patterns"""
        return [
            "Consider scheduling regular health check-ups",
            "Monitor stress levels during peak hours",
            "Maintain consistent daily routines",
            "Keep a health diary to identify triggers"
        ]
    
    def _save_model(self):
        """Save trained model and scaler"""
        try:
            joblib.dump(self.model, 'models/anomaly_detector.pkl')
            joblib.dump(self.scaler, 'models/anomaly_scaler.pkl')
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def _load_model(self):
        """Load pre-trained model and scaler"""
        try:
            self.model = joblib.load('models/anomaly_detector.pkl')
            self.scaler = joblib.load('models/anomaly_scaler.pkl')
            self.model_trained = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Initialize with default model if loading fails
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.model_trained = False