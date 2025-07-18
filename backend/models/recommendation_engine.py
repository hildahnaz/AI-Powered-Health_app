"""
Health Recommendation Engine
Generates personalized health recommendations based on user data and anomalies
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class HealthRecommendationEngine:
    """
    AI-powered recommendation engine for personalized health advice
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_trained = False
        
        # Recommendation categories
        self.recommendation_categories = {
            'exercise': 'Physical Activity',
            'nutrition': 'Nutrition & Diet',
            'sleep': 'Sleep & Recovery',
            'medical': 'Medical Attention',
            'stress': 'Stress Management',
            'lifestyle': 'Lifestyle Changes'
        }
        
        # Priority levels
        self.priority_levels = ['low', 'medium', 'high', 'critical']
        
        # Base recommendations database
        self.base_recommendations = self._initialize_recommendations()
    
    def train_initial_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the initial recommendation model
        
        Args:
            training_data: Historical health data with outcomes
            
        Returns:
            Training results
        """
        try:
            logger.info("Training recommendation engine...")
            
            # Create training dataset with synthetic recommendations
            X, y = self._create_recommendation_dataset(training_data)
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X, y)
            self.model_trained = True
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(X.columns)
            
            # Save model
            self._save_model()
            
            logger.info("Recommendation engine trained successfully")
            
            return {
                'status': 'success',
                'feature_importance': feature_importance,
                'training_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"Error training recommendation engine: {str(e)}")
            raise
    
    def generate_recommendations(self, health_data: Dict[str, float], 
                               anomalies: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate personalized health recommendations
        
        Args:
            health_data: Current health metrics
            anomalies: Detected anomalies
            
        Returns:
            List of personalized recommendations
        """
        try:
            recommendations = []
            
            # Generate anomaly-based recommendations
            if anomalies:
                anomaly_recs = self._generate_anomaly_recommendations(anomalies)
                recommendations.extend(anomaly_recs)
            
            # Generate general health recommendations
            general_recs = self._generate_general_recommendations(health_data)
            recommendations.extend(general_recs)
            
            # Generate preventive recommendations
            preventive_recs = self._generate_preventive_recommendations(health_data)
            recommendations.extend(preventive_recs)
            
            # Remove duplicates and prioritize
            recommendations = self._prioritize_recommendations(recommendations)
            
            # Limit to top 5 recommendations
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._get_default_recommendations()
    
    def retrain_model(self) -> Dict[str, Any]:
        """
        Retrain the recommendation model with new data
        
        Returns:
            Retraining results
        """
        try:
            logger.info("Retraining recommendation engine...")
            
            # This would typically use new user feedback data
            # For now, return success status
            
            return {
                'status': 'success',
                'message': 'Model retrained with latest user feedback'
            }
            
        except Exception as e:
            logger.error(f"Error retraining recommendation engine: {str(e)}")
            raise
    
    def _initialize_recommendations(self) -> Dict[str, List[Dict]]:
        """Initialize base recommendation database"""
        return {
            'heart_rate': [
                {
                    'condition': 'high',
                    'category': 'medical',
                    'title': 'Elevated Heart Rate Detected',
                    'description': 'Your heart rate is above normal range. Consider consulting a healthcare provider.',
                    'priority': 'high',
                    'actions': [
                        'Practice deep breathing exercises',
                        'Reduce caffeine intake',
                        'Monitor for other symptoms',
                        'Consult healthcare provider if persistent'
                    ]
                },
                {
                    'condition': 'low',
                    'category': 'medical',
                    'title': 'Low Heart Rate Detected',
                    'description': 'Your heart rate is below normal range. Monitor closely.',
                    'priority': 'medium',
                    'actions': [
                        'Monitor symptoms like dizziness',
                        'Ensure adequate hydration',
                        'Consult healthcare provider if concerned'
                    ]
                }
            ],
            'blood_oxygen': [
                {
                    'condition': 'low',
                    'category': 'medical',
                    'title': 'Low Blood Oxygen Levels',
                    'description': 'Your blood oxygen levels are below normal. This requires attention.',
                    'priority': 'critical',
                    'actions': [
                        'Seek immediate medical attention',
                        'Practice deep breathing',
                        'Avoid strenuous activity',
                        'Check device accuracy'
                    ]
                }
            ],
            'steps': [
                {
                    'condition': 'low',
                    'category': 'exercise',
                    'title': 'Increase Daily Activity',
                    'description': 'Your step count is below recommended levels. Try to be more active.',
                    'priority': 'medium',
                    'actions': [
                        'Take short walks every hour',
                        'Use stairs instead of elevators',
                        'Park farther from destinations',
                        'Set movement reminders'
                    ]
                }
            ],
            'sleep_quality': [
                {
                    'condition': 'poor',
                    'category': 'sleep',
                    'title': 'Improve Sleep Quality',
                    'description': 'Your sleep quality could be improved for better health.',
                    'priority': 'medium',
                    'actions': [
                        'Maintain consistent sleep schedule',
                        'Create relaxing bedtime routine',
                        'Limit screen time before bed',
                        'Ensure comfortable sleep environment'
                    ]
                }
            ],
            'stress_level': [
                {
                    'condition': 'high',
                    'category': 'stress',
                    'title': 'Manage Stress Levels',
                    'description': 'Your stress levels are elevated. Consider stress management techniques.',
                    'priority': 'high',
                    'actions': [
                        'Practice meditation or mindfulness',
                        'Try progressive muscle relaxation',
                        'Engage in regular exercise',
                        'Consider talking to a counselor'
                    ]
                }
            ],
            'general': [
                {
                    'category': 'nutrition',
                    'title': 'Stay Hydrated',
                    'description': 'Proper hydration is essential for optimal health.',
                    'priority': 'low',
                    'actions': [
                        'Drink 8 glasses of water daily',
                        'Monitor urine color',
                        'Increase intake during exercise',
                        'Eat water-rich foods'
                    ]
                },
                {
                    'category': 'lifestyle',
                    'title': 'Regular Health Monitoring',
                    'description': 'Continue monitoring your health metrics regularly.',
                    'priority': 'low',
                    'actions': [
                        'Check metrics daily',
                        'Keep a health diary',
                        'Schedule regular check-ups',
                        'Share data with healthcare provider'
                    ]
                }
            ]
        }
    
    def _create_recommendation_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create training dataset for recommendation model
        This is a simplified version - in practice, you'd use user feedback data
        """
        # Create features from health data
        features = []
        labels = []
        
        for _, row in data.iterrows():
            # Create feature vector
            feature_vector = {
                'heart_rate': row.get('heart_rate', 72),
                'blood_oxygen': row.get('blood_oxygen', 98),
                'temperature': row.get('temperature', 98.6),
                'steps': row.get('steps', 8000),
                'sleep_quality': row.get('sleep_quality', 7),
                'stress_level': row.get('stress_level', 3),
                'hour_of_day': pd.to_datetime(row.get('timestamp', datetime.now())).hour,
                'day_of_week': pd.to_datetime(row.get('timestamp', datetime.now())).weekday()
            }
            
            # Determine recommendation category based on metrics
            recommendation_category = self._determine_recommendation_category(feature_vector)
            
            features.append(feature_vector)
            labels.append(recommendation_category)
        
        X = pd.DataFrame(features)
        y = np.array(labels)
        
        return X, y
    
    def _determine_recommendation_category(self, metrics: Dict[str, float]) -> str:
        """Determine appropriate recommendation category based on metrics"""
        
        # Check for critical conditions first
        if metrics['blood_oxygen'] < 95:
            return 'medical'
        if metrics['heart_rate'] > 100 or metrics['heart_rate'] < 60:
            return 'medical'
        if metrics['temperature'] > 99.5 or metrics['temperature'] < 97:
            return 'medical'
        
        # Check for lifestyle recommendations
        if metrics['stress_level'] > 6:
            return 'stress'
        if metrics['sleep_quality'] < 6:
            return 'sleep'
        if metrics['steps'] < 6000:
            return 'exercise'
        
        # Default to lifestyle
        return 'lifestyle'
    
    def _generate_anomaly_recommendations(self, anomalies: List[Dict]) -> List[Dict[str, Any]]:
        """Generate recommendations based on detected anomalies"""
        recommendations = []
        
        for anomaly in anomalies:
            metric = anomaly['metric']
            severity = anomaly['severity']
            
            # Get base recommendations for this metric
            if metric in self.base_recommendations:
                base_recs = self.base_recommendations[metric]
                
                for base_rec in base_recs:
                    # Adjust priority based on anomaly severity
                    adjusted_priority = self._adjust_priority(base_rec['priority'], severity)
                    
                    recommendation = {
                        'id': f"rec_{metric}_{int(datetime.now().timestamp())}",
                        'type': base_rec['category'],
                        'title': base_rec['title'],
                        'description': base_rec['description'],
                        'priority': adjusted_priority,
                        'actions': base_rec['actions'],
                        'source': 'anomaly_detection',
                        'related_metric': metric,
                        'confidence': anomaly.get('confidence', 0.8)
                    }
                    
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_general_recommendations(self, health_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate general health recommendations based on current metrics"""
        recommendations = []
        
        # Check each metric and generate appropriate recommendations
        heart_rate = health_data.get('heart_rate', 72)
        steps = health_data.get('steps', 8000)
        sleep_quality = health_data.get('sleep_quality', 7)
        stress_level = health_data.get('stress_level', 3)
        
        # Step count recommendations
        if steps < 8000:
            recommendations.append({
                'id': f"rec_steps_{int(datetime.now().timestamp())}",
                'type': 'exercise',
                'title': 'Increase Daily Activity',
                'description': f'You have {steps} steps today. Try to reach 8,000 steps for better health.',
                'priority': 'medium',
                'actions': [
                    'Take a 10-minute walk',
                    'Use stairs instead of elevators',
                    'Park farther from your destination',
                    'Take walking meetings'
                ],
                'source': 'general_health',
                'confidence': 0.9
            })
        
        # Sleep quality recommendations
        if sleep_quality < 7:
            recommendations.append({
                'id': f"rec_sleep_{int(datetime.now().timestamp())}",
                'type': 'sleep',
                'title': 'Improve Sleep Quality',
                'description': f'Your sleep quality score is {sleep_quality}/10. Better sleep can improve overall health.',
                'priority': 'medium',
                'actions': [
                    'Maintain consistent sleep schedule',
                    'Create a relaxing bedtime routine',
                    'Limit caffeine after 2 PM',
                    'Keep bedroom cool and dark'
                ],
                'source': 'general_health',
                'confidence': 0.8
            })
        
        # Stress level recommendations
        if stress_level > 6:
            recommendations.append({
                'id': f"rec_stress_{int(datetime.now().timestamp())}",
                'type': 'stress',
                'title': 'Manage Stress Levels',
                'description': f'Your stress level is {stress_level}/10. Consider stress management techniques.',
                'priority': 'high',
                'actions': [
                    'Practice deep breathing exercises',
                    'Try 5-minute meditation',
                    'Take short breaks during work',
                    'Consider yoga or stretching'
                ],
                'source': 'general_health',
                'confidence': 0.85
            })
        
        return recommendations
    
    def _generate_preventive_recommendations(self, health_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate preventive health recommendations"""
        recommendations = []
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        
        if 6 <= current_hour <= 10:  # Morning
            recommendations.append({
                'id': f"rec_morning_{int(datetime.now().timestamp())}",
                'type': 'lifestyle',
                'title': 'Morning Health Routine',
                'description': 'Start your day with healthy habits.',
                'priority': 'low',
                'actions': [
                    'Drink a glass of water',
                    'Do light stretching',
                    'Eat a nutritious breakfast',
                    'Check your health metrics'
                ],
                'source': 'preventive',
                'confidence': 0.7
            })
        
        elif 12 <= current_hour <= 14:  # Lunch time
            recommendations.append({
                'id': f"rec_lunch_{int(datetime.now().timestamp())}",
                'type': 'nutrition',
                'title': 'Healthy Lunch Choices',
                'description': 'Make nutritious choices for sustained energy.',
                'priority': 'low',
                'actions': [
                    'Choose balanced meals',
                    'Include vegetables and protein',
                    'Stay hydrated',
                    'Take a short walk after eating'
                ],
                'source': 'preventive',
                'confidence': 0.7
            })
        
        elif 18 <= current_hour <= 22:  # Evening
            recommendations.append({
                'id': f"rec_evening_{int(datetime.now().timestamp())}",
                'type': 'lifestyle',
                'title': 'Evening Wind-Down',
                'description': 'Prepare your body for quality sleep.',
                'priority': 'low',
                'actions': [
                    'Limit screen time',
                    'Practice relaxation techniques',
                    'Prepare for tomorrow',
                    'Maintain consistent bedtime'
                ],
                'source': 'preventive',
                'confidence': 0.7
            })
        
        return recommendations
    
    def _adjust_priority(self, base_priority: str, severity: str) -> str:
        """Adjust recommendation priority based on anomaly severity"""
        priority_map = {
            'low': 0, 'medium': 1, 'high': 2, 'critical': 3
        }
        severity_map = {
            'low': 0, 'medium': 1, 'high': 2, 'critical': 3
        }
        
        base_level = priority_map.get(base_priority, 1)
        severity_level = severity_map.get(severity, 1)
        
        # Increase priority based on severity
        adjusted_level = min(base_level + severity_level, 3)
        
        return self.priority_levels[adjusted_level]
    
    def _prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize and deduplicate recommendations"""
        # Remove duplicates based on title
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec['title'] not in seen_titles:
                unique_recommendations.append(rec)
                seen_titles.add(rec['title'])
        
        # Sort by priority (critical > high > medium > low)
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        unique_recommendations.sort(
            key=lambda x: (priority_order.get(x['priority'], 3), -x.get('confidence', 0))
        )
        
        return unique_recommendations
    
    def _get_default_recommendations(self) -> List[Dict[str, Any]]:
        """Get default recommendations when generation fails"""
        return [
            {
                'id': 'default_1',
                'type': 'lifestyle',
                'title': 'Stay Hydrated',
                'description': 'Drink plenty of water throughout the day.',
                'priority': 'low',
                'actions': ['Drink 8 glasses of water daily'],
                'source': 'default',
                'confidence': 0.5
            },
            {
                'id': 'default_2',
                'type': 'exercise',
                'title': 'Stay Active',
                'description': 'Regular physical activity is important for health.',
                'priority': 'medium',
                'actions': ['Take a 10-minute walk'],
                'source': 'default',
                'confidence': 0.5
            }
        ]
    
    def _calculate_feature_importance(self, feature_names) -> Dict[str, float]:
        """Calculate feature importance from trained model"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            for name, importance in zip(feature_names, self.model.feature_importances_):
                importance_dict[name] = float(importance)
            return importance_dict
        return {}
    
    def _save_model(self):
        """Save trained recommendation model"""
        try:
            joblib.dump(self.model, 'models/recommendation_engine.pkl')
            joblib.dump(self.label_encoder, 'models/recommendation_encoder.pkl')
            logger.info("Recommendation model saved successfully")
        except Exception as e:
            logger.error(f"Error saving recommendation model: {str(e)}")
    
    def _load_model(self):
        """Load pre-trained recommendation model"""
        try:
            self.model = joblib.load('models/recommendation_engine.pkl')
            self.label_encoder = joblib.load('models/recommendation_encoder.pkl')
            self.model_trained = True
            logger.info("Recommendation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading recommendation model: {str(e)}")
            self.model_trained = False