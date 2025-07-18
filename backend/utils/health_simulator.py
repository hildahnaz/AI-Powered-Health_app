"""
Health Data Simulator
Generates realistic health monitoring data for testing and training
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random
import logging

logger = logging.getLogger(__name__)

class HealthDataSimulator:
    """
    Generates realistic health monitoring data that simulates wearable device outputs
    """
    
    def __init__(self):
        # Baseline health metrics for a typical healthy adult
        self.baseline_metrics = {
            'heart_rate': 72,
            'blood_oxygen': 98,
            'temperature': 98.6,
            'sleep_quality': 7.5,
            'stress_level': 3.0
        }
        
        # Daily patterns and variations
        self.circadian_patterns = {
            'heart_rate': {
                'morning_increase': 10,
                'afternoon_peak': 15,
                'evening_decrease': -5,
                'night_decrease': -15
            },
            'temperature': {
                'morning_low': -0.5,
                'afternoon_peak': 0.3,
                'evening_normal': 0,
                'night_low': -0.8
            },
            'stress_level': {
                'morning_moderate': 1,
                'work_hours_high': 2,
                'evening_decrease': -1,
                'night_low': -2
            }
        }
        
        # Activity patterns
        self.activity_patterns = {
            'sedentary': {'steps_per_hour': 50, 'hr_increase': 0},
            'light': {'steps_per_hour': 200, 'hr_increase': 5},
            'moderate': {'steps_per_hour': 500, 'hr_increase': 15},
            'vigorous': {'steps_per_hour': 800, 'hr_increase': 30}
        }
        
        # Current state tracking
        self.current_steps = 0
        self.current_day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def generate_current_metrics(self) -> Dict[str, float]:
        """
        Generate current health metrics based on time of day and realistic patterns
        
        Returns:
            Dictionary with current health metrics
        """
        try:
            current_time = datetime.now()
            hour = current_time.hour
            
            # Generate base metrics with circadian variations
            metrics = {}
            
            # Heart rate with circadian pattern
            base_hr = self.baseline_metrics['heart_rate']
            hr_variation = self._get_circadian_variation('heart_rate', hour)
            activity_boost = self._get_activity_boost(hour)
            noise = np.random.normal(0, 3)  # Random variation
            
            metrics['heart_rate'] = max(50, min(120, base_hr + hr_variation + activity_boost + noise))
            
            # Blood oxygen (relatively stable with slight variations)
            base_o2 = self.baseline_metrics['blood_oxygen']
            o2_noise = np.random.normal(0, 0.8)
            activity_impact = -1 if activity_boost > 20 else 0  # Slight decrease during intense activity
            
            metrics['blood_oxygen'] = max(94, min(100, base_o2 + o2_noise + activity_impact))
            
            # Temperature with circadian pattern
            base_temp = self.baseline_metrics['temperature']
            temp_variation = self._get_circadian_variation('temperature', hour)
            temp_noise = np.random.normal(0, 0.2)
            
            metrics['temperature'] = max(96.5, min(100.5, base_temp + temp_variation + temp_noise))
            
            # Steps (cumulative for the day)
            if current_time.date() != self.current_day_start.date():
                self.current_steps = 0  # Reset for new day
                self.current_day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Add steps based on activity level and time
            steps_increment = self._calculate_steps_increment(hour)
            self.current_steps += steps_increment
            metrics['steps'] = min(50000, self.current_steps)  # Cap at reasonable maximum
            
            # Sleep quality (only relevant during sleep hours, otherwise use previous day's score)
            if 23 <= hour or hour <= 6:
                sleep_stage = self._determine_sleep_stage(hour)
                metrics['sleep_quality'] = self._calculate_sleep_quality(sleep_stage, hour)
            else:
                # Use baseline with slight random variation for daytime
                metrics['sleep_quality'] = max(0, min(10, self.baseline_metrics['sleep_quality'] + np.random.normal(0, 0.5)))
            
            # Stress level with circadian and situational patterns
            base_stress = self.baseline_metrics['stress_level']
            stress_variation = self._get_circadian_variation('stress_level', hour)
            situational_stress = self._get_situational_stress(hour, current_time.weekday())
            stress_noise = np.random.normal(0, 0.5)
            
            metrics['stress_level'] = max(0, min(10, base_stress + stress_variation + situational_stress + stress_noise))
            
            # Round values appropriately
            metrics['heart_rate'] = round(metrics['heart_rate'])
            metrics['blood_oxygen'] = round(metrics['blood_oxygen'], 1)
            metrics['temperature'] = round(metrics['temperature'], 1)
            metrics['steps'] = int(metrics['steps'])
            metrics['sleep_quality'] = round(metrics['sleep_quality'], 1)
            metrics['stress_level'] = round(metrics['stress_level'], 1)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating current metrics: {str(e)}")
            return self._get_default_metrics()
    
    def generate_training_dataset(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Generate training dataset with realistic health data over specified days
        
        Args:
            days: Number of days to generate data for
            
        Returns:
            List of health data records
        """
        try:
            dataset = []
            start_date = datetime.now() - timedelta(days=days)
            
            # Reset daily tracking
            daily_steps = 0
            
            for day in range(days):
                current_date = start_date + timedelta(days=day)
                daily_steps = 0  # Reset steps for each day
                
                # Generate data points every hour
                for hour in range(24):
                    timestamp = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # Generate metrics for this time point
                    metrics = self._generate_historical_metrics(timestamp, daily_steps)
                    daily_steps = metrics['steps']  # Update cumulative steps
                    
                    # Add some realistic anomalies (5% chance)
                    if random.random() < 0.05:
                        metrics = self._introduce_realistic_anomaly(metrics, timestamp)
                    
                    dataset.append({
                        'timestamp': timestamp.isoformat(),
                        **metrics
                    })
            
            logger.info(f"Generated training dataset with {len(dataset)} records over {days} days")
            return dataset
            
        except Exception as e:
            logger.error(f"Error generating training dataset: {str(e)}")
            return []
    
    def _get_circadian_variation(self, metric: str, hour: int) -> float:
        """Get circadian rhythm variation for a specific metric and hour"""
        if metric not in self.circadian_patterns:
            return 0
        
        patterns = self.circadian_patterns[metric]
        
        if 6 <= hour <= 10:  # Morning
            return patterns.get('morning_increase', 0) * (hour - 6) / 4
        elif 11 <= hour <= 16:  # Afternoon
            return patterns.get('afternoon_peak', 0)
        elif 17 <= hour <= 21:  # Evening
            return patterns.get('evening_decrease', 0) * (hour - 17) / 4
        else:  # Night
            return patterns.get('night_decrease', 0)
    
    def _get_activity_boost(self, hour: int) -> float:
        """Get activity-based heart rate boost based on time of day"""
        # Simulate typical daily activity patterns
        if 7 <= hour <= 8:  # Morning exercise
            return random.choice([0, 0, 0, 15, 25])  # 40% chance of morning workout
        elif 12 <= hour <= 13:  # Lunch walk
            return random.choice([0, 0, 5, 10])  # 50% chance of lunch activity
        elif 17 <= hour <= 19:  # Evening exercise
            return random.choice([0, 0, 0, 20, 30])  # 40% chance of evening workout
        elif 9 <= hour <= 17:  # Work hours - occasional activity
            return random.choice([0, 0, 0, 0, 5])  # 20% chance of light activity
        else:
            return 0
    
    def _calculate_steps_increment(self, hour: int) -> int:
        """Calculate steps increment based on hour and activity patterns"""
        # Different activity levels throughout the day
        if 7 <= hour <= 8:  # Morning routine
            base_steps = random.choice([100, 200, 300, 800, 1200])  # Varies by exercise
        elif 9 <= hour <= 17:  # Work hours
            base_steps = random.choice([50, 100, 150, 200])  # Light office activity
        elif 12 <= hour <= 13:  # Lunch break
            base_steps = random.choice([200, 400, 600])  # Lunch walk
        elif 17 <= hour <= 19:  # Evening activity
            base_steps = random.choice([100, 300, 500, 1000])  # Evening exercise
        elif 20 <= hour <= 22:  # Evening routine
            base_steps = random.choice([100, 200, 300])  # Light evening activity
        else:  # Night/early morning
            base_steps = random.choice([0, 10, 20])  # Minimal activity
        
        # Add some randomness
        return int(base_steps * random.uniform(0.8, 1.2))
    
    def _determine_sleep_stage(self, hour: int) -> str:
        """Determine sleep stage based on hour"""
        if hour >= 23 or hour <= 2:  # Early sleep
            return random.choice(['light', 'deep', 'deep', 'light'])
        elif 3 <= hour <= 5:  # Deep sleep period
            return random.choice(['deep', 'deep', 'rem', 'light'])
        elif 6 <= hour <= 7:  # Light sleep/waking
            return random.choice(['light', 'light', 'rem', 'awake'])
        else:
            return 'awake'
    
    def _calculate_sleep_quality(self, sleep_stage: str, hour: int) -> float:
        """Calculate sleep quality based on sleep stage"""
        stage_quality = {
            'deep': random.uniform(8.0, 9.5),
            'rem': random.uniform(7.0, 8.5),
            'light': random.uniform(6.0, 7.5),
            'awake': random.uniform(3.0, 5.0)
        }
        
        base_quality = stage_quality.get(sleep_stage, 7.0)
        
        # Add some variation based on time (quality decreases towards morning)
        if hour >= 5:
            base_quality *= 0.9
        
        return max(0, min(10, base_quality))
    
    def _get_situational_stress(self, hour: int, weekday: int) -> float:
        """Get situational stress based on time and day"""
        stress_boost = 0
        
        # Work hours stress (Monday-Friday)
        if weekday < 5 and 9 <= hour <= 17:
            stress_boost += random.uniform(1, 3)
        
        # Monday morning stress
        if weekday == 0 and 8 <= hour <= 10:
            stress_boost += random.uniform(0.5, 1.5)
        
        # Friday afternoon relief
        if weekday == 4 and 15 <= hour <= 17:
            stress_boost -= random.uniform(0.5, 1.0)
        
        # Evening relaxation
        if 19 <= hour <= 22:
            stress_boost -= random.uniform(0.5, 1.5)
        
        # Random situational stress
        if random.random() < 0.1:  # 10% chance of random stress event
            stress_boost += random.uniform(1, 3)
        
        return stress_boost
    
    def _generate_historical_metrics(self, timestamp: datetime, current_steps: int) -> Dict[str, float]:
        """Generate historical metrics for a specific timestamp"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Heart rate
        base_hr = self.baseline_metrics['heart_rate']
        hr_variation = self._get_circadian_variation('heart_rate', hour)
        activity_boost = self._get_activity_boost(hour)
        hr_noise = np.random.normal(0, 3)
        heart_rate = max(50, min(120, base_hr + hr_variation + activity_boost + hr_noise))
        
        # Blood oxygen
        base_o2 = self.baseline_metrics['blood_oxygen']
        o2_noise = np.random.normal(0, 0.8)
        activity_impact = -1 if activity_boost > 20 else 0
        blood_oxygen = max(94, min(100, base_o2 + o2_noise + activity_impact))
        
        # Temperature
        base_temp = self.baseline_metrics['temperature']
        temp_variation = self._get_circadian_variation('temperature', hour)
        temp_noise = np.random.normal(0, 0.2)
        temperature = max(96.5, min(100.5, base_temp + temp_variation + temp_noise))
        
        # Steps (cumulative)
        steps_increment = self._calculate_steps_increment(hour)
        steps = current_steps + steps_increment
        
        # Sleep quality
        if 23 <= hour or hour <= 6:
            sleep_stage = self._determine_sleep_stage(hour)
            sleep_quality = self._calculate_sleep_quality(sleep_stage, hour)
        else:
            sleep_quality = max(0, min(10, self.baseline_metrics['sleep_quality'] + np.random.normal(0, 0.5)))
        
        # Stress level
        base_stress = self.baseline_metrics['stress_level']
        stress_variation = self._get_circadian_variation('stress_level', hour)
        situational_stress = self._get_situational_stress(hour, weekday)
        stress_noise = np.random.normal(0, 0.5)
        stress_level = max(0, min(10, base_stress + stress_variation + situational_stress + stress_noise))
        
        return {
            'heart_rate': round(heart_rate),
            'blood_oxygen': round(blood_oxygen, 1),
            'temperature': round(temperature, 1),
            'steps': int(steps),
            'sleep_quality': round(sleep_quality, 1),
            'stress_level': round(stress_level, 1)
        }
    
    def _introduce_realistic_anomaly(self, metrics: Dict[str, float], timestamp: datetime) -> Dict[str, float]:
        """Introduce realistic anomalies into the data"""
        anomalous_metrics = metrics.copy()
        
        # Choose random metric to make anomalous
        anomaly_type = random.choice(['heart_rate', 'blood_oxygen', 'temperature', 'stress_level'])
        
        if anomaly_type == 'heart_rate':
            # Simulate tachycardia or bradycardia
            if random.random() < 0.5:
                anomalous_metrics['heart_rate'] = random.randint(110, 140)  # Tachycardia
            else:
                anomalous_metrics['heart_rate'] = random.randint(45, 55)   # Bradycardia
        
        elif anomaly_type == 'blood_oxygen':
            # Simulate hypoxemia
            anomalous_metrics['blood_oxygen'] = round(random.uniform(88, 94), 1)
        
        elif anomaly_type == 'temperature':
            # Simulate fever or hypothermia
            if random.random() < 0.7:
                anomalous_metrics['temperature'] = round(random.uniform(100.5, 103), 1)  # Fever
            else:
                anomalous_metrics['temperature'] = round(random.uniform(95, 96.5), 1)    # Hypothermia
        
        elif anomaly_type == 'stress_level':
            # Simulate high stress episode
            anomalous_metrics['stress_level'] = round(random.uniform(8, 10), 1)
        
        return anomalous_metrics
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics when generation fails"""
        return {
            'heart_rate': 72,
            'blood_oxygen': 98.0,
            'temperature': 98.6,
            'steps': 8000,
            'sleep_quality': 7.5,
            'stress_level': 3.0
        }