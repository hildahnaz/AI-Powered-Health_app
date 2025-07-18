"""
Database Manager for Health Monitoring System
Handles all database operations including data storage, retrieval, and management
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages SQLite database operations for health monitoring data
    """
    
    def __init__(self, db_path: str = "health_monitoring.db"):
        self.db_path = db_path
        self.connection = None
        
    def initialize_database(self):
        """Initialize database with required tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            # Create tables
            self._create_tables()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _create_tables(self):
        """Create all required database tables"""
        cursor = self.connection.cursor()
        
        # Health data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                heart_rate REAL,
                blood_oxygen REAL,
                temperature REAL,
                steps INTEGER,
                sleep_quality REAL,
                stress_level REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Anomalies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT NOT NULL,
                actions TEXT,  -- JSON array of actions
                source TEXT,
                confidence REAL,
                implemented BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User profiles table (for future multi-user support)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                name TEXT,
                age INTEGER,
                gender TEXT,
                baseline_metrics TEXT,  -- JSON object
                health_goals TEXT,      -- JSON object
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                training_date DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_data_timestamp ON health_data(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_timestamp ON recommendations(timestamp)')
        
        self.connection.commit()
        logger.info("Database tables created successfully")
    
    def store_health_data(self, health_data: Dict[str, float], anomalies: List[Dict] = None):
        """
        Store health data and associated anomalies
        
        Args:
            health_data: Dictionary with health metrics
            anomalies: List of detected anomalies
        """
        try:
            cursor = self.connection.cursor()
            
            # Store health data
            cursor.execute('''
                INSERT INTO health_data 
                (timestamp, heart_rate, blood_oxygen, temperature, steps, sleep_quality, stress_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                health_data.get('heart_rate'),
                health_data.get('blood_oxygen'),
                health_data.get('temperature'),
                health_data.get('steps'),
                health_data.get('sleep_quality'),
                health_data.get('stress_level')
            ))
            
            # Store anomalies if any
            if anomalies:
                for anomaly in anomalies:
                    cursor.execute('''
                        INSERT INTO anomalies 
                        (timestamp, metric, value, severity, confidence, description)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.now(),
                        anomaly.get('metric'),
                        anomaly.get('value', 0),
                        anomaly.get('severity', 'medium'),
                        anomaly.get('confidence', 0.5),
                        anomaly.get('description', '')
                    ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing health data: {str(e)}")
            self.connection.rollback()
            raise
    
    def store_recommendations(self, recommendations: List[Dict]):
        """
        Store health recommendations
        
        Args:
            recommendations: List of recommendation dictionaries
        """
        try:
            cursor = self.connection.cursor()
            
            for rec in recommendations:
                cursor.execute('''
                    INSERT INTO recommendations 
                    (timestamp, type, title, description, priority, actions, source, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    rec.get('type'),
                    rec.get('title'),
                    rec.get('description'),
                    rec.get('priority'),
                    json.dumps(rec.get('actions', [])),
                    rec.get('source'),
                    rec.get('confidence', 0.5)
                ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing recommendations: {str(e)}")
            self.connection.rollback()
            raise
    
    def get_historical_data(self, hours: int = 24, metric: str = 'all') -> List[Dict]:
        """
        Retrieve historical health data
        
        Args:
            hours: Number of hours to look back
            metric: Specific metric to retrieve or 'all'
            
        Returns:
            List of health data records
        """
        try:
            cursor = self.connection.cursor()
            
            # Calculate start time
            start_time = datetime.now() - timedelta(hours=hours)
            
            if metric == 'all':
                cursor.execute('''
                    SELECT * FROM health_data 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp ASC
                ''', (start_time,))
            else:
                cursor.execute(f'''
                    SELECT timestamp, {metric} FROM health_data 
                    WHERE timestamp >= ? AND {metric} IS NOT NULL
                    ORDER BY timestamp ASC
                ''', (start_time,))
            
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                record = dict(row)
                # Convert timestamp to ISO format
                if 'timestamp' in record:
                    record['timestamp'] = record['timestamp']
                data.append(record)
            
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {str(e)}")
            return []
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Dict]:
        """
        Retrieve recent anomalies
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of anomaly records
        """
        try:
            cursor = self.connection.cursor()
            
            start_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute('''
                SELECT * FROM anomalies 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            ''', (start_time,))
            
            rows = cursor.fetchall()
            
            anomalies = []
            for row in rows:
                anomaly = dict(row)
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error retrieving recent anomalies: {str(e)}")
            return []
    
    def get_training_data(self, days: int = 30) -> pd.DataFrame:
        """
        Retrieve training data for model retraining
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with training data
        """
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT * FROM health_data 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, self.connection, params=(start_time,))
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            return pd.DataFrame()
    
    def get_report_data(self, days: int = 7) -> List[Dict]:
        """
        Retrieve data for health reports
        
        Args:
            days: Number of days to include in report
            
        Returns:
            List of health data records
        """
        try:
            cursor = self.connection.cursor()
            
            start_time = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT * FROM health_data 
                WHERE timestamp >= ? 
                ORDER BY timestamp ASC
            ''', (start_time,))
            
            rows = cursor.fetchall()
            
            data = []
            for row in rows:
                record = dict(row)
                data.append(record)
            
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving report data: {str(e)}")
            return []
    
    def store_model_performance(self, model_name: str, version: str, metrics: Dict[str, float]):
        """
        Store model performance metrics
        
        Args:
            model_name: Name of the model
            version: Model version
            metrics: Performance metrics dictionary
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance 
                (model_name, version, accuracy, precision_score, recall, f1_score, training_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name,
                version,
                metrics.get('accuracy'),
                metrics.get('precision'),
                metrics.get('recall'),
                metrics.get('f1_score'),
                datetime.now()
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing model performance: {str(e)}")
            self.connection.rollback()
            raise
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """
        Calculate data quality metrics
        
        Returns:
            Dictionary with data quality information
        """
        try:
            cursor = self.connection.cursor()
            
            # Total records
            cursor.execute('SELECT COUNT(*) as total FROM health_data')
            total_records = cursor.fetchone()['total']
            
            # Records in last 24 hours
            yesterday = datetime.now() - timedelta(hours=24)
            cursor.execute('SELECT COUNT(*) as recent FROM health_data WHERE timestamp >= ?', (yesterday,))
            recent_records = cursor.fetchone()['recent']
            
            # Completeness check
            completeness = {}
            metrics = ['heart_rate', 'blood_oxygen', 'temperature', 'steps', 'sleep_quality', 'stress_level']
            
            for metric in metrics:
                cursor.execute(f'SELECT COUNT(*) as non_null FROM health_data WHERE {metric} IS NOT NULL')
                non_null_count = cursor.fetchone()['non_null']
                completeness[metric] = (non_null_count / total_records * 100) if total_records > 0 else 0
            
            # Anomaly frequency
            cursor.execute('SELECT COUNT(*) as anomaly_count FROM anomalies WHERE timestamp >= ?', (yesterday,))
            anomaly_count = cursor.fetchone()['anomaly_count']
            
            return {
                'total_records': total_records,
                'recent_records': recent_records,
                'completeness': completeness,
                'anomaly_frequency': (anomaly_count / recent_records * 100) if recent_records > 0 else 0,
                'data_freshness': 'good' if recent_records > 0 else 'stale'
            }
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Clean up old data to manage database size
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        try:
            cursor = self.connection.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old health data
            cursor.execute('DELETE FROM health_data WHERE timestamp < ?', (cutoff_date,))
            health_deleted = cursor.rowcount
            
            # Clean up old anomalies
            cursor.execute('DELETE FROM anomalies WHERE timestamp < ?', (cutoff_date,))
            anomalies_deleted = cursor.rowcount
            
            # Clean up old recommendations
            cursor.execute('DELETE FROM recommendations WHERE timestamp < ?', (cutoff_date,))
            recommendations_deleted = cursor.rowcount
            
            self.connection.commit()
            
            logger.info(f"Cleaned up old data: {health_deleted} health records, "
                       f"{anomalies_deleted} anomalies, {recommendations_deleted} recommendations")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            self.connection.rollback()
            raise
    
    def backup_database(self, backup_path: str):
        """
        Create a backup of the database
        
        Args:
            backup_path: Path for the backup file
        """
        try:
            # Create backup directory if it doesn't exist
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Create backup
            backup_conn = sqlite3.connect(backup_path)
            self.connection.backup(backup_conn)
            backup_conn.close()
            
            logger.info(f"Database backup created at {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating database backup: {str(e)}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close_connection()