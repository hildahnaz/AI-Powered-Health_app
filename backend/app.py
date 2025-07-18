"""
AI-Powered Health Monitoring System - Main Flask Application
This file serves as the main entry point for the backend API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Import our custom modules
from models.anomaly_detector import HealthAnomalyDetector
from models.recommendation_engine import HealthRecommendationEngine
from utils.data_processor import HealthDataProcessor
from utils.health_simulator import HealthDataSimulator
from database.db_manager import DatabaseManager

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication
api = Api(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
db_manager = DatabaseManager()
anomaly_detector = HealthAnomalyDetector()
recommendation_engine = HealthRecommendationEngine()
data_processor = HealthDataProcessor()
health_simulator = HealthDataSimulator()

class HealthMetricsAPI(Resource):
    """API endpoint for current health metrics"""
    
    def get(self):
        try:
            # Get current simulated health data
            current_data = health_simulator.generate_current_metrics()
            
            # Process the data
            processed_data = data_processor.preprocess_single_datapoint(current_data)
            
            # Detect anomalies
            anomalies = anomaly_detector.detect_anomalies(processed_data)
            
            # Generate recommendations
            recommendations = recommendation_engine.generate_recommendations(
                processed_data, anomalies
            )
            
            response = {
                'timestamp': datetime.now().isoformat(),
                'metrics': processed_data,
                'anomalies': anomalies,
                'recommendations': recommendations,
                'status': 'success'
            }
            
            # Store in database
            db_manager.store_health_data(processed_data, anomalies)
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in HealthMetricsAPI: {str(e)}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

class HistoricalDataAPI(Resource):
    """API endpoint for historical health data"""
    
    def get(self):
        try:
            # Get query parameters
            hours = request.args.get('hours', 24, type=int)
            metric = request.args.get('metric', 'all')
            
            # Retrieve historical data
            historical_data = db_manager.get_historical_data(hours, metric)
            
            # Process for visualization
            processed_historical = data_processor.prepare_for_visualization(
                historical_data
            )
            
            response = {
                'data': processed_historical,
                'period_hours': hours,
                'metric_filter': metric,
                'status': 'success'
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in HistoricalDataAPI: {str(e)}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

class AnomalyAnalysisAPI(Resource):
    """API endpoint for detailed anomaly analysis"""
    
    def get(self):
        try:
            # Get recent anomalies
            recent_anomalies = db_manager.get_recent_anomalies(hours=24)
            
            # Perform detailed analysis
            analysis = anomaly_detector.analyze_anomaly_patterns(recent_anomalies)
            
            response = {
                'anomalies': recent_anomalies,
                'analysis': analysis,
                'model_performance': anomaly_detector.get_model_performance(),
                'status': 'success'
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in AnomalyAnalysisAPI: {str(e)}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

class ModelTrainingAPI(Resource):
    """API endpoint for retraining AI models"""
    
    def post(self):
        try:
            # Get training parameters
            data = request.get_json()
            retrain_anomaly = data.get('retrain_anomaly', True)
            retrain_recommendations = data.get('retrain_recommendations', True)
            
            results = {}
            
            if retrain_anomaly:
                # Get training data
                training_data = db_manager.get_training_data()
                
                # Retrain anomaly detection model
                anomaly_results = anomaly_detector.retrain_model(training_data)
                results['anomaly_model'] = anomaly_results
            
            if retrain_recommendations:
                # Retrain recommendation engine
                recommendation_results = recommendation_engine.retrain_model()
                results['recommendation_model'] = recommendation_results
            
            response = {
                'training_results': results,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in ModelTrainingAPI: {str(e)}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

class HealthReportAPI(Resource):
    """API endpoint for generating health reports"""
    
    def get(self):
        try:
            # Get query parameters
            days = request.args.get('days', 7, type=int)
            report_type = request.args.get('type', 'summary')
            
            # Generate comprehensive health report
            report_data = db_manager.get_report_data(days)
            
            # Process report based on type
            if report_type == 'detailed':
                report = data_processor.generate_detailed_report(report_data)
            else:
                report = data_processor.generate_summary_report(report_data)
            
            response = {
                'report': report,
                'period_days': days,
                'report_type': report_type,
                'generated_at': datetime.now().isoformat(),
                'status': 'success'
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error in HealthReportAPI: {str(e)}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

# Register API endpoints
api.add_resource(HealthMetricsAPI, '/api/health/current')
api.add_resource(HistoricalDataAPI, '/api/health/historical')
api.add_resource(AnomalyAnalysisAPI, '/api/health/anomalies')
api.add_resource(ModelTrainingAPI, '/api/models/train')
api.add_resource(HealthReportAPI, '/api/health/report')

@app.route('/api/health/status', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    # Initialize database
    db_manager.initialize_database()
    
    # Train initial models
    logger.info("Training initial AI models...")
    
    # Generate initial training data
    training_data = health_simulator.generate_training_dataset(days=30)
    processed_training_data = data_processor.preprocess_dataset(training_data)
    
    # Train anomaly detection model
    anomaly_detector.train_initial_model(processed_training_data)
    
    # Train recommendation engine
    recommendation_engine.train_initial_model(processed_training_data)
    
    logger.info("AI models trained successfully!")
    
    # Start Flask development server
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )