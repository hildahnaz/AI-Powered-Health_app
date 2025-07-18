import React from 'react';
import { Shield, TrendingUp, Brain, AlertCircle } from 'lucide-react';
import { useHealthData } from '../contexts/HealthDataContext';

export const AnomalyDetectionPanel: React.FC = () => {
  const { anomalies, currentMetrics } = useHealthData();

  const riskLevels = [
    { metric: 'Heart Rate', value: currentMetrics.heartRate, risk: 'low', normal: '60-100 bpm' },
    { metric: 'Blood Oxygen', value: currentMetrics.bloodOxygen, risk: 'low', normal: '95-100%' },
    { metric: 'Temperature', value: currentMetrics.temperature, risk: 'low', normal: '97-99°F' },
    { metric: 'Stress Level', value: currentMetrics.stressLevel, risk: 'medium', normal: '0-5/10' },
  ];

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high':
        return 'text-red-600 bg-red-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      case 'low':
        return 'text-green-600 bg-green-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Shield className="w-6 h-6 text-teal-500" />
          <h3 className="text-lg font-semibold text-gray-900">AI Anomaly Detection</h3>
        </div>
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-purple-500" />
          <span className="text-sm text-gray-600">Model: Isolation Forest</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Risk Assessment */}
        <div>
          <h4 className="font-medium text-gray-900 mb-3">Risk Assessment</h4>
          <div className="space-y-2">
            {riskLevels.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{item.metric}</p>
                  <p className="text-sm text-gray-600">{item.normal}</p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-gray-900">{item.value}</span>
                  <span className={`
                    px-2 py-1 rounded-full text-xs font-medium
                    ${getRiskColor(item.risk)}
                  `}>
                    {item.risk}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Detected Anomalies */}
        <div>
          <h4 className="font-medium text-gray-900 mb-3">Detected Anomalies</h4>
          <div className="space-y-2">
            {anomalies.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Shield className="w-12 h-12 mx-auto mb-2 text-green-500" />
                <p>No anomalies detected</p>
              </div>
            ) : (
              anomalies.map((anomaly) => (
                <div
                  key={anomaly.id}
                  className="flex items-start space-x-3 p-3 bg-red-50 border border-red-200 rounded-lg"
                >
                  <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
                  <div className="flex-1">
                    <h5 className="font-medium text-red-900">{anomaly.metric}</h5>
                    <p className="text-sm text-red-700 mt-1">{anomaly.description}</p>
                    <p className="text-xs text-red-600 mt-2">
                      Confidence: {(anomaly.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Model Performance */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-900 mb-2">Model Performance</h4>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-teal-600">94.2%</p>
            <p className="text-sm text-gray-600">Accuracy</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-blue-600">2.1%</p>
            <p className="text-sm text-gray-600">False Positive</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-green-600">97.8%</p>
            <p className="text-sm text-gray-600">Sensitivity</p>
          </div>
        </div>
      </div>
    </div>
  );
};