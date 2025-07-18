import React from 'react';
import { MetricsGrid } from './MetricsGrid';
import { HealthChart } from './HealthChart';
import { AlertsPanel } from './AlertsPanel';
import { RecommendationsPanel } from './RecommendationsPanel';
import { AnomalyDetectionPanel } from './AnomalyDetectionPanel';

export const Dashboard: React.FC = () => {
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Health Dashboard</h2>
          <p className="text-gray-600 mt-1">Real-time health monitoring and AI-powered insights</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Live Data</span>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <MetricsGrid />

      {/* Charts and Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Health Chart */}
        <div className="lg:col-span-2">
          <HealthChart />
        </div>
        
        {/* Alerts Panel */}
        <div className="space-y-6">
          <AlertsPanel />
          <RecommendationsPanel />
        </div>
      </div>

      {/* Anomaly Detection Panel */}
      <AnomalyDetectionPanel />
    </div>
  );
};