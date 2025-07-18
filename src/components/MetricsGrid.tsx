import React from 'react';
import { Heart, Activity, Thermometer, Droplets, Moon, TrendingUp, TrendingDown } from 'lucide-react';
import { useHealthData } from '../contexts/HealthDataContext';

export const MetricsGrid: React.FC = () => {
  const { currentMetrics } = useHealthData();

  const metrics = [
    {
      id: 'heartRate',
      label: 'Heart Rate',
      value: currentMetrics.heartRate,
      unit: 'bpm',
      icon: Heart,
      color: 'bg-red-500',
      trend: currentMetrics.heartRateTrend,
      normal: currentMetrics.heartRate >= 60 && currentMetrics.heartRate <= 100,
    },
    {
      id: 'steps',
      label: 'Steps Today',
      value: currentMetrics.steps,
      unit: 'steps',
      icon: Activity,
      color: 'bg-blue-500',
      trend: 'up',
      normal: currentMetrics.steps >= 8000,
    },
    {
      id: 'temperature',
      label: 'Body Temperature',
      value: currentMetrics.temperature,
      unit: '°F',
      icon: Thermometer,
      color: 'bg-orange-500',
      trend: currentMetrics.temperatureTrend,
      normal: currentMetrics.temperature >= 97.0 && currentMetrics.temperature <= 99.5,
    },
    {
      id: 'bloodOxygen',
      label: 'Blood Oxygen',
      value: currentMetrics.bloodOxygen,
      unit: '%',
      icon: Droplets,
      color: 'bg-teal-500',
      trend: currentMetrics.bloodOxygenTrend,
      normal: currentMetrics.bloodOxygen >= 95,
    },
    {
      id: 'sleep',
      label: 'Sleep Quality',
      value: currentMetrics.sleepQuality,
      unit: '/10',
      icon: Moon,
      color: 'bg-purple-500',
      trend: currentMetrics.sleepTrend,
      normal: currentMetrics.sleepQuality >= 7,
    },
    {
      id: 'stress',
      label: 'Stress Level',
      value: currentMetrics.stressLevel,
      unit: '/10',
      icon: Activity,
      color: 'bg-yellow-500',
      trend: currentMetrics.stressTrend,
      normal: currentMetrics.stressLevel <= 5,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {metrics.map((metric) => {
        const Icon = metric.icon;
        const TrendIcon = metric.trend === 'up' ? TrendingUp : TrendingDown;
        
        return (
          <div
            key={metric.id}
            className={`
              bg-white rounded-lg shadow-sm border p-6 transition-all duration-200 hover:shadow-md
              ${!metric.normal ? 'border-red-200 bg-red-50' : 'border-gray-200'}
            `}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${metric.color}`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-600">{metric.label}</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {metric.value}
                    <span className="text-sm font-normal text-gray-500 ml-1">
                      {metric.unit}
                    </span>
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-1">
                <TrendIcon 
                  className={`w-4 h-4 ${
                    metric.trend === 'up' ? 'text-green-500' : 'text-red-500'
                  }`} 
                />
                {!metric.normal && (
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                )}
              </div>
            </div>
            
            {!metric.normal && (
              <div className="mt-2 text-xs text-red-600 font-medium">
                Outside normal range
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};