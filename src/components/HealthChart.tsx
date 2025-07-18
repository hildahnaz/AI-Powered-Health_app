import React, { useState } from 'react';
import { useHealthData } from '../contexts/HealthDataContext';

export const HealthChart: React.FC = () => {
  const { historicalData } = useHealthData();
  const [selectedMetric, setSelectedMetric] = useState<'heartRate' | 'bloodOxygen' | 'temperature'>('heartRate');

  const metrics = [
    { key: 'heartRate', label: 'Heart Rate', color: 'rgb(239, 68, 68)', unit: 'bpm' },
    { key: 'bloodOxygen', label: 'Blood Oxygen', color: 'rgb(20, 184, 166)', unit: '%' },
    { key: 'temperature', label: 'Temperature', color: 'rgb(249, 115, 22)', unit: '°F' },
  ];

  const selectedMetricData = metrics.find(m => m.key === selectedMetric);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Health Trends</h3>
        <div className="flex space-x-2">
          {metrics.map((metric) => (
            <button
              key={metric.key}
              onClick={() => setSelectedMetric(metric.key as any)}
              className={`
                px-3 py-1 rounded-full text-sm font-medium transition-colors
                ${selectedMetric === metric.key 
                  ? 'bg-teal-100 text-teal-700' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }
              `}
            >
              {metric.label}
            </button>
          ))}
        </div>
      </div>
      
      <div className="h-64 relative">
        <svg width="100%" height="100%" viewBox="0 0 800 200" className="overflow-visible">
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="40" height="20" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 20" fill="none" stroke="#f3f4f6" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
          
          {/* Chart line */}
          {historicalData.length > 1 && (
            <polyline
              fill="none"
              stroke={selectedMetricData?.color}
              strokeWidth="2"
              points={historicalData.map((point, index) => {
                const x = (index / (historicalData.length - 1)) * 800;
                const value = point[selectedMetric];
                const normalizedValue = selectedMetric === 'heartRate' 
                  ? ((value - 50) / 100) * 200
                  : selectedMetric === 'bloodOxygen'
                  ? ((value - 90) / 10) * 200
                  : ((value - 96) / 4) * 200;
                const y = 200 - normalizedValue;
                return `${x},${y}`;
              }).join(' ')}
            />
          )}
          
          {/* Data points */}
          {historicalData.map((point, index) => {
            const x = (index / (historicalData.length - 1)) * 800;
            const value = point[selectedMetric];
            const normalizedValue = selectedMetric === 'heartRate' 
              ? ((value - 50) / 100) * 200
              : selectedMetric === 'bloodOxygen'
              ? ((value - 90) / 10) * 200
              : ((value - 96) / 4) * 200;
            const y = 200 - normalizedValue;
            
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="4"
                fill={selectedMetricData?.color}
                className="hover:r-6 transition-all duration-200"
              />
            );
          })}
        </svg>
        
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-gray-500 -ml-8">
          {selectedMetric === 'heartRate' && (
            <>
              <span>150</span>
              <span>100</span>
              <span>50</span>
            </>
          )}
          {selectedMetric === 'bloodOxygen' && (
            <>
              <span>100%</span>
              <span>95%</span>
              <span>90%</span>
            </>
          )}
          {selectedMetric === 'temperature' && (
            <>
              <span>100°F</span>
              <span>98°F</span>
              <span>96°F</span>
            </>
          )}
        </div>
      </div>
      
      <div className="mt-4 text-sm text-gray-600">
        Last 24 hours • {selectedMetricData?.label} ({selectedMetricData?.unit})
      </div>
    </div>
  );
};