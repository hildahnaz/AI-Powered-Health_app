import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface HealthMetrics {
  heartRate: number;
  bloodOxygen: number;
  temperature: number;
  steps: number;
  sleepQuality: number;
  stressLevel: number;
  heartRateTrend: 'up' | 'down';
  bloodOxygenTrend: 'up' | 'down';
  temperatureTrend: 'up' | 'down';
  sleepTrend: 'up' | 'down';
  stressTrend: 'up' | 'down';
}

interface HistoricalDataPoint {
  timestamp: number;
  heartRate: number;
  bloodOxygen: number;
  temperature: number;
}

interface Recommendation {
  id: string;
  type: 'exercise' | 'nutrition' | 'sleep' | 'medical';
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
}

interface Anomaly {
  id: string;
  metric: string;
  description: string;
  confidence: number;
  timestamp: number;
}

interface HealthDataContextType {
  currentMetrics: HealthMetrics;
  historicalData: HistoricalDataPoint[];
  recommendations: Recommendation[];
  anomalies: Anomaly[];
}

const HealthDataContext = createContext<HealthDataContextType | undefined>(undefined);

export const useHealthData = () => {
  const context = useContext(HealthDataContext);
  if (!context) {
    throw new Error('useHealthData must be used within a HealthDataProvider');
  }
  return context;
};

export const HealthDataProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentMetrics, setCurrentMetrics] = useState<HealthMetrics>({
    heartRate: 72,
    bloodOxygen: 98,
    temperature: 98.6,
    steps: 8432,
    sleepQuality: 7.5,
    stressLevel: 3,
    heartRateTrend: 'up',
    bloodOxygenTrend: 'up',
    temperatureTrend: 'down',
    sleepTrend: 'up',
    stressTrend: 'down',
  });

  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);

  const recommendations: Recommendation[] = [
    {
      id: '1',
      type: 'exercise',
      title: 'Take a 10-minute walk',
      description: 'Your step count is below your daily goal. A short walk can boost your energy and mood.',
      priority: 'medium',
    },
    {
      id: '2',
      type: 'nutrition',
      title: 'Stay hydrated',
      description: 'Your heart rate is slightly elevated. Drinking water can help maintain optimal circulation.',
      priority: 'low',
    },
    {
      id: '3',
      type: 'sleep',
      title: 'Consider earlier bedtime',
      description: 'Your stress levels indicate you might benefit from better sleep quality tonight.',
      priority: 'medium',
    },
  ];

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentMetrics(prev => ({
        ...prev,
        heartRate: Math.max(60, Math.min(100, prev.heartRate + (Math.random() - 0.5) * 4)),
        bloodOxygen: Math.max(94, Math.min(100, prev.bloodOxygen + (Math.random() - 0.5) * 2)),
        temperature: Math.max(97, Math.min(100, prev.temperature + (Math.random() - 0.5) * 0.5)),
        steps: prev.steps + Math.floor(Math.random() * 10),
        stressLevel: Math.max(0, Math.min(10, prev.stressLevel + (Math.random() - 0.5) * 0.5)),
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Generate historical data
  useEffect(() => {
    const generateHistoricalData = () => {
      const data: HistoricalDataPoint[] = [];
      const now = Date.now();
      
      for (let i = 24; i >= 0; i--) {
        const timestamp = now - (i * 60 * 60 * 1000); // Every hour for last 24 hours
        data.push({
          timestamp,
          heartRate: 70 + Math.sin(i * 0.5) * 15 + (Math.random() - 0.5) * 10,
          bloodOxygen: 97 + Math.sin(i * 0.3) * 2 + (Math.random() - 0.5) * 2,
          temperature: 98.6 + Math.sin(i * 0.4) * 0.8 + (Math.random() - 0.5) * 0.5,
        });
      }
      
      setHistoricalData(data);
    };

    generateHistoricalData();
  }, []);

  // Simulate anomaly detection
  useEffect(() => {
    const detectAnomalies = () => {
      const newAnomalies: Anomaly[] = [];
      
      if (currentMetrics.heartRate > 95) {
        newAnomalies.push({
          id: 'hr-anomaly',
          metric: 'Heart Rate',
          description: 'Heart rate is unusually high for your baseline',
          confidence: 0.87,
          timestamp: Date.now(),
        });
      }
      
      if (currentMetrics.bloodOxygen < 96) {
        newAnomalies.push({
          id: 'o2-anomaly',
          metric: 'Blood Oxygen',
          description: 'Blood oxygen levels are below normal range',
          confidence: 0.92,
          timestamp: Date.now(),
        });
      }
      
      setAnomalies(newAnomalies);
    };

    detectAnomalies();
  }, [currentMetrics]);

  const value = {
    currentMetrics,
    historicalData,
    recommendations,
    anomalies,
  };

  return (
    <HealthDataContext.Provider value={value}>
      {children}
    </HealthDataContext.Provider>
  );
};