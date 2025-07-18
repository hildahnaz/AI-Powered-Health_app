import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: number;
  read: boolean;
}

interface AlertContextType {
  alerts: Alert[];
  addAlert: (alert: Omit<Alert, 'id' | 'timestamp' | 'read'>) => void;
  dismissAlert: (id: string) => void;
  markAsRead: (id: string) => void;
}

const AlertContext = createContext<AlertContextType | undefined>(undefined);

export const useAlert = () => {
  const context = useContext(AlertContext);
  if (!context) {
    throw new Error('useAlert must be used within an AlertProvider');
  }
  return context;
};

export const AlertProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: '1',
      type: 'warning',
      title: 'Elevated Heart Rate',
      message: 'Your heart rate has been above 90 bpm for the last 10 minutes during rest.',
      timestamp: Date.now() - 300000, // 5 minutes ago
      read: false,
    },
    {
      id: '2',
      type: 'info',
      title: 'Daily Goal Achieved',
      message: 'Congratulations! You\'ve reached your daily step goal of 8,000 steps.',
      timestamp: Date.now() - 1800000, // 30 minutes ago
      read: false,
    },
  ]);

  const addAlert = (alert: Omit<Alert, 'id' | 'timestamp' | 'read'>) => {
    const newAlert: Alert = {
      ...alert,
      id: Date.now().toString(),
      timestamp: Date.now(),
      read: false,
    };
    setAlerts(prev => [newAlert, ...prev]);
  };

  const dismissAlert = (id: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== id));
  };

  const markAsRead = (id: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === id ? { ...alert, read: true } : alert
    ));
  };

  const value = {
    alerts,
    addAlert,
    dismissAlert,
    markAsRead,
  };

  return (
    <AlertContext.Provider value={value}>
      {children}
    </AlertContext.Provider>
  );
};