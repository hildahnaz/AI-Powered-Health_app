import React, { useState, useEffect } from 'react';
import { Dashboard } from './components/Dashboard';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { HealthDataProvider } from './contexts/HealthDataContext';
import { AlertProvider } from './contexts/AlertContext';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <HealthDataProvider>
      <AlertProvider>
        <div className="min-h-screen bg-gray-50">
          <div className="flex h-screen">
            {/* Sidebar */}
            <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
            
            {/* Main Content */}
            <div className="flex-1 flex flex-col overflow-hidden">
              <Header onMenuClick={() => setSidebarOpen(true)} />
              <main className="flex-1 overflow-y-auto">
                <Dashboard />
              </main>
            </div>
          </div>
        </div>
      </AlertProvider>
    </HealthDataProvider>
  );
}

export default App;