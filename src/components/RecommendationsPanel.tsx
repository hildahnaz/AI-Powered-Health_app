import React from 'react';
import { Lightbulb, Heart, Activity, Moon, Apple } from 'lucide-react';
import { useHealthData } from '../contexts/HealthDataContext';

export const RecommendationsPanel: React.FC = () => {
  const { recommendations } = useHealthData();

  const getRecommendationIcon = (type: string) => {
    switch (type) {
      case 'exercise':
        return <Activity className="w-5 h-5 text-blue-500" />;
      case 'nutrition':
        return <Apple className="w-5 h-5 text-green-500" />;
      case 'sleep':
        return <Moon className="w-5 h-5 text-purple-500" />;
      case 'medical':
        return <Heart className="w-5 h-5 text-red-500" />;
      default:
        return <Lightbulb className="w-5 h-5 text-yellow-500" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Recommendations</h3>
      
      <div className="space-y-3">
        {recommendations.map((rec) => (
          <div
            key={rec.id}
            className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            {getRecommendationIcon(rec.type)}
            <div className="flex-1">
              <h4 className="font-medium text-gray-900">{rec.title}</h4>
              <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
              <div className="flex items-center mt-2">
                <span className={`
                  inline-flex items-center px-2 py-1 rounded-full text-xs font-medium
                  ${rec.priority === 'high' ? 'bg-red-100 text-red-800' :
                    rec.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }
                `}>
                  {rec.priority} priority
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};