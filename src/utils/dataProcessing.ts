// Data Processing utilities for Health Monitoring System
// This file contains functions for data cleaning, preprocessing, and feature engineering

export interface RawHealthData {
  timestamp: number;
  heartRate?: number;
  bloodOxygen?: number;
  temperature?: number;
  steps?: number;
  sleepQuality?: number;
  stressLevel?: number;
}

export interface ProcessedHealthData {
  timestamp: number;
  heartRate: number;
  bloodOxygen: number;
  temperature: number;
  steps: number;
  sleepQuality: number;
  stressLevel: number;
  heartRateVariability: number;
  movingAvgHeartRate: number;
  dailyStepGoalProgress: number;
}

export class HealthDataProcessor {
  private static readonly NORMAL_RANGES = {
    heartRate: { min: 60, max: 100 },
    bloodOxygen: { min: 95, max: 100 },
    temperature: { min: 97.0, max: 99.5 },
    steps: { min: 0, max: 50000 },
    sleepQuality: { min: 0, max: 10 },
    stressLevel: { min: 0, max: 10 },
  };

  private static readonly DAILY_STEP_GOAL = 8000;

  /**
   * Clean and preprocess raw health data
   */
  static preprocessData(rawData: RawHealthData[]): ProcessedHealthData[] {
    // Remove invalid entries
    const validData = rawData.filter(this.isValidDataPoint);
    
    // Handle missing values
    const imputedData = this.imputeMissingValues(validData);
    
    // Remove outliers
    const cleanedData = this.removeOutliers(imputedData);
    
    // Feature engineering
    const processedData = this.engineerFeatures(cleanedData);
    
    return processedData;
  }

  /**
   * Validate data point
   */
  private static isValidDataPoint(data: RawHealthData): boolean {
    return (
      data.timestamp > 0 &&
      (data.heartRate === undefined || this.isInRange(data.heartRate, 'heartRate')) &&
      (data.bloodOxygen === undefined || this.isInRange(data.bloodOxygen, 'bloodOxygen')) &&
      (data.temperature === undefined || this.isInRange(data.temperature, 'temperature')) &&
      (data.steps === undefined || this.isInRange(data.steps, 'steps')) &&
      (data.sleepQuality === undefined || this.isInRange(data.sleepQuality, 'sleepQuality')) &&
      (data.stressLevel === undefined || this.isInRange(data.stressLevel, 'stressLevel'))
    );
  }

  /**
   * Check if value is within normal range
   */
  private static isInRange(value: number, metric: keyof typeof HealthDataProcessor.NORMAL_RANGES): boolean {
    const range = this.NORMAL_RANGES[metric];
    return value >= range.min && value <= range.max;
  }

  /**
   * Impute missing values using forward fill and interpolation
   */
  private static imputeMissingValues(data: RawHealthData[]): RawHealthData[] {
    const result = [...data];
    
    // Forward fill missing values
    for (let i = 1; i < result.length; i++) {
      const current = result[i];
      const previous = result[i - 1];
      
      if (current.heartRate === undefined && previous.heartRate !== undefined) {
        current.heartRate = previous.heartRate;
      }
      if (current.bloodOxygen === undefined && previous.bloodOxygen !== undefined) {
        current.bloodOxygen = previous.bloodOxygen;
      }
      if (current.temperature === undefined && previous.temperature !== undefined) {
        current.temperature = previous.temperature;
      }
      if (current.steps === undefined && previous.steps !== undefined) {
        current.steps = previous.steps;
      }
      if (current.sleepQuality === undefined && previous.sleepQuality !== undefined) {
        current.sleepQuality = previous.sleepQuality;
      }
      if (current.stressLevel === undefined && previous.stressLevel !== undefined) {
        current.stressLevel = previous.stressLevel;
      }
    }

    // Fill any remaining missing values with median
    const medianValues = this.calculateMedianValues(result);
    
    return result.map(item => ({
      ...item,
      heartRate: item.heartRate ?? medianValues.heartRate,
      bloodOxygen: item.bloodOxygen ?? medianValues.bloodOxygen,
      temperature: item.temperature ?? medianValues.temperature,
      steps: item.steps ?? medianValues.steps,
      sleepQuality: item.sleepQuality ?? medianValues.sleepQuality,
      stressLevel: item.stressLevel ?? medianValues.stressLevel,
    }));
  }

  /**
   * Calculate median values for imputation
   */
  private static calculateMedianValues(data: RawHealthData[]): Required<Omit<RawHealthData, 'timestamp'>> {
    const metrics = ['heartRate', 'bloodOxygen', 'temperature', 'steps', 'sleepQuality', 'stressLevel'] as const;
    const medians: any = {};

    metrics.forEach(metric => {
      const values = data
        .map(item => item[metric])
        .filter(val => val !== undefined) as number[];
      
      if (values.length > 0) {
        values.sort((a, b) => a - b);
        const mid = Math.floor(values.length / 2);
        medians[metric] = values.length % 2 === 0 
          ? (values[mid - 1] + values[mid]) / 2 
          : values[mid];
      } else {
        // Default values if no data available
        const defaults = {
          heartRate: 72,
          bloodOxygen: 98,
          temperature: 98.6,
          steps: 0,
          sleepQuality: 7,
          stressLevel: 3,
        };
        medians[metric] = defaults[metric];
      }
    });

    return medians;
  }

  /**
   * Remove outliers using IQR method
   */
  private static removeOutliers(data: RawHealthData[]): RawHealthData[] {
    const metrics = ['heartRate', 'bloodOxygen', 'temperature', 'steps', 'sleepQuality', 'stressLevel'] as const;
    const outlierFlags: boolean[] = new Array(data.length).fill(false);

    metrics.forEach(metric => {
      const values = data.map(item => item[metric]!);
      const q1 = this.percentile(values, 25);
      const q3 = this.percentile(values, 75);
      const iqr = q3 - q1;
      const lowerBound = q1 - 1.5 * iqr;
      const upperBound = q3 + 1.5 * iqr;

      data.forEach((item, index) => {
        const value = item[metric]!;
        if (value < lowerBound || value > upperBound) {
          outlierFlags[index] = true;
        }
      });
    });

    // Remove data points marked as outliers
    return data.filter((_, index) => !outlierFlags[index]);
  }

  /**
   * Calculate percentile
   */
  private static percentile(values: number[], percentile: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) {
      return sorted[lower];
    }
    
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  /**
   * Engineer features from cleaned data
   */
  private static engineerFeatures(data: RawHealthData[]): ProcessedHealthData[] {
    return data.map((item, index) => {
      const heartRateVariability = this.calculateHRV(data, index);
      const movingAvgHeartRate = this.calculateMovingAverage(data, index, 'heartRate', 5);
      const dailyStepGoalProgress = (item.steps! / this.DAILY_STEP_GOAL) * 100;
      )

      return {
        timestamp: item.timestamp,
        heartRate: item.heartRate!,
        bloodOxygen: item.bloodOxygen!,
        temperature: item.temperature!,
        steps: item.steps!,
        sleepQuality: item.sleepQuality!,
        stressLevel: item.stressLevel!,
        heartRateVariability,
        movingAvgHeartRate,
        dailyStepGoalProgress: Math.min(dailyStepGoalProgress, 100),
      };
    });
  }

  /**
   * Calculate Heart Rate Variability (simplified)
   */
  private static calculateHRV(data: RawHealthData[], currentIndex: number): number {
    const windowSize = 5;
    const startIndex = Math.max(0, currentIndex - windowSize);
    const endIndex = Math.min(data.length, currentIndex + windowSize + 1);
    
    const heartRates = data.slice(startIndex, endIndex)
      .map(item => item.heartRate!)
      .filter(hr => hr !== undefined);

    if (heartRates.length < 2) return 0;

    const differences = heartRates.slice(1).map((hr, i) => Math.abs(hr - heartRates[i]));
    const meanDifference = differences.reduce((sum, diff) => sum + diff, 0) / differences.length;
    
    return meanDifference;
  }

  /**
   * Calculate moving average
   */
  private static calculateMovingAverage(
    data: RawHealthData[], 
    currentIndex: number, 
    metric: keyof RawHealthData, 
    windowSize: number
  ): number {
    const startIndex = Math.max(0, currentIndex - windowSize + 1);
    const endIndex = currentIndex + 1;
    
    const values = data.slice(startIndex, endIndex)
      .map(item => item[metric] as number)
      .filter(val => val !== undefined);

    if (values.length === 0) return 0;

    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  /**
   * Normalize data for machine learning
   */
  static normalizeData(data: ProcessedHealthData[]): ProcessedHealthData[] {
    const normalizedData = [...data];
    const metrics = ['heartRate', 'bloodOxygen', 'temperature', 'steps', 'sleepQuality', 'stressLevel'] as const;

    metrics.forEach(metric => {
      const values = normalizedData.map(item => item[metric]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min;

      if (range > 0) {
        normalizedData.forEach(item => {
          (item[metric] as number) = (item[metric] - min) / range;
        });
      }
    });

    return normalizedData;
  }

  /**
   * Create time windows for sequence modeling
   */
  static createTimeWindows(data: ProcessedHealthData[], windowSize: number): ProcessedHealthData[][] {
    const windows: ProcessedHealthData[][] = [];
    
    for (let i = 0; i <= data.length - windowSize; i++) {
      windows.push(data.slice(i, i + windowSize));
    }
    
    return windows;
  }

  /**
   * Split data into training and testing sets
   */
  static splitData(data: ProcessedHealthData[], testRatio: number = 0.2): {
    training: ProcessedHealthData[];
    testing: ProcessedHealthData[];
  } {
    const splitIndex = Math.floor(data.length * (1 - testRatio));
    
    return {
      training: data.slice(0, splitIndex),
      testing: data.slice(splitIndex),
    };
  }
}