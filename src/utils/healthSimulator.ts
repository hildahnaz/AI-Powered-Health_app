// Health Data Simulator for generating realistic health monitoring data
// This simulates data that would come from wearable devices

export interface SimulatedHealthData {
  timestamp: number;
  heartRate: number;
  bloodOxygen: number;
  temperature: number;
  steps: number;
  sleepQuality: number;
  stressLevel: number;
  activity: 'resting' | 'walking' | 'exercising' | 'sleeping';
  location: 'home' | 'work' | 'gym' | 'outdoor';
}

export class HealthDataSimulator {
  private baselineMetrics = {
    heartRate: 72,
    bloodOxygen: 98,
    temperature: 98.6,
    sleepQuality: 7.5,
    stressLevel: 3,
  };

  private currentSteps = 0;
  private dailyStepGoal = 8000;
  private lastActivity: 'resting' | 'walking' | 'exercising' | 'sleeping' = 'resting';

  /**
   * Generate realistic health data for a given time period
   */
  generateHistoricalData(startTime: number, endTime: number, intervalMinutes: number = 5): SimulatedHealthData[] {
    const data: SimulatedHealthData[] = [];
    const intervalMs = intervalMinutes * 60 * 1000;
    
    let currentTime = startTime;
    this.currentSteps = 0;

    while (currentTime <= endTime) {
      const simulatedData = this.generateDataPoint(currentTime);
      data.push(simulatedData);
      currentTime += intervalMs;
    }

    return data;
  }

  /**
   * Generate a single realistic data point
   */
  generateDataPoint(timestamp: number): SimulatedHealthData {
    const hour = new Date(timestamp).getHours();
    const activity = this.determineActivity(hour);
    const location = this.determineLocation(hour, activity);

    return {
      timestamp,
      heartRate: this.simulateHeartRate(activity, hour),
      bloodOxygen: this.simulateBloodOxygen(activity),
      temperature: this.simulateTemperature(activity, hour),
      steps: this.simulateSteps(activity),
      sleepQuality: this.simulateSleepQuality(hour),
      stressLevel: this.simulateStressLevel(hour, activity),
      activity,
      location,
    };
  }

  /**
   * Determine activity based on time of day
   */
  private determineActivity(hour: number): 'resting' | 'walking' | 'exercising' | 'sleeping' {
    // Sleep period: 11 PM to 7 AM
    if (hour >= 23 || hour <= 7) {
      return 'sleeping';
    }
    
    // Exercise period: 7-8 AM or 6-7 PM
    if ((hour >= 7 && hour <= 8) || (hour >= 18 && hour <= 19)) {
      return Math.random() < 0.3 ? 'exercising' : 'walking';
    }
    
    // Work hours: more likely to be resting
    if (hour >= 9 && hour <= 17) {
      const rand = Math.random();
      if (rand < 0.7) return 'resting';
      if (rand < 0.9) return 'walking';
      return 'exercising';
    }
    
    // Evening: mixed activities
    const rand = Math.random();
    if (rand < 0.5) return 'resting';
    if (rand < 0.8) return 'walking';
    return 'exercising';
  }

  /**
   * Determine location based on time and activity
   */
  private determineLocation(hour: number, activity: 'resting' | 'walking' | 'exercising' | 'sleeping'): 'home' | 'work' | 'gym' | 'outdoor' {
    if (activity === 'sleeping') return 'home';
    if (activity === 'exercising') return Math.random() < 0.6 ? 'gym' : 'outdoor';
    
    if (hour >= 9 && hour <= 17) {
      return Math.random() < 0.8 ? 'work' : 'outdoor';
    }
    
    if (activity === 'walking') return Math.random() < 0.7 ? 'outdoor' : 'home';
    return 'home';
  }

  /**
   * Simulate heart rate based on activity and time
   */
  private simulateHeartRate(activity: 'resting' | 'walking' | 'exercising' | 'sleeping', hour: number): number {
    let baseRate = this.baselineMetrics.heartRate;
    
    // Activity-based adjustments
    switch (activity) {
      case 'sleeping':
        baseRate = this.baselineMetrics.heartRate - 15;
        break;
      case 'walking':
        baseRate = this.baselineMetrics.heartRate + 20;
        break;
      case 'exercising':
        baseRate = this.baselineMetrics.heartRate + 50;
        break;
      case 'resting':
        baseRate = this.baselineMetrics.heartRate;
        break;
    }

    // Circadian rhythm adjustments
    const circadianAdjustment = Math.sin((hour - 6) * Math.PI / 12) * 5;
    baseRate += circadianAdjustment;

    // Add realistic noise
    const noise = (Math.random() - 0.5) * 8;
    const heartRate = Math.max(50, Math.min(180, baseRate + noise));

    return Math.round(heartRate);
  }

  /**
   * Simulate blood oxygen levels
   */
  private simulateBloodOxygen(activity: 'resting' | 'walking' | 'exercising' | 'sleeping'): number {
    let baseOxygen = this.baselineMetrics.bloodOxygen;
    
    // Activity-based adjustments
    switch (activity) {
      case 'sleeping':
        baseOxygen = this.baselineMetrics.bloodOxygen - 0.5;
        break;
      case 'exercising':
        baseOxygen = this.baselineMetrics.bloodOxygen - 1;
        break;
      default:
        baseOxygen = this.baselineMetrics.bloodOxygen;
    }

    // Add realistic noise
    const noise = (Math.random() - 0.5) * 2;
    const bloodOxygen = Math.max(90, Math.min(100, baseOxygen + noise));

    return Math.round(bloodOxygen * 10) / 10;
  }

  /**
   * Simulate body temperature
   */
  private simulateTemperature(activity: 'resting' | 'walking' | 'exercising' | 'sleeping', hour: number): number {
    let baseTemp = this.baselineMetrics.temperature;
    
    // Activity-based adjustments
    switch (activity) {
      case 'sleeping':
        baseTemp = this.baselineMetrics.temperature - 0.5;
        break;
      case 'exercising':
        baseTemp = this.baselineMetrics.temperature + 1;
        break;
      case 'walking':
        baseTemp = this.baselineMetrics.temperature + 0.3;
        break;
      default:
        baseTemp = this.baselineMetrics.temperature;
    }

    // Circadian rhythm adjustments
    const circadianAdjustment = Math.sin((hour - 18) * Math.PI / 12) * 0.5;
    baseTemp += circadianAdjustment;

    // Add realistic noise
    const noise = (Math.random() - 0.5) * 0.6;
    const temperature = Math.max(96, Math.min(102, baseTemp + noise));

    return Math.round(temperature * 10) / 10;
  }

  /**
   * Simulate step count
   */
  private simulateSteps(activity: 'resting' | 'walking' | 'exercising' | 'sleeping'): number {
    let stepIncrement = 0;
    
    switch (activity) {
      case 'sleeping':
        stepIncrement = 0;
        break;
      case 'resting':
        stepIncrement = Math.floor(Math.random() * 10);
        break;
      case 'walking':
        stepIncrement = Math.floor(Math.random() * 150) + 50;
        break;
      case 'exercising':
        stepIncrement = Math.floor(Math.random() * 300) + 100;
        break;
    }

    this.currentSteps += stepIncrement;
    return this.currentSteps;
  }

  /**
   * Simulate sleep quality
   */
  private simulateSleepQuality(hour: number): number {
    // Sleep quality is only relevant during sleep hours
    if (hour < 7 || hour >= 23) {
      // Simulate sleep quality based on sleep stage
      const sleepStage = this.determineSleepStage(hour);
      let quality = this.baselineMetrics.sleepQuality;
      
      switch (sleepStage) {
        case 'deep':
          quality = 8 + Math.random() * 2;
          break;
        case 'rem':
          quality = 7 + Math.random() * 2;
          break;
        case 'light':
          quality = 6 + Math.random() * 2;
          break;
        case 'awake':
          quality = 3 + Math.random() * 2;
          break;
      }
      
      return Math.max(0, Math.min(10, quality));
    }
    
    return this.baselineMetrics.sleepQuality;
  }

  /**
   * Determine sleep stage
   */
  private determineSleepStage(hour: number): 'deep' | 'rem' | 'light' | 'awake' {
    if (hour >= 8 && hour <= 22) return 'awake';
    
    // Simulate sleep cycles
    const sleepHour = hour <= 7 ? hour + 24 : hour;
    const cyclePosition = (sleepHour - 23) % 1.5; // 90-minute cycles
    
    if (cyclePosition < 0.3) return 'light';
    if (cyclePosition < 0.7) return 'deep';
    if (cyclePosition < 1.2) return 'rem';
    return 'light';
  }

  /**
   * Simulate stress level
   */
  private simulateStressLevel(hour: number, activity: 'resting' | 'walking' | 'exercising' | 'sleeping'): number {
    let baseStress = this.baselineMetrics.stressLevel;
    
    // Time-based stress patterns
    if (hour >= 9 && hour <= 17) {
      // Work hours - higher stress
      baseStress += 2;
    } else if (hour >= 18 && hour <= 20) {
      // Evening - moderate stress
      baseStress += 1;
    } else if (hour >= 23 || hour <= 7) {
      // Sleep hours - lower stress
      baseStress -= 2;
    }

    // Activity-based adjustments
    switch (activity) {
      case 'sleeping':
        baseStress = 1;
        break;
      case 'exercising':
        baseStress -= 1; // Exercise reduces stress
        break;
      case 'walking':
        baseStress -= 0.5;
        break;
    }

    // Add realistic noise
    const noise = (Math.random() - 0.5) * 2;
    const stressLevel = Math.max(0, Math.min(10, baseStress + noise));

    return Math.round(stressLevel * 10) / 10;
  }

  /**
   * Introduce anomalies for testing
   */
  introduceAnomaly(data: SimulatedHealthData[], anomalyType: 'heartRate' | 'bloodOxygen' | 'temperature' | 'stress'): SimulatedHealthData[] {
    const anomalousData = [...data];
    const anomalyIndex = Math.floor(Math.random() * anomalousData.length);
    
    switch (anomalyType) {
      case 'heartRate':
        anomalousData[anomalyIndex].heartRate = 110 + Math.random() * 30;
        break;
      case 'bloodOxygen':
        anomalousData[anomalyIndex].bloodOxygen = 88 + Math.random() * 5;
        break;
      case 'temperature':
        anomalousData[anomalyIndex].temperature = 100 + Math.random() * 2;
        break;
      case 'stress':
        anomalousData[anomalyIndex].stressLevel = 8 + Math.random() * 2;
        break;
    }
    
    return anomalousData;
  }

  /**
   * Generate a full day of data
   */
  generateDayData(date: Date): SimulatedHealthData[] {
    const startTime = new Date(date);
    startTime.setHours(0, 0, 0, 0);
    
    const endTime = new Date(date);
    endTime.setHours(23, 59, 59, 999);
    
    return this.generateHistoricalData(startTime.getTime(), endTime.getTime(), 5);
  }

  /**
   * Generate a week of data
   */
  generateWeekData(startDate: Date): SimulatedHealthData[] {
    const data: SimulatedHealthData[] = [];
    
    for (let i = 0; i < 7; i++) {
      const currentDate = new Date(startDate);
      currentDate.setDate(startDate.getDate() + i);
      
      const dayData = this.generateDayData(currentDate);
      data.push(...dayData);
    }
    
    return data;
  }
}