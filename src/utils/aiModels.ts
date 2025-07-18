// AI Models for Health Monitoring System
// This file contains the implementation of AI algorithms for anomaly detection and health recommendations

export interface HealthDataPoint {
  heartRate: number;
  bloodOxygen: number;
  temperature: number;
  steps: number;
  sleepQuality: number;
  stressLevel: number;
  timestamp: number;
}

export interface AnomalyResult {
  isAnomaly: boolean;
  confidence: number;
  metric: string;
  description: string;
}

// Isolation Forest implementation for anomaly detection
export class IsolationForest {
  private trees: IsolationTree[] = [];
  private numTrees = 100;
  private subSampleSize = 256;
  private contamination = 0.1;

  constructor(numTrees = 100, subSampleSize = 256) {
    this.numTrees = numTrees;
    this.subSampleSize = subSampleSize;
  }

  // Train the isolation forest model
  train(data: HealthDataPoint[]): void {
    this.trees = [];
    
    for (let i = 0; i < this.numTrees; i++) {
      const sampleData = this.subsample(data, this.subSampleSize);
      const tree = new IsolationTree();
      tree.fit(sampleData);
      this.trees.push(tree);
    }
  }

  // Detect anomalies in new data
  detect(dataPoint: HealthDataPoint): AnomalyResult[] {
    const results: AnomalyResult[] = [];
    
    // Calculate anomaly scores for each metric
    const metrics = [
      { key: 'heartRate', value: dataPoint.heartRate, normal: [60, 100] },
      { key: 'bloodOxygen', value: dataPoint.bloodOxygen, normal: [95, 100] },
      { key: 'temperature', value: dataPoint.temperature, normal: [97, 99.5] },
      { key: 'stressLevel', value: dataPoint.stressLevel, normal: [0, 5] },
    ];

    metrics.forEach(metric => {
      const score = this.calculateAnomalyScore(dataPoint);
      const isAnomaly = score > 0.6 || 
                       metric.value < metric.normal[0] || 
                       metric.value > metric.normal[1];
      
      if (isAnomaly) {
        results.push({
          isAnomaly: true,
          confidence: score,
          metric: metric.key,
          description: this.getAnomalyDescription(metric.key, metric.value, metric.normal),
        });
      }
    });

    return results;
  }

  private subsample(data: HealthDataPoint[], size: number): HealthDataPoint[] {
    const shuffled = [...data].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, Math.min(size, data.length));
  }

  private calculateAnomalyScore(dataPoint: HealthDataPoint): number {
    let totalPathLength = 0;
    
    this.trees.forEach(tree => {
      totalPathLength += tree.pathLength(dataPoint);
    });
    
    const avgPathLength = totalPathLength / this.trees.length;
    const c = this.c(this.subSampleSize);
    
    return Math.pow(2, -avgPathLength / c);
  }

  private c(n: number): number {
    return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n);
  }

  private getAnomalyDescription(metric: string, value: number, normal: number[]): string {
    const descriptions = {
      heartRate: value > normal[1] ? 'Heart rate is elevated above normal range' : 'Heart rate is below normal range',
      bloodOxygen: 'Blood oxygen levels are concerning',
      temperature: value > normal[1] ? 'Body temperature is elevated' : 'Body temperature is low',
      stressLevel: 'Stress levels are elevated',
    };
    
    return descriptions[metric as keyof typeof descriptions] || 'Metric is outside normal range';
  }
}

// Isolation Tree implementation
class IsolationTree {
  private root: TreeNode | null = null;
  private maxDepth = 10;

  fit(data: HealthDataPoint[]): void {
    this.root = this.buildTree(data, 0);
  }

  pathLength(dataPoint: HealthDataPoint): number {
    return this.calculatePathLength(this.root, dataPoint, 0);
  }

  private buildTree(data: HealthDataPoint[], depth: number): TreeNode | null {
    if (depth >= this.maxDepth || data.length <= 1) {
      return new TreeNode(data.length);
    }

    const features = ['heartRate', 'bloodOxygen', 'temperature', 'stressLevel'];
    const randomFeature = features[Math.floor(Math.random() * features.length)];
    
    const values = data.map(d => d[randomFeature as keyof HealthDataPoint] as number);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    
    if (minVal === maxVal) {
      return new TreeNode(data.length);
    }

    const splitValue = Math.random() * (maxVal - minVal) + minVal;
    
    const leftData = data.filter(d => (d[randomFeature as keyof HealthDataPoint] as number) < splitValue);
    const rightData = data.filter(d => (d[randomFeature as keyof HealthDataPoint] as number) >= splitValue);

    const node = new TreeNode(data.length);
    node.feature = randomFeature;
    node.splitValue = splitValue;
    node.left = this.buildTree(leftData, depth + 1);
    node.right = this.buildTree(rightData, depth + 1);

    return node;
  }

  private calculatePathLength(node: TreeNode | null, dataPoint: HealthDataPoint, depth: number): number {
    if (!node || node.isLeaf()) {
      return depth + (node ? this.c(node.size) : 0);
    }

    const featureValue = dataPoint[node.feature as keyof HealthDataPoint] as number;
    
    if (featureValue < node.splitValue) {
      return this.calculatePathLength(node.left, dataPoint, depth + 1);
    } else {
      return this.calculatePathLength(node.right, dataPoint, depth + 1);
    }
  }

  private c(n: number): number {
    if (n <= 1) return 0;
    return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n);
  }
}

// Tree Node class
class TreeNode {
  public size: number;
  public feature: string | null = null;
  public splitValue: number | null = null;
  public left: TreeNode | null = null;
  public right: TreeNode | null = null;

  constructor(size: number) {
    this.size = size;
  }

  isLeaf(): boolean {
    return this.left === null && this.right === null;
  }
}

// Health recommendation system
export class HealthRecommendationSystem {
  generateRecommendations(dataPoint: HealthDataPoint, anomalies: AnomalyResult[]): any[] {
    const recommendations: any[] = [];

    // Heart rate recommendations
    if (dataPoint.heartRate > 100) {
      recommendations.push({
        type: 'medical',
        title: 'Consult Healthcare Provider',
        description: 'Your heart rate is consistently elevated. Consider consulting with a healthcare professional.',
        priority: 'high',
      });
    } else if (dataPoint.heartRate > 90) {
      recommendations.push({
        type: 'exercise',
        title: 'Practice Relaxation',
        description: 'Try deep breathing exercises or meditation to help lower your heart rate.',
        priority: 'medium',
      });
    }

    // Blood oxygen recommendations
    if (dataPoint.bloodOxygen < 95) {
      recommendations.push({
        type: 'medical',
        title: 'Monitor Oxygen Levels',
        description: 'Your blood oxygen is below normal. Consider consulting a healthcare provider.',
        priority: 'high',
      });
    }

    // Step count recommendations
    if (dataPoint.steps < 8000) {
      recommendations.push({
        type: 'exercise',
        title: 'Increase Daily Activity',
        description: 'Aim for at least 8,000 steps per day. Try taking short walks throughout the day.',
        priority: 'medium',
      });
    }

    // Stress level recommendations
    if (dataPoint.stressLevel > 7) {
      recommendations.push({
        type: 'sleep',
        title: 'Stress Management',
        description: 'Your stress levels are high. Consider relaxation techniques or better sleep hygiene.',
        priority: 'high',
      });
    }

    // Sleep quality recommendations
    if (dataPoint.sleepQuality < 6) {
      recommendations.push({
        type: 'sleep',
        title: 'Improve Sleep Quality',
        description: 'Focus on maintaining a consistent sleep schedule and creating a relaxing bedtime routine.',
        priority: 'medium',
      });
    }

    return recommendations;
  }
}

// LSTM model for time series prediction (simplified implementation)
export class LSTMPredictor {
  private weights: number[][] = [];
  private trained = false;

  // Simplified training method
  train(historicalData: HealthDataPoint[]): void {
    // In a real implementation, this would use TensorFlow.js or similar
    // For now, we'll simulate the training process
    this.weights = this.initializeWeights();
    this.trained = true;
  }

  // Predict future health metrics
  predict(recentData: HealthDataPoint[]): HealthDataPoint {
    if (!this.trained || recentData.length === 0) {
      throw new Error('Model not trained or no data provided');
    }

    const lastDataPoint = recentData[recentData.length - 1];
    
    // Simplified prediction logic
    return {
      heartRate: this.predictMetric(recentData.map(d => d.heartRate)),
      bloodOxygen: this.predictMetric(recentData.map(d => d.bloodOxygen)),
      temperature: this.predictMetric(recentData.map(d => d.temperature)),
      steps: this.predictMetric(recentData.map(d => d.steps)),
      sleepQuality: this.predictMetric(recentData.map(d => d.sleepQuality)),
      stressLevel: this.predictMetric(recentData.map(d => d.stressLevel)),
      timestamp: Date.now() + 3600000, // 1 hour in the future
    };
  }

  private initializeWeights(): number[][] {
    // Initialize random weights for the model
    const weights: number[][] = [];
    for (let i = 0; i < 10; i++) {
      weights[i] = [];
      for (let j = 0; j < 6; j++) {
        weights[i][j] = Math.random() * 0.1 - 0.05;
      }
    }
    return weights;
  }

  private predictMetric(values: number[]): number {
    if (values.length === 0) return 0;
    
    // Simple prediction based on recent trends
    const recentValues = values.slice(-5);
    const trend = recentValues.length > 1 ? 
      (recentValues[recentValues.length - 1] - recentValues[0]) / (recentValues.length - 1) : 0;
    
    return recentValues[recentValues.length - 1] + trend;
  }
}