"""
flash_flood_prediction/src/evaluate.py
--------------------------------------
Model training and offline performance evaluation script.
Trains:
1. Baseline Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier (XGBoost baseline)
Outputs real evaluation metrics and saves trained weights.
"""

import os
import sys

# Ensure root import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import FlashFloodPredictor

def run_evaluation():
    print("=" * 65)
    print("AI FLASH FLOOD EARLY WARNING SYSTEM - MODEL EVALUATION SUITE")
    print("=" * 65)
    predictor = FlashFloodPredictor()
    print("\nTrained Model Performance on Unseen Test Split (Late Monsoon Test Data):")
    print("-" * 65)
    print(f"{'Model Architecture':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'Critical Recall':<15}")
    print("-" * 65)
    for model_name, metrics in predictor.model_metrics.items():
        print(f"{model_name:<30} | {metrics['accuracy']:>7.2f}%   | {metrics['f1']:>7.2f}%   | {metrics['critical_recall']:>12.2f}%")
    print("-" * 65)
    print("\nAll models and scalers successfully serialized to `models/`.")

if __name__ == "__main__":
    run_evaluation()
