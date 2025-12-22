#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eler: Ensemble Learning-based Automated Verification of Code Clones
Ensemble Learning Module

This script trains and evaluates the ensemble learning model for clone verification.
"""

import argparse
import csv
import os
import sys
from itertools import islice
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Eler Ensemble Learning - Train and evaluate clone verification model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python Ensemble_learning.py -d ./output/ -o ./models/ -m train
  
  # Evaluation (5-fold cross-validation)
  python Ensemble_learning.py -d ./output/ -o ./results/ -m evaluate -t 6
  
  # Prediction with pre-trained models
  python Ensemble_learning.py -d ./output/ -o ./results/ -m predict --model_dir ./models/
        """
    )
    parser.add_argument(
        '-d', '--dir',
        help='Directory containing feature CSV files (clone and non-clone)',
        required=True,
        type=str
    )
    parser.add_argument(
        '-o', '--output',
        help='Directory to save trained models or results',
        required=True,
        type=str
    )
    parser.add_argument(
        '-m', '--mode',
        help='Operation mode: train, predict, or evaluate (default: evaluate)',
        choices=['train', 'predict', 'evaluate'],
        default='evaluate',
        type=str
    )
    parser.add_argument(
        '-t', '--threshold',
        help='Voting threshold n for majority voting (1-11, default: 6)',
        type=int,
        default=6
    )
    parser.add_argument(
        '--model_dir',
        help='Directory containing pre-trained models (for predict mode)',
        type=str,
        default='./models/'
    )
    parser.add_argument(
        '-k', '--kfold',
        help='Number of folds for cross-validation (default: 5)',
        type=int,
        default=5
    )
    parser.add_argument(
        '--seed',
        help='Random seed for reproducibility (default: 42)',
        type=int,
        default=42
    )
    
    return parser.parse_args()


def feature_extraction_all(feature_csv):
    """
    Extract feature vectors from CSV file.
    
    Args:
        feature_csv: Path to feature CSV file
        
    Returns:
        List of feature vectors
    """
    features = []
    
    if not os.path.exists(feature_csv):
        print(f"Warning: File not found - {feature_csv}")
        return features
    
    with open(feature_csv, 'r') as f:
        data = csv.reader(f)
        for line in islice(data, 1, None):  # Skip header
            try:
                feature = [float(i) for i in line[2:]]  # Skip ID columns
                features.append(feature)
            except Exception as e:
                pass
    
    print(f"  Loaded {len(features)} samples from {os.path.basename(feature_csv)}")
    return features


def obtain_dataset(dir_path):
    """
    Load dataset from feature CSV files.
    
    Args:
        dir_path: Directory containing feature CSV files
        
    Returns:
        Tuple of (vectors, labels)
    """
    # Define expected file patterns
    clone_patterns = [
        'type-1_sim.csv', 'type-2_sim.csv', 'type-3_sim.csv',
        'type-4_sim.csv', 'type-5_sim.csv', 'type-6_sim.csv',
        'T1_sim.csv', 'T2_sim.csv', 'VST3_sim.csv', 
        'ST3_sim.csv', 'MT3_sim.csv', 'WT3T4_sim.csv'
    ]
    nonclone_patterns = ['noclone_sim.csv', 'nonclone_sim.csv', 'BCB_nonclone_sim.csv']
    
    Vectors = []
    Labels = []
    
    print("\nLoading clone pairs...")
    for pattern in clone_patterns:
        filepath = os.path.join(dir_path, pattern)
        if os.path.exists(filepath):
            features = feature_extraction_all(filepath)
            Vectors.extend(features)
            Labels.extend([1 for _ in range(len(features))])
    
    print("\nLoading non-clone pairs...")
    for pattern in nonclone_patterns:
        filepath = os.path.join(dir_path, pattern)
        if os.path.exists(filepath):
            features = feature_extraction_all(filepath)
            Vectors.extend(features)
            Labels.extend([0 for _ in range(len(features))])
    
    print(f"\nTotal samples: {len(Vectors)}")
    print(f"  Clone pairs: {sum(Labels)}")
    print(f"  Non-clone pairs: {len(Labels) - sum(Labels)}")
    
    return Vectors, Labels


def random_features(vectors, labels, seed=42):
    """Shuffle feature vectors and labels together."""
    random.seed(seed)
    Vec_Lab = []
    
    for i in range(len(vectors)):
        vec = vectors[i].copy()
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)
    
    random.shuffle(Vec_Lab)
    
    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]


def get_classifiers():
    """Return dictionary of classifiers."""
    return {
        'knn_1': KNeighborsClassifier(n_neighbors=1),
        'knn_3': KNeighborsClassifier(n_neighbors=3),
        'knn_5': KNeighborsClassifier(n_neighbors=5),
        'decision_tree': DecisionTreeClassifier(),
        'adaboost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=64), random_state=0),
        'gdbt': GradientBoostingClassifier(max_depth=64, random_state=0),
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'NearestCentroid': NearestCentroid(),
        'RidgeClassifier': RidgeClassifier(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis()
    }


def train_models(train_X, train_Y, output_dir):
    """
    Train all classifiers and save models.
    
    Args:
        train_X: Training features
        train_Y: Training labels
        output_dir: Directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    classifiers = get_classifiers()
    
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        clf.fit(train_X, train_Y)
        model_path = os.path.join(output_dir, f'clf_{name}.pkl')
        joblib.dump(clf, model_path)
        print(f"    Saved to {model_path}")


def load_models(model_dir):
    """Load pre-trained models from directory."""
    classifiers = {}
    model_names = [
        'knn_1', 'knn_3', 'knn_5', 'decision_tree', 'adaboost',
        'gdbt', 'GaussianNB', 'LogisticRegression', 'NearestCentroid',
        'RidgeClassifier', 'QuadraticDiscriminantAnalysis'
    ]
    
    for name in model_names:
        model_path = os.path.join(model_dir, f'clf_{name}.pkl')
        if os.path.exists(model_path):
            classifiers[name] = joblib.load(model_path)
        else:
            print(f"Warning: Model not found - {model_path}")
    
    return classifiers


def ensemble_predict(classifiers, test_X, threshold=6):
    """
    Make ensemble predictions using majority voting.
    
    Args:
        classifiers: Dictionary of trained classifiers
        test_X: Test features
        threshold: Voting threshold (n)
        
    Returns:
        Ensemble predictions, individual predictions
    """
    predictions = {}
    
    for name, clf in classifiers.items():
        predictions[name] = clf.predict(test_X)
    
    # Majority voting
    y_pred = np.zeros(len(test_X), dtype=int)
    for i in range(len(test_X)):
        votes = sum(pred[i] for pred in predictions.values())
        if votes >= threshold:
            y_pred[i] = 1
    
    return y_pred, predictions


def evaluate_kfold(vectors, labels, k=5, threshold=6, output_dir=None, seed=42):
    """
    Perform k-fold cross-validation.
    
    Args:
        vectors: Feature vectors
        labels: Labels
        k: Number of folds
        threshold: Voting threshold
        output_dir: Directory to save models (optional)
        seed: Random seed
        
    Returns:
        Average metrics (F1, Precision, Recall)
    """
    X = np.array(vectors)
    Y = np.array(labels)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    metrics = {'f1': [], 'precision': [], 'recall': []}
    
    print(f"\n{'='*60}")
    print(f"  {k}-Fold Cross-Validation (threshold={threshold})")
    print(f"{'='*60}")
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]
        
        # Train classifiers
        classifiers = get_classifiers()
        for name, clf in classifiers.items():
            clf.fit(train_X, train_Y)
        
        # Ensemble prediction
        y_pred, _ = ensemble_predict(classifiers, test_X, threshold)
        
        # Calculate metrics
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        
        metrics['f1'].append(f1)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        
        print(f"  Fold {fold}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Save models from last fold if output_dir specified
        if fold == k and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for name, clf in classifiers.items():
                model_path = os.path.join(output_dir, f'clf_{name}.pkl')
                joblib.dump(clf, model_path)
    
    print(f"\n{'='*60}")
    print(f"  Average: F1={np.mean(metrics['f1']):.4f}, "
          f"Precision={np.mean(metrics['precision']):.4f}, "
          f"Recall={np.mean(metrics['recall']):.4f}")
    print(f"{'='*60}")
    
    return metrics


def predict_with_models(vectors, labels, model_dir, threshold=6):
    """
    Make predictions using pre-trained models.
    
    Args:
        vectors: Feature vectors
        labels: True labels (for evaluation)
        model_dir: Directory containing pre-trained models
        threshold: Voting threshold
    """
    test_X = np.array(vectors)
    test_Y = np.array(labels)
    
    # Load models
    classifiers = load_models(model_dir)
    
    if len(classifiers) == 0:
        print("Error: No models found.")
        return
    
    # Make predictions
    y_pred, individual_preds = ensemble_predict(classifiers, test_X, threshold)
    
    # Calculate metrics
    precision = precision_score(y_true=test_Y, y_pred=y_pred)
    recall = recall_score(y_true=test_Y, y_pred=y_pred)
    f1 = f1_score(y_true=test_Y, y_pred=y_pred)
    
    print(f"\nPrediction Results (threshold={threshold}):")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Print individual model predictions
    print("\nIndividual Model Predictions:")
    for name, pred in individual_preds.items():
        p = precision_score(y_true=test_Y, y_pred=pred)
        r = recall_score(y_true=test_Y, y_pred=pred)
        print(f"  {name}: Precision={p:.4f}, Recall={r:.4f}")


def main():
    """Main function."""
    args = parse_args()
    
    # Validate paths
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Validate threshold
    if not 1 <= args.threshold <= 11:
        print(f"Error: Threshold must be between 1 and 11.")
        sys.exit(1)
    
    # Load dataset
    print(f"Loading dataset from: {args.dir}")
    Vectors, Labels = obtain_dataset(args.dir)
    
    if len(Vectors) == 0:
        print("Error: No data loaded. Check the directory and file formats.")
        sys.exit(1)
    
    # Shuffle data
    vectors, labels = random_features(Vectors, Labels, seed=args.seed)
    
    # Execute based on mode
    if args.mode == 'train':
        print(f"\nTraining mode - saving models to: {args.output}")
        X = np.array(vectors)
        Y = np.array(labels)
        train_models(X, Y, args.output)
        print("\nTraining completed!")
        
    elif args.mode == 'evaluate':
        print(f"\nEvaluation mode - {args.kfold}-fold cross-validation")
        evaluate_kfold(vectors, labels, k=args.kfold, threshold=args.threshold, 
                       output_dir=args.output, seed=args.seed)
        
    elif args.mode == 'predict':
        print(f"\nPrediction mode - using models from: {args.model_dir}")
        predict_with_models(vectors, labels, args.model_dir, threshold=args.threshold)


if __name__ == '__main__':
    main()
