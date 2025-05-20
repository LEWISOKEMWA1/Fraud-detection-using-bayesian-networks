import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from data_processing import generate_synthetic_data
from dynamic_data_processor import DynamicDataProcessor
from bayesian_model import predict_probabilities
from visualization import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_metrics_comparison
)

def load_model(model_type='bayes', config_path='config.json'):
    """Load a trained model from file."""
    try:
        import pickle
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        model_dir = config['output']['directories']['models']
        model_path = f'{model_dir}/{model_type}_model.pkl'
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found.")
            print(f"Please run the main pipeline first: python src/main.py --config {config_path}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Successfully loaded {model_type.upper()} model.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_data_from_csv(config):
    """Load data from a CSV file specified in the config."""
    data_source = config["data_source"]
    file_path = data_source["csv_file_path"]
    
    if not os.path.exists(file_path):
        print(f"ERROR: CSV file not found at {file_path}")
        return None
    
    try:
        print(f"Loading test data from {file_path}...")
        df = pd.read_csv(
            file_path,
            sep=data_source["csv_separator"],
            encoding=data_source["csv_encoding"],
            header=0 if data_source["data_has_header"] else None
        )
        
        # Ensure target column exists
        target_column = config["data_processing"]["target_column"]
        if target_column not in df.columns:
            print(f"ERROR: Target column '{target_column}' not found in CSV file")
            return None
        
        print(f"Successfully loaded {len(df)} records for testing")
        return df
        
    except Exception as e:
        print(f"ERROR loading CSV file: {e}")
        return None

def generate_test_data(config_path='config.json'):
    """Generate or load and preprocess test data based on configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    df = None
    if config["data_source"]["use_real_data"]:
        df = load_data_from_csv(config)
    
    # Fall back to synthetic data if loading failed or not configured
    if df is None:
        test_config = config['testing']
        data_config = config['data_processing']
        
        num_samples = test_config.get('num_samples', 500)
        fraud_ratio = data_config.get('fraud_ratio', 0.2)
        
        print(f"Generating {num_samples} test samples with {fraud_ratio*100}% fraud ratio...")
        
        # Generate new synthetic data
        df = generate_synthetic_data(num_samples, fraud_ratio)
    
    # Process using the dynamic processor
    processor = DynamicDataProcessor(config_path)
    X, y = processor.process_pipeline(df, return_test_data=False)
    
    print(f"Test data prepared with {X.shape[1]} features.")
    return X, y, processor

def evaluate_on_test_data(model, X_test, y_test, config_path='config.json', model_type='bayes'):
    """Evaluate model performance on test data."""
    if model is None:
        return None
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Evaluating {model_type.upper()} model on test data...")
    
    # Get probability predictions
    y_probs = predict_probabilities(model, X_test)
    
    # Get threshold from config
    threshold = config['bayesian_model']['prediction'][model_type]['threshold']
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calculate metrics
    y_true = y_test.astype(int)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_pred': y_pred,
        'y_probs': y_probs
    }
    
    return metrics

def display_and_save_results(metrics, model_type='bayes', config_path='config.json'):
    """Display and save evaluation results."""
    if metrics is None:
        return
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    output_dir = config['output']['directories']['test_results']
    os.makedirs(output_dir, exist_ok=True)
    
    # Print metrics
    print("\nTest Results:")
    print("-" * 40)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f"{metric.capitalize()}: {metrics[metric]:.4f}")
    
    # Save metrics to JSON
    metrics_json = {
        k: v.tolist() if hasattr(v, 'tolist') else v
        for k, v in metrics.items()
    }
    
    with open(f'{output_dir}/{model_type}_test_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Generate and save plots
    plot_config = config['output']['plots']
    
    # Confusion matrix
    if plot_config.get('generate_confusion_matrix', True):
        cm_fig = plot_confusion_matrix(metrics['confusion_matrix'], 
                                      title=f'{model_type.upper()} Model Confusion Matrix (Test Data)')
        cm_fig.savefig(f'{output_dir}/{model_type}_test_confusion_matrix.png')
    
    # ROC curve
    if plot_config.get('generate_roc_curve', True):
        roc_fig = plot_roc_curve(metrics['y_pred'], metrics['y_probs'],
                                title=f'{model_type.upper()} Model ROC Curve (Test Data)')
        roc_fig.savefig(f'{output_dir}/{model_type}_test_roc_curve.png')
    
    # Metrics comparison
    if plot_config.get('generate_metrics_plot', True):
        metrics_fig = plot_metrics_comparison(
            {k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']},
            title=f'{model_type.upper()} Model Metrics (Test Data)'
        )
        metrics_fig.savefig(f'{output_dir}/{model_type}_test_metrics.png')
    
    plt.close('all')
    print(f"Results and plots saved to {output_dir}/")

def vary_fraud_ratio_test(config_path='config.json', model_type='bayes'):
    """Test model performance across different fraud ratios defined in config."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = load_model(model_type=model_type, config_path=config_path)
    if model is None:
        return
    
    # Skip fraud ratio testing if using real data
    if config["data_source"]["use_real_data"]:
        print("\nSkipping fraud ratio testing when using real data.")
        return
    
    # Get fraud ratios from config
    fraud_ratios = config['testing']['fraud_ratios']
    results = {}
    
    print("\nTesting model performance across different fraud ratios...")
    for ratio in fraud_ratios:
        print(f"\nGenerating test data with {ratio*100}% fraud ratio...")
        
        # Generate synthetic data with current fraud ratio
        df = generate_synthetic_data(
            num_samples=config['testing']['num_samples'],
            fraud_ratio=ratio
        )
        
        # Process using the dynamic processor
        processor = DynamicDataProcessor(config_path)
        X_test, y_test = processor.process_pipeline(df, return_test_data=False)
        
        # Evaluate model
        metrics = evaluate_on_test_data(model, X_test, y_test, config_path=config_path, model_type=model_type)
        
        results[ratio] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc']
        }
    
    # Save results
    output_dir = config['output']['directories']['test_results']
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/fraud_ratio_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics_to_plot:
        values = [results[ratio][metric] for ratio in fraud_ratios]
        plt.plot(fraud_ratios, values, marker='o', label=metric.capitalize())
    
    plt.xlabel('Fraud Ratio')
    plt.ylabel('Metric Value')
    plt.title('Model Performance Across Different Fraud Ratios')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fraud_ratios, [f"{r*100}%" for r in fraud_ratios])
    
    plt.savefig(f'{output_dir}/fraud_ratio_comparison.png')
    plt.close()
    
    print(f"Fraud ratio comparison results saved to {output_dir}/")

def main(config_path='config.json'):
    """Main function to test the model on new data."""
    print("Bank Fraud Detection - Model Testing")
    print("====================================")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get model type from args or use default
    model_type = 'bayes'  # Default model type
    
    # Ensure model directory exists
    model_dir = config['output']['directories']['models']
    if not os.path.exists(model_dir):
        print(f"Models directory '{model_dir}' not found. Please run main.py first to train models.")
        print(f"Run: python src/main.py --config {config_path}")
        return
    
    # First test with default parameters
    model = load_model(model_type=model_type, config_path=config_path)
    if model:
        X_test, y_test, _ = generate_test_data(config_path=config_path)
        metrics = evaluate_on_test_data(model, X_test, y_test, config_path=config_path, model_type=model_type)
        display_and_save_results(metrics, model_type=model_type, config_path=config_path)
        
        # Test model across different fraud ratios
        vary_fraud_ratio_test(config_path=config_path, model_type=model_type)
    
    print("\nTesting completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Detection Model Testing')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--model-type', type=str, default='bayes',
                        choices=['mle', 'bayes'],
                        help='Model type to test (default: bayes)')
    
    args = parser.parse_args()
    main(config_path=args.config) 