import os
import json
import argparse
import numpy as np
import pickle
import pandas as pd
from dynamic_data_processor import DynamicDataProcessor
from data_processing import generate_synthetic_data
from bayesian_model import train_and_evaluate
from visualization import save_all_plots

def print_banner(text, char='='):
    """Print a banner with the given text."""
    width = len(text) + 6
    print(char * width)
    print(f"{char * 2} {text} {char * 2}")
    print(char * width)

def save_model(model, model_type, config):
    """Save a model to disk using pickle."""
    output_dir = config["output"]["directories"]["models"]
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = f'{output_dir}/{model_type}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_data_from_csv(config):
    """Load data from a CSV file specified in the config."""
    data_source = config["data_source"]
    file_path = data_source["csv_file_path"]
    
    if not os.path.exists(file_path):
        print(f"ERROR: CSV file not found at {file_path}")
        print("Please check the file path in your config file")
        print("Defaulting to synthetic data...")
        return None
    
    try:
        print(f"Loading data from {file_path}...")
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
            print(f"Available columns: {', '.join(df.columns)}")
            print("Defaulting to synthetic data...")
            return None
        
        print(f"Successfully loaded {len(df)} records with {df.columns.size} columns")
        return df
        
    except Exception as e:
        print(f"ERROR loading CSV file: {e}")
        print("Defaulting to synthetic data...")
        return None

def main(config_path="fraud-detection--main/config.json", compare_structures=False):
    """Run the complete fraud detection pipeline using configuration file."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create output directories
    print_banner("Fraud Detection with Bayesian Networks")
    print("\nInitializing pipeline...")
    
    for directory in config["output"]["directories"].values():
        os.makedirs(directory, exist_ok=True)
    
    # Create subdirectories for different estimator plots
    for estimator in config["bayesian_model"]["estimators"]:
        if config["bayesian_model"]["estimators"][estimator]["enabled"]:
            os.makedirs(f"{config['output']['directories']['plots']}/{estimator}", exist_ok=True)
    
    print("Starting fraud detection pipeline...\n")
    
    # Get data either from CSV or generate synthetic data
    print_banner("Data Preparation", "-")
    
    df = None
    if config["data_source"]["use_real_data"]:
        df = load_data_from_csv(config)
    
    # Fall back to synthetic data if loading failed or not configured
    if df is None:
        print("Generating synthetic data...")
        df = generate_synthetic_data(
            num_samples=config["data_processing"]["num_samples"],
            fraud_ratio=config["data_processing"]["fraud_ratio"]
        )
    
    # Process data using dynamic processor
    print("Processing data using dynamic processor...")
    data_processor = DynamicDataProcessor(config_path)
    X_train, X_test, y_train, y_test = data_processor.process_pipeline(df)
    
    print(f"Prepared training set with {len(X_train)} samples and {X_train.columns.size} features")
    print(f"Prepared test set with {len(X_test)} samples")
    
    # Train and evaluate models with different estimators
    print_banner("Model Training and Evaluation", "-")
    print("\nComparing estimation methods:")
    
    # Store results for all enabled estimators
    models = {}
    all_metrics = {}
    
    # Train MLE model if enabled
    if config["bayesian_model"]["estimators"]["mle"]["enabled"]:
        print("\nTraining model with MLE estimator...")
        mle_model, mle_metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            estimator_type='mle',
            threshold=config["bayesian_model"]["prediction"]["mle"]["threshold"]
        )
        
        # Save model if enabled in config
        if config["output"]["save_models"]:
            save_model(mle_model, 'mle', config)
        
        # Store model and metrics
        models['mle'] = mle_model
        all_metrics['mle'] = mle_metrics
    
    # Train Bayesian model if enabled
    if config["bayesian_model"]["estimators"]["bayes"]["enabled"]:
        print("\nTraining model with BAYES estimator...")
        bayes_config = config["bayesian_model"]["estimators"]["bayes"]
        print(f"Using Bayesian estimation with equivalent_sample_size={bayes_config['equivalent_sample_size']}")
        
        bayes_model, bayes_metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            estimator_type='bayes',
            equivalent_sample_size=bayes_config['equivalent_sample_size'],
            prior_type=bayes_config['prior_type'],
            threshold=config["bayesian_model"]["prediction"]["bayes"]["threshold"]
        )
        
        # Save model if enabled in config
        if config["output"]["save_models"]:
            save_model(bayes_model, 'bayes', config)
        
        # Store model and metrics
        models['bayes'] = bayes_model
        all_metrics['bayes'] = bayes_metrics
    
    # Store results for comparison
    results = {
        estimator: {k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
        for estimator, metrics in all_metrics.items()
    }
    
    # Save metrics and generate plots for each estimator
    for estimator, metrics in all_metrics.items():
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {
            k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in metrics.items()
        }
        
        # Save detailed metrics to JSON if enabled
        if config["output"]["save_metrics"]:
            with open(f"{config['output']['directories']['results']}/{estimator}_metrics.json", 'w') as f:
                json.dump(metrics_json, f, indent=4)
        
        # Generate and save plots if enabled
        print(f"Generating plots for {estimator.upper()} estimator...")
        if config["output"]["plots"]["generate_confusion_matrix"] or \
           config["output"]["plots"]["generate_roc_curve"] or \
           config["output"]["plots"]["generate_metrics_plot"] or \
           config["output"]["plots"]["generate_network_structure"]:
            
            save_all_plots(
                metrics,
                models[estimator],
                pca_explained_variance_ratio=data_processor.get_pca_stats()['explained_variance_ratio'] if data_processor.get_pca_stats() else None,
                output_dir=f"{config['output']['directories']['plots']}/{estimator}",
                plot_config=config["output"]["plots"]
            )
    
    # Print comparison of results if we have multiple estimators
    if len(results) > 1:
        print_banner("Results Comparison", "-")
        print("\nPerformance Metrics Comparison:")
        
        # Get list of estimators and metrics
        estimators = list(results.keys())
        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Print header
        header = "{:<10}".format("Metric")
        for estimator in estimators:
            header += " {:<10}".format(estimator.upper())
        if len(estimators) > 1:
            header += " {:<10}".format("Difference")
        print("\n" + header)
        print("-" * (12 + 12 * len(estimators)))
        
        # Print metrics for each estimator
        for metric in metrics_list:
            line = "{:<10}".format(metric)
            for estimator in estimators:
                line += " {:<10.4f}".format(results[estimator][metric])
            
            # Add difference between estimators if we have exactly 2
            if len(estimators) == 2:
                diff = results[estimators[1]][metric] - results[estimators[0]][metric]
                line += " {:<10}".format(f"{diff:+.4f}")
            
            print(line)
        
        # Save comparison as JSON if enabled
        if config["output"]["save_metrics"] and len(estimators) == 2:
            comparison_results = {
                estimators[0]: results[estimators[0]],
                estimators[1]: results[estimators[1]],
                'differences': {
                    metric: float(results[estimators[1]][metric] - results[estimators[0]][metric])
                    for metric in metrics_list
                }
            }
            
            with open(f"{config['output']['directories']['results']}/estimation_methods_comparison.json", 'w') as f:
                json.dump(comparison_results, f, indent=4)
    
    print("\nKey Observations:")
    if 'bayes' in results and 'mle' in results:
        if results['bayes']['accuracy'] > results['mle']['accuracy']:
            print("- Bayesian estimation shows better overall accuracy")
        if results['bayes']['precision'] > results['mle']['precision']:
            print("- Bayesian estimation demonstrates better precision")
        if results['mle']['recall'] > results['bayes']['recall']:
            print("- MLE shows better recall")
        print("- Bayesian estimation generally provides more robust models less prone to overfitting")
        print("- MLE may capture more of the training data patterns but can be less generalizable")
    
    print("\nPipeline completed successfully!")
    print(f"Results have been saved to the '{config['output']['directories']['results']}' directory")
    print(f"Plots have been saved to the '{config['output']['directories']['plots']}' directory")
    
    # Run structure comparison if requested
    if compare_structures and config["bayesian_model"]["structure"]["learn_from_data"]:
        print_banner("Network Structure Comparison", "-")
        print("\nComparing manually-defined structure vs. learned structure...")
        from structure_learning import run_structure_comparison
        run_structure_comparison(config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Detection with Bayesian Networks')
    parser.add_argument('--config', type=str, default='fraud-detection--main/config.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--compare-structures', action='store_true', 
                        help='Run structure comparison after main pipeline')
    
    args = parser.parse_args()
    main(config_path=args.config, compare_structures=args.compare_structures) 