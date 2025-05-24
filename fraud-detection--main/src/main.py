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
from structure_learning import run_structure_learning_process # Added import

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

    if config["bayesian_model"]["structure"]["learn_from_data"]:
        print_banner("Structure Learning and Parameter Estimation", "-")
        # Call the enhanced structure learning process
        # This function will handle its own model training, evaluation, comparison, and saving of results.
        # Note: run_structure_learning_process currently takes config_path and handles data internally.
        # If it were to take X_train etc., its internal logic would need to change.
        # For now, calling with config_path as per its current definition.
        structure_learning_results = run_structure_learning_process(
            config_path, X_train, X_test, y_train, y_test, data_processor
        )
        
        print("\nStructure learning process complete.")
        print(f"Results, learned DAG, and model comparisons saved in '{config['output']['directories']['results']}' and '{config['output']['directories']['plots']}'.")
        # The main.py's subsequent model training, plotting, and comparison is skipped here,
        # as structure_learning.py now handles this comprehensively for the learn_from_data=True case.
    
    else: # Original path: learn_from_data is False
        print_banner("Model Training and Evaluation (Predefined Structure)", "-")
        # Existing logic for training MLE and/or Bayesian models using train_and_evaluate
        # This part uses manually specified structures or the default from bayesian_model.build_network_structure.
        
        # Store results for all enabled estimators
        models = {}
        all_metrics = {}
        
        # Train MLE model if enabled
        if config["bayesian_model"]["estimators"]["mle"]["enabled"]:
            print("\nTraining model with MLE estimator...")
            # Pass full config to train_and_evaluate
            mle_model, mle_metrics = train_and_evaluate(
                X_train, X_test, y_train, y_test, config, 
                estimator_type='mle',
                threshold=config["bayesian_model"]["prediction"]["mle"]["threshold"]
            )
            if config["output"]["save_models"]:
                save_model(mle_model, 'mle', config)
            models['mle'] = mle_model
            all_metrics['mle'] = mle_metrics

        # Train Bayesian model if enabled
        if config["bayesian_model"]["estimators"]["bayes"]["enabled"]:
            print("\nTraining model with BAYES estimator...")
            bayes_config_params = config["bayesian_model"]["estimators"]["bayes"]
            print(f"Using Bayesian estimation with equivalent_sample_size={bayes_config_params['equivalent_sample_size']}")
            # Pass full config to train_and_evaluate
            bayes_model, bayes_metrics = train_and_evaluate(
                X_train, X_test, y_train, y_test, config, 
                estimator_type='bayes',
                equivalent_sample_size=bayes_config_params['equivalent_sample_size'],
                prior_type=bayes_config_params['prior_type'],
                threshold=config["bayesian_model"]["prediction"]["bayes"]["threshold"]
            )
            if config["output"]["save_models"]:
                save_model(bayes_model, 'bayes', config)
            models['bayes'] = bayes_model
            all_metrics['bayes'] = bayes_metrics

        # Process and save results if any models were trained in this block
        if models: # Only proceed if learn_from_data was False and models were trained
            results = {
                estimator: {k: metrics_data[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
                for estimator, metrics_data in all_metrics.items()
            }
            
            for estimator, metrics_data_item in all_metrics.items():
                metrics_json = {
                    k: v.tolist() if hasattr(v, 'tolist') else v
                    for k, v in metrics_data_item.items()
                }
                if config["output"]["save_metrics"]:
                    with open(f"{config['output']['directories']['results']}/{estimator}_metrics.json", 'w') as f:
                        json.dump(metrics_json, f, indent=4)
                
                print(f"Generating plots for {estimator.upper()} estimator...")
                # Ensure y_test is passed to save_all_plots for correct ROC curve
                # This assumes visualization.py's save_all_plots is updated to accept y_test
                if config["output"]["plots"]["generate_confusion_matrix"] or \
                   config["output"]["plots"]["generate_roc_curve"] or \
                   config["output"]["plots"]["generate_metrics_plot"] or \
                   config["output"]["plots"]["generate_network_structure"]:
                    
                    save_all_plots(
                        metrics_data_item,
                        y_test, # Pass y_test here
                        models[estimator],
                        pca_explained_variance_ratio=data_processor.get_pca_stats()['explained_variance_ratio'] if data_processor.get_pca_stats() else None,
                        output_dir=f"{config['output']['directories']['plots']}/{estimator}",
                        plot_config=config["output"]["plots"]
                    )

            if len(results) > 1:
                print_banner("Results Comparison (Predefined Structure)", "-")
                print("\nPerformance Metrics Comparison:")
                estimators_list = list(results.keys())
                metrics_list_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                header = "{:<10}".format("Metric")
                for est_key in estimators_list:
                    header += " {:<10}".format(est_key.upper())
                if len(estimators_list) > 1: # Should be true if len(results) > 1
                    header += " {:<10}".format("Difference")
                print("\n" + header)
                print("-" * (12 + 12 * len(estimators_list))) # Adjusted width
                for metric_key in metrics_list_to_compare:
                    line = "{:<10}".format(metric_key)
                    for est_key in estimators_list:
                        line += " {:<10.4f}".format(results[est_key][metric_key])
                    if len(estimators_list) == 2:
                        diff_val = results[estimators_list[1]][metric_key] - results[estimators_list[0]][metric_key]
                        line += " {:<10}".format(f"{diff_val:+.4f}")
                    print(line)
                
                if config["output"]["save_metrics"] and len(estimators_list) == 2:
                    comparison_results_json = {
                        estimators_list[0]: results[estimators_list[0]],
                        estimators_list[1]: results[estimators_list[1]],
                        'differences': {
                            metric_key: float(results[estimators_list[1]][metric_key] - results[estimators_list[0]][metric_key])
                            for metric_key in metrics_list_to_compare
                        }
                    }
                    with open(f"{config['output']['directories']['results']}/estimation_methods_comparison.json", 'w') as f:
                        json.dump(comparison_results_json, f, indent=4)

            print("\nKey Observations (Predefined Structure):")
            if 'bayes' in results and 'mle' in results:
                if results['bayes']['accuracy'] > results['mle']['accuracy']:
                    print("- Bayesian estimation shows better overall accuracy")
                if results['bayes']['precision'] > results['mle']['precision']:
                    print("- Bayesian estimation demonstrates better precision")
                if results['mle']['recall'] > results['bayes']['recall']:
                    print("- MLE shows better recall")
                print("- Bayesian estimation generally provides more robust models less prone to overfitting")
                print("- MLE may capture more of the training data patterns but can be less generalizable")
        else: # This else corresponds to if not models:
            print("\nNo estimators enabled for predefined structure path.")
    
    print("\nPipeline completed successfully!")
    print(f"Results have been saved to the '{config['output']['directories']['results']}' directory")
    print(f"Plots have been saved to the '{config['output']['directories']['plots']}' directory")
    
    # The old --compare-structures block is removed as per instructions.
    # The new structure learning flow is integrated above.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Detection with Bayesian Networks')
    parser.add_argument('--config', type=str, default='fraud-detection--main/config.json',
                        help='Path to configuration JSON file')
    # Removed --compare-structures argument
    
    args = parser.parse_args()
    main(config_path=args.config) # Removed compare_structures=args.compare_structures