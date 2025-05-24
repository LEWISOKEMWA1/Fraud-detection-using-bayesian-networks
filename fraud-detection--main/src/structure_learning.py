import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import itertools # Add this import
from pgmpy.estimators import HillClimbSearch, BDeuScore, BICScore # Adjusted imports for clarity, BICScore not used in new func but good to keep if BIC is an option elsewhere
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from sklearn.model_selection import train_test_split

from data_processing import generate_synthetic_data # This might be removed if data generation is always dynamic
from dynamic_data_processor import DynamicDataProcessor
from bayesian_model import train_model, evaluate_model # evaluate_model might be simplified or removed in this file later
from visualization import plot_network_structure

def learn_bayesian_structure_hill_climbing(data_for_structure_learning: pd.DataFrame, target_column: str, config: dict):
    '''
    Learns a Bayesian Network structure using an explicit Hill Climbing algorithm.

    Args:
        data_for_structure_learning (pd.DataFrame): DataFrame containing only the columns to be used as nodes
                                                    (selected PCs + target_column).
        target_column (str): The name of the target variable column. (Not directly used in current HC logic but good for context)
        config (dict): Configuration dictionary, expected to have
                       config['structure_learning']['bdeu_equivalent_sample_size'] (e.g., 10)
                       config['structure_learning']['max_iter_hill_climb'] (e.g., 100)

    Returns:
        DiscreteBayesianNetwork: The learned Bayesian network structure.
    '''
    node_names = list(data_for_structure_learning.columns)
    current_model = DiscreteBayesianNetwork()
    current_model.add_nodes_from(node_names) # Start with an empty graph (nodes, no edges)

    # Initialize BDeuScore
    bdeu_equivalent_sample_size = config.get('structure_learning', {}).get('bdeu_equivalent_sample_size', 10)
    scorer = BDeuScore(data=data_for_structure_learning, equivalent_sample_size=bdeu_equivalent_sample_size)
    current_score = scorer.score(current_model)

    print(f"Initial (empty graph) BDeu score: {current_score:.4f}")

    iteration = 0
    # Add a max_iter config, e.g. config['structure_learning']['max_iter_hill_climb']
    max_iterations = config.get('structure_learning', {}).get('max_iter_hill_climb', 100) 

    while iteration < max_iterations:
        iteration += 1
        best_model_in_iteration = None
        best_score_in_iteration = -float('inf')
        best_operation_description = ""

        current_edges = set(current_model.edges())

        # Try all possible single edge additions, deletions, or reversals
        for u, v in itertools.permutations(node_names, 2):
            # 1. Try adding edge u -> v
            if (u, v) not in current_edges:
                temp_model = DiscreteBayesianNetwork(list(current_edges) + [(u,v)])
                temp_model.add_nodes_from(node_names) # Ensure all nodes are present
                if temp_model.check_model(): # pgmpy's check_model verifies acyclicity and node presence
                    score = scorer.score(temp_model)
                    if score > best_score_in_iteration:
                        best_score_in_iteration = score
                        best_model_in_iteration = temp_model
                        best_operation_description = f"Add edge {u} -> {v}"

            # 2. Try deleting edge u -> v (if it exists)
            if (u, v) in current_edges:
                temp_edges = list(current_edges - {(u,v)})
                temp_model = DiscreteBayesianNetwork(temp_edges)
                temp_model.add_nodes_from(node_names)
                if temp_model.check_model(): # Should always be true if original was valid
                    score = scorer.score(temp_model)
                    if score > best_score_in_iteration:
                        best_score_in_iteration = score
                        best_model_in_iteration = temp_model
                        best_operation_description = f"Delete edge {u} -> {v}"
            
            # 3. Try reversing edge u -> v (if u->v exists, try v->u)
            if (u, v) in current_edges:
                # Ensure v->u doesn't already exist to avoid trivial reversal if both directions somehow exist
                if (v,u) not in current_edges:
                    temp_edges = list(current_edges - {(u,v)}) + [(v,u)]
                    temp_model = DiscreteBayesianNetwork(temp_edges)
                    temp_model.add_nodes_from(node_names)
                    if temp_model.check_model():
                        score = scorer.score(temp_model)
                        if score > best_score_in_iteration:
                            best_score_in_iteration = score
                            best_model_in_iteration = temp_model
                            best_operation_description = f"Reverse edge {u} -> {v} to {v} -> {u}"
        
        if best_model_in_iteration is not None and best_score_in_iteration > current_score:
            current_model = best_model_in_iteration
            current_score = best_score_in_iteration
            print(f"Iteration {iteration}: Best op: '{best_operation_description}', New BDeu score: {current_score:.4f}")
        else:
            print(f"Iteration {iteration}: No improvement found. Terminating Hill Climbing.")
            break
    
    if iteration == max_iterations:
        print(f"Reached max_iterations ({max_iterations}) for Hill Climbing.")

    print(f"Learned structure with {len(current_model.edges())} edges. Final BDeu score: {current_score:.4f}")
    return current_model

# The old compare_structures function might need significant changes or removal
# For now, let's comment it out or adapt it later if comparison is reintroduced.
# def compare_structures(manual_model, learned_model, X_test, y_test, config_path='config.json'):
#     """Compare performance of two different Bayesian Network structures."""
# The function compare_parameter_learners_on_dag will be removed.

def run_structure_learning_process(config_path='config.json'):
    """Runs the process of learning a BN structure, training with Bayesian params, and evaluating."""
    print("Starting Bayesian Network structure learning and Bayesian parameter estimation process...")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    target_column = config['data_processing']['target_column']

    for directory in config.get('output', {}).get('directories', {}).values():
        os.makedirs(directory, exist_ok=True)
    
    print("Preparing data...")
    if not config.get("data_source", {}).get("use_real_data", False):
        print("Generating synthetic data...")
        df = generate_synthetic_data(
            num_samples=config['data_processing']['num_samples'],
            fraud_ratio=config['data_processing']['fraud_ratio']
        )
    else:
        csv_path = config.get("data_source", {}).get("csv_file_path", None)
        if csv_path and os.path.exists(csv_path):
            print(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path, sep=config.get("data_source", {}).get("csv_separator", ","))
        else:
            print("Configured to use real data, but CSV not found. Generating synthetic data.")
            df = generate_synthetic_data(
                num_samples=config['data_processing']['num_samples'],
                fraud_ratio=config['data_processing']['fraud_ratio']
            )

    data_processor = DynamicDataProcessor(config_path)
    X_train, X_test, y_train, y_test = data_processor.process_pipeline(df)
    
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index, name=target_column)
    else:
        y_train = y_train.rename(target_column)

    data_for_structure_learning = pd.concat([X_train, y_train], axis=1)
    
    print(f"\nData for structure learning columns: {data_for_structure_learning.columns.tolist()}")
    print(f"Number of samples for structure learning: {len(data_for_structure_learning)}")

    print("\nLearning Bayesian Network structure using custom Hill Climbing algorithm...")
    learned_model_structure = learn_bayesian_structure_hill_climbing(
        data_for_structure_learning,
        target_column,
        config
    )
    print(f"Learned structure has {len(learned_model_structure.edges())} edges: {list(learned_model_structure.edges())}")

    # Plot the single learned DAG structure
    plots_dir = config.get('output', {}).get('directories', {}).get('plots', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    learned_dag_plot_path = os.path.join(plots_dir, 'learned_dag_structure.png')
    try:
        # Temporarily fit with MLE to allow plotting if model has no CPDs yet.
        # This is only for visualization of the structure.
        # The actual MLE and Bayes models for evaluation will be trained separately.
        plot_model = learned_model_structure.copy()
        if not plot_model.get_cpds(): # Check if model has CPDs
             # Fit with MLE if no CPDs, so plot_network_structure can infer states if needed
            plot_model.fit(data_for_structure_learning, estimator=MaximumLikelihoodEstimator)

        fig_dag = plot_network_structure(plot_model, title='Learned DAG Structure')
        fig_dag.savefig(learned_dag_plot_path)
        print(f"Learned DAG structure plot saved to {learned_dag_plot_path}")
    except Exception as e:
        print(f"Error plotting learned DAG structure: {e}")
    finally:
        if 'fig_dag' in locals():
             plt.close(fig_dag)


    print("\nStarting Bayesian parameter learning for the learned DAG...")

    # Bayesian Parameter Learning & Evaluation
    bayes_model_for_training = learned_model_structure.copy() 
    bayesian_estimator_config = config.get('bayesian_model', {}).get('estimators', {}).get('bayes', {})
    
    model_bayes_params = train_model(
        bayes_model_for_training,
        data_for_structure_learning, 
        estimator_type='bayes',
        **bayesian_estimator_config 
    )
    print("Evaluating Bayesian parameterized model...")
    bayes_metrics = evaluate_model(
        model_bayes_params,
        X_test,
        y_test,
        config_path=config_path 
    )

    # Print summary of Bayesian model's metrics
    print("\nLearned DAG with Bayesian Parameters - Performance:")
    print("-" * 50)
    for metric, value in bayes_metrics.items():
        if metric not in ['confusion_matrix', 'y_pred', 'y_probs', 'threshold']: # Exclude non-scalar or bulky metrics
            print(f"{metric:<12}: {value:.4f}")
        elif metric == 'threshold':
            print(f"{metric:<12}: {value}")
    print("-" * 50)

    # Save Bayesian metrics to JSON
    results_dir = config.get('output', {}).get('directories', {}).get('results', 'results')
    os.makedirs(results_dir, exist_ok=True)
    bayes_metrics_filepath = os.path.join(results_dir, 'learned_dag_bayes_metrics.json')
    
    serializable_bayes_metrics = {}
    for k, v in bayes_metrics.items():
        if isinstance(v, (np.ndarray, pd.Series)):
            serializable_bayes_metrics[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
            serializable_bayes_metrics[k] = float(v) if isinstance(v, (np.float32, np.float64)) else int(v)
        elif k not in ['confusion_matrix', 'y_pred', 'y_probs']: # Exclude bulky or complex types
            serializable_bayes_metrics[k] = v
        elif k == 'confusion_matrix' and isinstance(v, np.ndarray): # Handle confusion matrix specifically
             serializable_bayes_metrics[k] = v.tolist()


    with open(bayes_metrics_filepath, 'w') as f:
        json.dump(serializable_bayes_metrics, f, indent=4)
    print(f"Bayesian parameter learning metrics saved to {bayes_metrics_filepath}")
    
    # Save the Bayesian parameterized model
    if config.get('output', {}).get('save_models', False):
        models_dir = config.get('output', {}).get('directories', {}).get('models', 'models')
        os.makedirs(models_dir, exist_ok=True)
        try:
            import joblib
            joblib.dump(model_bayes_params, os.path.join(models_dir, 'learned_dag_bayes_model.pkl'))
            print(f"Bayesian parameterized model saved to {models_dir}/learned_dag_bayes_model.pkl")
        except ImportError:
            print("joblib not installed. Bayesian model not saved. Install joblib to save models.")
        except Exception as e:
            print(f"Error saving Bayesian model: {e}")

    print(f"\nStructure learning and Bayesian parameter estimation process completed!")
    
    return {
        'learned_dag_structure_edges': list(learned_model_structure.edges()),
        'bayes_metrics': bayes_metrics,
        'trained_bayes_model': model_bayes_params # Optional: return trained model
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bayesian Network Structure Learning and Bayesian Parameter Estimation Process')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration JSON file')
    
    args = parser.parse_args()
    run_structure_learning_process(config_path=args.config)