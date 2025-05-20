import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve
import networkx as nx

def plot_confusion_matrix(confusion_matrix, title='Confusion Matrix'):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Fraud', 'Fraud'],
        yticklabels=['Non-Fraud', 'Fraud']
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_probs, title='ROC Curve'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt.gcf()

def plot_metrics_comparison(metrics_dict, title='Model Performance Metrics'):
    """Plot comparison of different metrics."""
    metrics = {k: v for k, v in metrics_dict.items() 
              if k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values())
    plt.title(title)
    plt.ylabel('Score')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    return plt.gcf()

def plot_network_structure(model, title='Bayesian Network Structure'):
    """Plot the structure of the Bayesian Network."""
    plt.figure(figsize=(12, 10))
    
    # Create networkx graph
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    
    # Count incoming edges to each node to determine node importance
    in_degree = dict(G.in_degree())
    
    # Set node size based on importance (number of incoming edges)
    node_size = {node: 2000 + (in_degree.get(node, 0) * 500) for node in G.nodes()}
    
    # Use a more stable layout for complex networks
    pos = nx.kamada_kawai_layout(G)
    
    # Set node colors: highlight fraud_Cases node
    node_colors = ['red' if node == 'fraud_Cases' else 'lightblue' for node in G.nodes()]
    
    # Draw nodes with varying sizes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[node_size[node] for node in G.nodes()],
        node_color=node_colors,
        alpha=0.8
    )
    
    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos,
        arrowsize=20,
        arrowstyle='-|>',
        edge_color='gray',
        width=1.5
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold'
    )
    
    plt.title(title, fontsize=16)
    plt.axis('off')  # Turn off the axis
    plt.tight_layout()
    return plt.gcf()

def plot_feature_importance(pca_explained_variance_ratio, title='PCA Components Explained Variance'):
    """Plot the explained variance ratio of PCA components."""
    plt.figure(figsize=(10, 6))
    components = [f'PCA_{i+1}' for i in range(len(pca_explained_variance_ratio))]
    
    bars = plt.bar(components, pca_explained_variance_ratio)
    plt.title(title)
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('PCA Components')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()

def save_all_plots(metrics, model, pca_explained_variance_ratio=None, output_dir='plots', plot_config=None):
    """Save all visualization plots to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default config if none provided
    if plot_config is None:
        plot_config = {
            "generate_confusion_matrix": True,
            "generate_roc_curve": True,
            "generate_metrics_plot": True,
            "generate_network_structure": True
        }
    
    # Plot and save confusion matrix
    if plot_config.get("generate_confusion_matrix", True):
        cm_fig = plot_confusion_matrix(metrics['confusion_matrix'])
        cm_fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot and save ROC curve
    if plot_config.get("generate_roc_curve", True):
        roc_fig = plot_roc_curve(metrics['y_pred'], metrics['y_probs'])
        roc_fig.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # Plot and save metrics comparison
    if plot_config.get("generate_metrics_plot", True):
        metrics_fig = plot_metrics_comparison(metrics)
        metrics_fig.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    
    # Plot and save network structure
    if plot_config.get("generate_network_structure", True):
        network_fig = plot_network_structure(model)
        network_fig.savefig(os.path.join(output_dir, 'network_structure.png'))
    
    # Plot and save PCA explained variance if provided
    if pca_explained_variance_ratio is not None and plot_config.get("generate_pca_plot", True):
        pca_fig = plot_feature_importance(pca_explained_variance_ratio)
        pca_fig.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    
    plt.close('all') 