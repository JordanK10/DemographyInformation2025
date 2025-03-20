import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from pmi import (
    pointwise_mutual_information,
    conditional_pointwise_mutual_information,
    local_mutual_information,
    conditional_local_mutual_information,
    get_conditional_probability,
    mutual_information,
    conditional_mutual_information
)

def analyze_n_variable_decomposition(data, target_col, *variables):
    """
    Analyze and decompose mutual information for n variables in all possible orderings.
    
    This function calculates the mutual information decomposition between a target variable
    and n explanatory variables, considering all possible orderings of the variables.
    For each ordering, it computes:
    1. First-order mutual information
    2. Second-order conditional mutual information
    3. Third-order conditional mutual information
    ...
    n. nth-order conditional mutual information
    
    Args:
        data (pd.DataFrame): Input data containing all variables
        target_col (str): Name of the target variable column
        *variables: Variable number of explanatory variable names
        
    Returns:
        pd.DataFrame: Results containing the following columns:
            - Ordering: String representation of variable ordering
            - Level: Information decomposition level (1 to n)
            - Variable: The variable being analyzed
            - Information: Calculated information value
    """
    all_orderings = list(permutations(variables))
    results = []
    
    for ordering in all_orderings:
        # First level: direct mutual information
        mi1 = mutual_information(data, target_col, ordering[0])
        results.append({
            'Ordering': " → ".join(ordering),
            'Level': 1,
            'Variable': ordering[0],
            'Information': mi1
        })
        
        # Higher levels: conditional mutual information
        for level in range(2, len(variables) + 1):
            # Base conditions are all previous variables in the ordering
            base_conditions = list(ordering[:level-1])
            # Current variable to condition on
            current_var = ordering[level-1]
            
            # Calculate conditional mutual information
            cmi = conditional_mutual_information(
                data,
                target_col,
                base_conditions=base_conditions,
                conditioned_conditions=[current_var]
            )
            
            # Store results
            results.append({
                'Ordering': " → ".join(ordering),
                'Level': level,
                'Variable': current_var,
                'Information': cmi
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def analyze_three_variable_decomposition(data, target_col, var1, var2, var3):
    """Wrapper for backward compatibility"""
    return analyze_n_variable_decomposition(data, target_col, var1, var2, var3)

def plot_feature_importance_by_level(data, target_col, feature_cols, discretize=True, num_bins=10, figsize=(12, 6)):
    """
    Plot feature importance by level using mutual information.
    
    Args:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of feature importance by level
    """
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    
    # Calculate information for each feature and level
    results = []
    for feature in feature_cols:
        # First level: direct mutual information using PMI
        pmi = pointwise_mutual_information(data, target_col, feature)
        # Calculate average mutual information
        mi = (pmi * get_conditional_probability(data, target_col, [feature])).sum()
        results.append({
            'Feature': feature,
            'Level': 1,
            'Information': mi
        })
        
        # Calculate conditional information with other features
        for other_feature in feature_cols:
            if other_feature != feature:
                # Calculate conditional PMI
                cpmi = conditional_pointwise_mutual_information(
                    data,
                    target_col,
                    base_conditions=[other_feature],
                    conditioned_conditions=[feature]
                )
                # Calculate average conditional mutual information
                cond_probs = get_conditional_probability(
                    data, target_col, [other_feature, feature]
                )
                cmi = (cpmi * cond_probs).sum()
                results.append({
                    'Feature': feature,
                    'Level': 2,
                    'Information': cmi
                })
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    plot_data = pd.DataFrame(results)
    
    # Create grouped bar plot
    sns.barplot(
        data=plot_data,
        x='Feature',
        y='Information',
        hue='Level',
        ax=ax
    )
    
    # Customize plot
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Feature Importance by Level for {target_col}')
    ax.set_xlabel('Features')
    ax.set_ylabel('Information Content (nats)')
    plt.tight_layout()
    
    return fig

def plot_feature_importance_by_ordering(data, target_col, feature_cols, discretize=True, num_bins=10, figsize=(12, 6)):
    """
    Plot feature importance by ordering using mutual information.
    
    Args:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of feature importance by ordering
    """
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    
    # Calculate information for each feature in different orderings
    results = []
    
    # First add direct mutual information
    for feature in feature_cols:
        # Calculate direct mutual information using PMI
        pmi = pointwise_mutual_information(data, target_col, feature)
        # Calculate average mutual information
        mi = (pmi * get_conditional_probability(data, target_col, [feature])).sum()
        results.append({
            'Feature': feature,
            'Ordering': 'Direct',
            'Information': mi
        })
    
    # Then add conditional mutual information for different orderings
    for ordering in permutations(feature_cols, 2):
        # Calculate conditional PMI
        cpmi = conditional_pointwise_mutual_information(
            data,
            target_col,
            base_conditions=[ordering[0]],
            conditioned_conditions=[ordering[1]]
        )
        # Calculate average conditional mutual information
        cond_probs = get_conditional_probability(
            data, target_col, [ordering[0], ordering[1]]
        )
        cmi = (cpmi * cond_probs).sum()
        
        # Store results
        order_str = f"{ordering[0]} → {ordering[1]}"
        results.append({
            'Feature': ordering[1],
            'Ordering': order_str,
            'Information': cmi
        })
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    plot_data = pd.DataFrame(results)
    
    # Create grouped bar plot
    sns.barplot(
        data=plot_data,
        x='Feature',
        y='Information',
        hue='Ordering',
        ax=ax
    )
    
    # Customize plot
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Feature Importance by Ordering for {target_col}')
    ax.set_xlabel('Features')
    ax.set_ylabel('Information Content (nats)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig

def plot_feature_importance_by_level_and_ordering(data, target_col, feature_cols, discretize=True, num_bins=10, figsize=(12, 6)):
    """
    Plot feature importance by level and ordering using mutual information.
    
    Args:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of feature importance by level and ordering
    """
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    
    # Calculate information for each feature at different levels and orderings
    results = []
    
    # First level: direct mutual information
    for feature in feature_cols:
        lmi = local_mutual_information(data, target_col, feature)
        mi = lmi.mean()  # Use raw value without abs()
        results.append({
            'Feature': feature,
            'Level': 1,
            'Ordering': 'Direct',
            'Information': mi
        })
    
    # Second level: conditional mutual information
    for ordering in permutations(feature_cols, 2):
        clmi = conditional_local_mutual_information(
            data,
            target_col,
            base_conditions=[ordering[0]],
            conditioned_conditions=[ordering[1]]
        )
        cmi = clmi.mean()  # Use raw value without abs()
        
        # Store results
        order_str = f"{ordering[0]} → {ordering[1]}"
        results.append({
            'Feature': ordering[1],
            'Level': 2,
            'Ordering': order_str,
            'Information': cmi
        })
    
    # Create plot with facets
    plot_data = pd.DataFrame(results)
    g = sns.FacetGrid(plot_data, col='Level', col_wrap=2, height=figsize[1], aspect=figsize[0]/figsize[1]/2)
    g.map_dataframe(sns.barplot, x='Feature', y='Information', hue='Ordering')
    
    # Customize plot
    g.fig.suptitle(f'Feature Importance by Level and Ordering for {target_col}')
    for ax in g.axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
        ax.set_ylabel('Information Content (nats)')
    
    plt.tight_layout()
    return g.fig 