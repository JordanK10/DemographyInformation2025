#===============================
# SECTION 1: SETUP AND DEPENDENCIES
#===============================
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from itertools import permutations, combinations

#===============================
# SECTION 2: CORE FUNCTIONS - INFORMATION THEORY CALCULATIONS
#===============================
# Vectorized implementation of mutual information
def calculate_mutual_info_vectorized(x, y):
    # Create joint probability table
    joint_counts = pd.crosstab(x, y)
    joint_prob = joint_counts / len(x)
    
    # Calculate marginal probabilities
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    
    # Calculate mutual information using vectorized operations
    # Create outer product for denominator
    denominator = np.outer(p_x, p_y)
    # Avoid division by zero and log of zero
    mask = joint_prob.values > 0
    mi = np.sum(
        np.where(
            mask,
            joint_prob.values * np.log2(joint_prob.values / denominator),
            0
        )
    )
    
    return mi

# Function to calculate conditional mutual information: I(X;Y|Z)
def calculate_cond_mutual_info_vectorized(x, y, z):
    """
    Vectorized implementation of conditional mutual information I(X;Y|Z)
    """
    # Initialize result
    cond_mi = 0
    
    # Pre-compute unique z values and their probabilities
    z_values, z_counts = np.unique(z, return_counts=True)
    p_z = z_counts / len(z)
    
    # Vectorized computation for each z value
    for z_val, prob_z in zip(z_values, p_z):
        # Create mask once
        mask = (z == z_val)
        n_samples = sum(mask)
        
        # Skip if insufficient samples
        if n_samples <= 1:
            continue
        
        # Get conditional variables
        x_z = x[mask]
        y_z = y[mask]
        
        # Check variation using numpy operations
        if len(np.unique(x_z)) <= 1 or len(np.unique(y_z)) <= 1:
            continue
        
        # Calculate conditional MI using vectorized mutual info
        mi_z = calculate_mutual_info_vectorized(x_z, y_z)
        
        # Accumulate weighted result
        cond_mi += prob_z * mi_z
    
    return cond_mi

def analyze_feature_importance(data, target_col, feature_cols, discretize=True, num_bins=10):
    """
    Analyze feature importance using mutual information.
    """
    results = []
    for feature in feature_cols:
        mi = calculate_mutual_info_vectorized(data[target_col], data[feature])
        results.append({
            'Feature': feature,
            'Mutual_Information': mi
        })
    return pd.DataFrame(results).sort_values('Mutual_Information', ascending=False)

def analyze_feature_combinations(data, target_col, feature_cols, max_features=4):
    """
    Analyze feature combinations using mutual information.
    """
    results = []
    for n in range(1, min(max_features + 1, len(feature_cols) + 1)):
        for feature_set in combinations(feature_cols, n):
            total_mi = 0
            for feature in feature_set:
                mi = calculate_mutual_info_vectorized(data[target_col], data[feature])
                total_mi += mi
            results.append({
                'Features': ' + '.join(feature_set),
                'Num_Features': len(feature_set),
                'Total_Information': total_mi
            })
    return pd.DataFrame(results).sort_values('Total_Information', ascending=False)

def plot_feature_importance(data, target_col, feature_cols):
    """
    Plot feature importance using mutual information.
    """
    importance_df = analyze_feature_importance(data, target_col, feature_cols)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Feature', y='Mutual_Information')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Feature Importance for {target_col}')
    plt.tight_layout()
    return plt.gcf()

def plot_feature_combinations(data, target_col, feature_cols, max_features=4):
    """
    Plot feature combinations using mutual information.
    """
    combinations_df = analyze_feature_combinations(data, target_col, feature_cols, max_features)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=combinations_df, x='Features', y='Total_Information')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Feature Combinations Information for {target_col}')
    plt.tight_layout()
    return plt.gcf()

def analyze_three_variable_decomposition(data, target_col, var1, var2, var3):
    """
    Analyze and plot mutual information decomposition for three variables in all possible orderings.
    
    Parameters:
        data: pandas DataFrame
        target_col: target variable (e.g., 'income')
        var1, var2, var3: three variables to analyze (e.g., 'profession_4', 'gender', 'education')
    """
    variables = [var1, var2, var3]
    all_orderings = list(permutations(variables))
    results = []
    
    for ordering in all_orderings:
        # First level
        mi1 = calculate_mutual_info_vectorized(data[target_col], data[ordering[0]])
        
        # Second level (conditional on first)
        mi2 = calculate_cond_mutual_info_vectorized(data[target_col], data[ordering[1]], data[ordering[0]])
        
        # Third level (conditional on first and second)
        combined_cond = data[list(ordering[:2])].apply(lambda x: f"{x[0]}|{x[1]}", axis=1)
        mi3 = calculate_cond_mutual_info_vectorized(data[target_col], data[ordering[2]], combined_cond)
        
        # Store results
        order_str = " → ".join(ordering)
        results.extend([
            {'Ordering': order_str, 'Level': 1, 'Variable': ordering[0], 'Information': mi1},
            {'Ordering': order_str, 'Level': 2, 'Variable': ordering[1], 'Information': mi2},
            {'Ordering': order_str, 'Level': 3, 'Variable': ordering[2], 'Information': mi3}
        ])
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    #### 
        # Create plot
        # plt.figure(figsize=(12, 5))
        
    # # Create a color map for variables using Set3 palette
    # var_colors = {
    #     var1: '#8DD3C7',  # Soft teal
    #     var2: '#FFFFB3',  # Light yellow
    #     var3: '#BEBADA'   # Soft purple
    # }
    
    # # For each ordering, plot bars in the correct stacking order
    # orderings = results_df['Ordering'].unique()
    # x = np.arange(len(orderings))
    
    # # Plot each ordering
    # for i, ordering in enumerate(orderings):
    #     ordering_data = results_df[results_df['Ordering'] == ordering]
    #     bottom = 0
        
    #     # Get the variables in their order of appearance
    #     variables_in_order = ordering.split(" → ")
        
    #     # Plot each variable's contribution
    #     for var in variables_in_order:
    #         height = ordering_data[ordering_data['Variable'] == var]['Information'].values[0]
    #         plt.bar(
    #             i,
    #             height,
    #             bottom=bottom,
    #             color=var_colors[var],
    #             label=str(var)
    #         )
    #         bottom += height
    
    # plt.xticks(x, ["" for _ in orderings], rotation=45, ha='right', fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.title(f'Information Decomposition for {var1}, {var2}, {var3} on {target_col}', fontsize=18)
    # plt.xlabel('Ordering', fontsize=18)
    # plt.ylabel('Information (bits)', fontsize=18)
    
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(),fontsize=14)

    # plt.tight_layout()
    
    # return {
    #     'plot': plt.gcf(),
    #     'results': results_df
    # 
    return results_df


#===============================
# SECTION 5: CORE FUNCTIONS - INFORMATION THEORY CALCULATIONS
#===============================
# R-equivalent implementation of mutual information

# Vectorized implementation (for testing/comparison)
def calculate_mutual_info_vectorized(x, y):
    # Create joint probability table
    joint_counts = pd.crosstab(x, y)
    joint_prob = joint_counts / len(x)
    
    # Calculate marginal probabilities
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    
    # Calculate mutual information using vectorized operations
    # Create outer product for denominator
    denominator = np.outer(p_x, p_y)
    # Avoid division by zero and log of zero
    mask = joint_prob.values > 0
    mi = np.sum(
        np.where(
            mask,
            joint_prob.values * np.log2(joint_prob.values / denominator),
            0
        )
    )
    
    return mi

# Function to calculate conditional mutual information: I(X;Y|Z)
def calculate_cond_mutual_info_vectorized(x, y, z):
    """
    Vectorized implementation of conditional mutual information I(X;Y|Z)
    """
    # Initialize result
    cond_mi = 0
    
    # Pre-compute unique z values and their probabilities
    z_values, z_counts = np.unique(z, return_counts=True)
    p_z = z_counts / len(z)
    
    # Vectorized computation for each z value
    for z_val, prob_z in zip(z_values, p_z):
        # Create mask once
        mask = (z == z_val)
        n_samples = sum(mask)
        
        # Skip if insufficient samples
        if n_samples <= 1:
            continue
            
        # Get conditional variables
        x_z = x[mask]
        y_z = y[mask]
        
        # Check variation using numpy operations
        if len(np.unique(x_z)) <= 1 or len(np.unique(y_z)) <= 1:
            continue
        
        # Calculate conditional MI using vectorized mutual info
        mi_z = calculate_mutual_info_vectorized(x_z, y_z)
        
        # Accumulate weighted result
        cond_mi += prob_z * mi_z
    
    return cond_mi

def discretize_variable(x, num_bins=100):
    if isinstance(x, (pd.Series, np.ndarray)) and np.issubdtype(x.dtype, np.number):
        x_binned = pd.qcut(x, 
                          q=num_bins,
                          labels=[f"B{i+1}" for i in range(num_bins)],
                          duplicates='drop')
        return x_binned
    else:
        return x

def discretize_variable_vectorized(x, num_bins=100):
    """
    Vectorized implementation of variable discretization
    """
    if isinstance(x, (pd.Series, np.ndarray)) and np.issubdtype(x.dtype, np.number):
        # Handle edge cases
        if len(np.unique(x)) <= num_bins:
            return pd.Series(x).astype('category')
            
        # Use numpy's quantile for faster computation
        edges = np.nanquantile(x, np.linspace(0, 1, num_bins + 1))
        # Remove duplicate edges
        edges = np.unique(edges)
        # Create labels
        labels = [f"B{i+1}" for i in range(len(edges)-1)]
        
        return pd.cut(x, bins=edges, labels=labels, include_lowest=True)
    return x

def multilevel_info_decomposition(data, target_col, feature_cols, discretize=True, num_bins=10):
    """Direct translation of R's multilevel_info_decomposition function"""
    # Make a copy of the data
    df = data.copy()
    
    # Discretize target variable if needed
    if discretize and np.issubdtype(df[target_col].dtype, np.number):
        df['target_discretized'] = discretize_variable(df[target_col], num_bins)
        target_var = 'target_discretized'
    else:
        target_var = target_col
    
    # Discretize feature variables if needed
    if discretize:
        for col in feature_cols:
            if np.issubdtype(df[col].dtype, np.number):
                df[f"{col}_discretized"] = discretize_variable(df[col], num_bins)
                feature_cols = [f"{col}_discretized" if c == col else c for c in feature_cols]
    
    # Number of features
    n_features = len(feature_cols)
    
    # Create all possible permutations of feature ordering
    feature_orders = []
    if n_features <= 6:  # Limit full permutation to 6 features
        feature_orders = list(permutations(feature_cols))
    else:
        # Just use the original order and a few random ones
        feature_orders = [tuple(feature_cols)]
        np.random.seed(123)
        for _ in range(4):
            feature_orders.append(tuple(np.random.permutation(feature_cols)))
    
    # Store results
    results = []
    
    # For each ordering
    for order in feature_orders:
        order_str = " → ".join(order)
        
        # Calculate first level mutual information
        mi = calculate_mutual_info_vectorized(df[target_var], df[order[0]])
        results.append({
            'Ordering': order_str,
            'Level': 1,
            'Feature': order[0],
            'Information': mi,
            'Proportion': mi
        })
        
        # Calculate conditional mutual information for subsequent levels
        for i in range(1, len(order)):
            cond_mi = calculate_cond_mutual_info_vectorized(
                df[target_var],
                df[order[i]],
                df[list(order[:i])]
            )
            results.append({
                'Ordering': order_str,
                'Level': i + 1,
                'Feature': order[i],
                'Information': cond_mi,
                'Proportion': cond_mi
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create plotting function
    def plot_results(top_n=None):
        plot_data = results_df.copy()
        if top_n:
            ordering_totals = plot_data.groupby('Ordering')['Information'].sum()
            top_orderings = ordering_totals.nlargest(top_n).index
            plot_data = plot_data[plot_data['Ordering'].isin(top_orderings)]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=plot_data, x='Ordering', y='Information', hue='Level')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Multilevel Information Decomposition for {target_col}')
        plt.tight_layout()
        return plt.gcf()
    
    return {
        'summary': results_df,
        'plot': plot_results
    }

def multilevel_info_decomposition_vectorized(data, target_col, feature_cols, discretize=True, num_bins=10):
    """
    Vectorized implementation of multilevel information decomposition
    """
    # Make a copy and convert to numpy arrays for faster computation
    df = data.copy()
    
    # Pre-process target variable
    if discretize and np.issubdtype(df[target_col].dtype, np.number):
        target_data = discretize_variable_vectorized(df[target_col], num_bins)
    else:
        target_data = df[target_col]
    
    # Pre-process features
    processed_features = {}
    feature_data = {}
    for col in feature_cols:
        if discretize and np.issubdtype(df[col].dtype, np.number):
            processed_features[col] = f"{col}_discretized"
            feature_data[col] = discretize_variable_vectorized(df[col], num_bins)
        else:
            processed_features[col] = col
            feature_data[col] = df[col]
    
    # Compute feature orderings
    n_features = len(feature_cols)
    if n_features <= 6:
        feature_orders = list(permutations(feature_cols))
    else:
        np.random.seed(123)
        base_order = tuple(feature_cols)
        random_orders = [tuple(np.random.permutation(feature_cols)) for _ in range(4)]
        feature_orders = [base_order] + random_orders
    
    # Pre-allocate results list with estimated size
    results = []
    
    # Process each ordering in parallel using pandas operations
    for order in feature_orders:
        order_str = " → ".join(order)
        
        # First level MI (vectorized)
        mi = calculate_mutual_info_vectorized(target_data, feature_data[order[0]])
        results.append({
            'Ordering': order_str,
            'Level': 1,
            'Feature': processed_features[order[0]],
            'Information': mi,
            'Proportion': mi
        })
        
        # Higher level conditional MI (vectorized)
        for i in range(1, len(order)):
            # Create conditioning dataset
            cond_features = pd.concat([feature_data[f] for f in order[:i]], axis=1)
            
            cond_mi = calculate_cond_mutual_info_vectorized(
                target_data,
                feature_data[order[i]],
                cond_features
            )
            
            results.append({
                'Ordering': order_str,
                'Level': i + 1,
                'Feature': processed_features[order[i]],
                'Information': cond_mi,
                'Proportion': cond_mi
            })
    
    # Create results DataFrame efficiently
    results_df = pd.DataFrame(results)
    
    # Optimized plotting function
    def plot_results(top_n=None):
        if top_n:
            # Use numpy operations for faster aggregation
            ordering_totals = results_df.groupby('Ordering')['Information'].sum()
            top_orderings = ordering_totals.nlargest(top_n).index
            plot_data = results_df[results_df['Ordering'].isin(top_orderings)]
        else:
            plot_data = results_df
        
        # Create plot with optimized settings
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=plot_data, x='Ordering', y='Information', hue='Level', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Multilevel Information Decomposition for {target_col}')
        plt.tight_layout()
        return fig
    
    return {
        'summary': results_df,
        'plot': plot_results
    }


def analyze_feature_combinations(data, target_col, feature_cols, max_features=5, discretize=True, num_bins=10):
    """
    Analyze feature combinations using mutual information decomposition.
    Optimized implementation using vectorized operations.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        max_features (int): Maximum number of features to combine
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
    
    Returns:
        pd.DataFrame: Results sorted by total information
    """
    # Make a copy and pre-process data
    df = data.copy()
    
    # Pre-process target variable once
    if discretize and np.issubdtype(df[target_col].dtype, np.number):
        target_data = discretize_variable_vectorized(df[target_col], num_bins)
    else:
        target_data = df[target_col]
    
    # Pre-process all features once
    processed_features = {}
    feature_data = {}
    for col in feature_cols:
        if discretize and np.issubdtype(df[col].dtype, np.number):
            processed_features[col] = f"{col}_discretized"
            feature_data[col] = discretize_variable_vectorized(df[col], num_bins)
        else:
            processed_features[col] = col
            feature_data[col] = df[col]
    
    # Pre-compute all possible combinations up to max_features
    all_combinations = []
    for n in range(1, min(max_features + 1, len(feature_cols) + 1)):
        all_combinations.extend(combinations(feature_cols, n))
    
    # Process each combination using vectorized operations
    results = []
    for feature_set in all_combinations:
        subset_data = {col: feature_data[col] for col in feature_set}
        info_decomp = multilevel_info_decomposition_vectorized(
            pd.DataFrame(subset_data),
            target_data.name if hasattr(target_data, 'name') else 'target',
            list(feature_set),
            discretize=False
        )
        total_info = info_decomp['summary']['Information'].sum()
        results.append({
            'Features': ' + '.join(feature_set),
            'Num_Features': len(feature_set),
            'Total_Information': total_info
        })
    
    return pd.DataFrame(results).sort_values('Total_Information', ascending=False)

def analyze_feature_importance(data, target_col, feature_cols, discretize=True, num_bins=10):
    """
    Analyze feature importance using mutual information.
    Optimized implementation using vectorized operations.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
    
    Returns:
        pd.DataFrame: Results sorted by mutual information
    """
    # Make a copy and pre-process data once
    df = data.copy()
    
    # Pre-process target variable once
    if discretize and np.issubdtype(df[target_col].dtype, np.number):
        target_data = discretize_variable_vectorized(df[target_col], num_bins)
    else:
        target_data = df[target_col]
    
    # Pre-process all features at once
    feature_data = {}
    processed_names = {}
    for col in feature_cols:
        if discretize and np.issubdtype(df[col].dtype, np.number):
            feature_data[col] = discretize_variable_vectorized(df[col], num_bins)
            processed_names[col] = f"{col}_discretized"
        else:
            feature_data[col] = df[col]
            processed_names[col] = col
    
    # Calculate MI for all features using vectorized operations
    results = [{
        'Feature': processed_names[feature],
        'Mutual_Information': calculate_mutual_info_vectorized(target_data, feature_data[feature])
    } for feature in feature_cols]
    
    # Create DataFrame and sort efficiently
    return pd.DataFrame(results).sort_values('Mutual_Information', ascending=False)

def plot_feature_importance(data, target_col, feature_cols, discretize=True, num_bins=10, figsize=(10, 6)):
    """
    Plot feature importance using mutual information.
    Optimized implementation using vectorized operations.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of feature importance
    """
    # Calculate feature importance using vectorized implementation
    importance_df = analyze_feature_importance(data, target_col, feature_cols, 
                                            discretize=discretize, num_bins=num_bins)
    
    # Create plot with optimized settings
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use optimized barplot with pre-sorted data
    sns.barplot(
        data=importance_df,
        x='Feature',
        y='Mutual_Information',
        order=importance_df.sort_values('Mutual_Information', ascending=True).Feature,
        ax=ax
    )
    
    # Optimize text rendering
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Feature Importance for {target_col}')
    ax.set_xlabel('Features')
    ax.set_ylabel('Mutual Information')
    
    # Optimize layout
    plt.tight_layout()
    
    return fig

def plot_feature_combinations(data, target_col, feature_cols, max_features=5, discretize=True, num_bins=10, figsize=(12, 6)):
    """
    Plot feature combinations using mutual information.
    Optimized implementation using vectorized operations.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        max_features (int): Maximum number of features to combine
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of feature combinations
    """
    # Calculate feature combinations using vectorized implementation
    combinations_df = analyze_feature_combinations(data, target_col, feature_cols, 
                                                max_features=max_features,
                                                discretize=discretize, 
                                                num_bins=num_bins)
    
    # Create plot with optimized settings
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use optimized barplot with pre-sorted data
    sns.barplot(
        data=combinations_df,
        x='Features',
        y='Total_Information',
        order=combinations_df.sort_values('Total_Information', ascending=True).Features,
        ax=ax
    )
    
    # Optimize text rendering
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Feature Combinations Information for {target_col}')
    ax.set_xlabel('Feature Combinations')
    ax.set_ylabel('Total Mutual Information')
    
    # Optimize layout
    plt.tight_layout()
    
    return fig

def plot_multilevel_info(data, target_col, feature_cols, discretize=True, num_bins=10, top_n=None, figsize=(12, 6)):
    """
    Plot multilevel information decomposition.
    Optimized implementation using vectorized operations.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        top_n (int): Number of top orderings to show (optional)
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of multilevel information decomposition
    """
    # Calculate multilevel information decomposition using vectorized implementation
    decomp = multilevel_info_decomposition_vectorized(data, target_col, feature_cols, 
                                                    discretize=discretize, 
                                                    num_bins=num_bins)
    
    # Create plot with optimized settings
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter for top N orderings if specified using efficient operations
    plot_data = decomp['summary'].copy()
    if top_n:
        # Use numpy operations for faster aggregation
        ordering_totals = plot_data.groupby('Ordering')['Information'].sum()
        top_orderings = ordering_totals.nlargest(top_n).index
        plot_data = plot_data[plot_data['Ordering'].isin(top_orderings)]
    
    # Create optimized stacked bar plot
    sns.barplot(
        data=plot_data,
        x='Ordering',
        y='Information',
        hue='Level',
        ax=ax
    )
    
    # Optimize text rendering
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Multilevel Information Decomposition for {target_col}')
    ax.set_xlabel('Feature Orderings')
    ax.set_ylabel('Information Content')
    
    # Optimize layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance_by_level(data, target_col, feature_cols, discretize=True, num_bins=10, figsize=(12, 6)):
    """
    Plot feature importance by level using mutual information.
    Optimized implementation using vectorized operations.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of feature importance by level
    """
    # Calculate multilevel information decomposition using vectorized implementation
    decomp = multilevel_info_decomposition_vectorized(data, target_col, feature_cols, 
                                                    discretize=discretize, 
                                                    num_bins=num_bins)
    
    # Create plot with optimized settings
    fig, ax = plt.subplots(figsize=figsize)
    
    # Aggregate information efficiently using pandas operations
    plot_data = (decomp['summary']
                .groupby(['Feature', 'Level'])['Information']
                .mean()
                .reset_index())
    
    # Create optimized grouped bar plot
    sns.barplot(
        data=plot_data,
        x='Feature',
        y='Information',
        hue='Level',
        ax=ax
    )
    
    # Optimize text rendering
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Feature Importance by Level for {target_col}')
    ax.set_xlabel('Features')
    ax.set_ylabel('Average Information Content')
    
    # Optimize layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance_by_ordering(data, target_col, feature_cols, discretize=True, num_bins=10, figsize=(12, 6)):
    """
    Plot feature importance by ordering using mutual information.
    Optimized implementation using vectorized operations.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of feature importance by ordering
    """
    # Calculate multilevel information decomposition using vectorized implementation
    decomp = multilevel_info_decomposition_vectorized(data, target_col, feature_cols, 
                                                    discretize=discretize, 
                                                    num_bins=num_bins)
    
    # Create plot with optimized settings
    fig, ax = plt.subplots(figsize=figsize)
    
    # Aggregate information efficiently using pandas operations
    plot_data = (decomp['summary']
                .groupby(['Feature', 'Ordering'])['Information']
                .mean()
                .reset_index())
    
    # Create optimized grouped bar plot
    sns.barplot(
        data=plot_data,
        x='Feature',
        y='Information',
        hue='Ordering',
        ax=ax
    )
    
    # Optimize text rendering
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Feature Importance by Ordering for {target_col}')
    ax.set_xlabel('Features')
    ax.set_ylabel('Average Information Content')
    
    # Optimize layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance_by_level_and_ordering(data, target_col, feature_cols, discretize=True, num_bins=10, figsize=(12, 6)):
    """
    Plot feature importance by level and ordering using mutual information.
    Optimized implementation using vectorized operations.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Target variable column name
        feature_cols (list): List of feature column names
        discretize (bool): Whether to discretize numeric variables
        num_bins (int): Number of bins for discretization
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        matplotlib.figure.Figure: Plot of feature importance by level and ordering
    """
    # Calculate multilevel information decomposition using vectorized implementation
    decomp = multilevel_info_decomposition_vectorized(data, target_col, feature_cols, 
                                                    discretize=discretize, 
                                                    num_bins=num_bins)
    
    # Create optimized faceted plot
    g = sns.FacetGrid(decomp['summary'], 
                      col='Level', 
                      col_wrap=2,
                      height=figsize[1],
                      aspect=figsize[0]/figsize[1]/2)
    
    # Map optimized barplot to facets
    g.map_dataframe(sns.barplot, 
                    x='Feature', 
                    y='Information', 
                    hue='Ordering')
    
    # Set title efficiently
    g.fig.suptitle(f'Feature Importance by Level and Ordering for {target_col}')
    
    # Optimize text rendering for all subplots
    for ax in g.axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
    
    # Optimize layout
    plt.tight_layout()
    
    return g.fig


# Test multilevel information decomposition implementations
if __name__ == "__main__":
    print("\nTesting multilevel information decomposition implementations...")
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 1000
    
    dummy_data = pd.DataFrame({
        'target': np.random.normal(0, 1, n_samples),
        'feature1': ['A', 'B', 'C'] * (n_samples // 3),
        'feature2': np.random.choice(['X', 'Y'], n_samples),
        'feature3': np.random.uniform(0, 10, n_samples)
    })
    
    feature_cols = ['feature1', 'feature2', 'feature3']
    
    print("\nTest case 1: Basic functionality")
    # Test both implementations with simple case
    result_orig = multilevel_info_decomposition(dummy_data, 'target', feature_cols[:2])
    result_vec = multilevel_info_decomposition_vectorized(dummy_data, 'target', feature_cols[:2])
    
    print("Original implementation shape:", result_orig['summary'].shape)
    print("Vectorized implementation shape:", result_vec['summary'].shape)
    print("Results match:", np.allclose(
        result_orig['summary']['Information'],
        result_vec['summary']['Information'],
        rtol=1e-10
    ))
    
    print("\nTest case 2: Edge case - single feature")
    result_orig_single = multilevel_info_decomposition(dummy_data, 'target', feature_cols[:1])
    result_vec_single = multilevel_info_decomposition_vectorized(dummy_data, 'target', feature_cols[:1])
    print("Single feature results match:", np.allclose(
        result_orig_single['summary']['Information'],
        result_vec_single['summary']['Information'],
        rtol=1e-10
    ))
    
    print("\nTest case 3: All features with discretization")
    result_orig_all = multilevel_info_decomposition(dummy_data, 'target', feature_cols, discretize=True)
    result_vec_all = multilevel_info_decomposition_vectorized(dummy_data, 'target', feature_cols, discretize=True)
    print("All features results match:", np.allclose(
        result_orig_all['summary']['Information'],
        result_vec_all['summary']['Information'],
        rtol=1e-10
    ))
    
    # Test ordering consistency
    print("\nTest case 4: Ordering consistency")
    orig_orderings = set(result_orig_all['summary']['Ordering'])
    vec_orderings = set(result_vec_all['summary']['Ordering'])
    print("Orderings match:", orig_orderings == vec_orderings)
