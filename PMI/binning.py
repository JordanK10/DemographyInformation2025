import numpy as np
import pandas as pd

def create_variable_bins(data, variable, n_bins=None):
    """
    Create bins for a variable, handling both numerical and categorical data.
    For categorical variables, randomly assign values to bins.
    
    Args:
        data (pd.DataFrame): Input data containing the variable
        variable (str): Name of the variable to bin
        n_bins (int, optional): Number of bins to create. If None, will be determined automatically.
        
    Returns:
        dict: Mapping of original values to bin labels
    """
    # Get unique values
    unique_values = data[variable].unique()
    n_unique = len(unique_values)
    
    # If variable is categorical
    if data[variable].dtype == 'object':
        # Randomly assign values to bins
        np.random.seed(42)  # For reproducibility
        if n_bins is None:
            n_bins = min(5, n_unique)  # Default to 5 bins or less if fewer unique values
        
        # Randomly assign values to bins
        bin_indices = np.random.randint(0, n_bins, size=n_unique)
        
        # Create descriptive bin labels based on the values in each bin
        bin_mapping = {}
        bin_contents = {}
        for val, bin_idx in zip(unique_values, bin_indices):
            bin_name = f"Bin_{bin_idx+1}"
            bin_mapping[val] = bin_name
            if bin_name not in bin_contents:
                bin_contents[bin_name] = []
            bin_contents[bin_name].append(val)
        
        # Create descriptive labels based on bin contents
        descriptive_labels = {}
        for bin_name, values in bin_contents.items():
            if len(values) <= 3:
                # If few values, list them all
                descriptive_labels[bin_name] = f"{variable}={','.join(map(str, values))}"
            else:
                # If many values, show range
                descriptive_labels[bin_name] = f"{variable}={values[0]}-{values[-1]}"
        
        return bin_mapping, descriptive_labels
    
    # For numerical variables, use quantile-based binning
    if n_bins is None:
        # Use Sturges' rule for automatic bin selection
        n_bins = int(np.log2(n_unique) + 1)
    
    # Create bins using pandas qcut for numerical variables
    try:
        # Try to create quantile-based bins
        bins = pd.qcut(data[variable], q=n_bins, labels=False, duplicates='drop')
        bin_edges = pd.qcut(data[variable], q=n_bins, retbins=True)[1]
        
        # Create mapping of values to bin labels
        bin_mapping = {}
        bin_contents = {}
        for val in unique_values:
            bin_idx = np.digitize(val, bin_edges) - 1
            bin_idx = min(bin_idx, n_bins - 1)  # Handle edge case
            bin_name = f"Bin_{bin_idx+1}"
            bin_mapping[val] = bin_name
            if bin_name not in bin_contents:
                bin_contents[bin_name] = []
            bin_contents[bin_name].append(val)
        
        # Create descriptive labels based on bin edges
        descriptive_labels = {}
        for bin_name, values in bin_contents.items():
            bin_idx = int(bin_name.split('_')[1]) - 1
            if bin_idx == 0:
                descriptive_labels[bin_name] = f"{variable}≤{bin_edges[0]:.1f}"
            elif bin_idx == n_bins - 1:
                descriptive_labels[bin_name] = f"{variable}>{bin_edges[-2]:.1f}"
            else:
                descriptive_labels[bin_name] = f"{bin_edges[bin_idx-1]:.1f}<{variable}≤{bin_edges[bin_idx]:.1f}"
        
        return bin_mapping, descriptive_labels
    
    except ValueError:
        # If qcut fails (e.g., too many duplicate values), use regular cut
        bins = pd.cut(data[variable], bins=n_bins, labels=False, duplicates='drop')
        bin_edges = pd.cut(data[variable], bins=n_bins, retbins=True)[1]
        
        # Create mapping of values to bin labels
        bin_mapping = {}
        bin_contents = {}
        for val in unique_values:
            bin_idx = np.digitize(val, bin_edges) - 1
            bin_idx = min(bin_idx, n_bins - 1)  # Handle edge case
            bin_name = f"Bin_{bin_idx+1}"
            bin_mapping[val] = bin_name
            if bin_name not in bin_contents:
                bin_contents[bin_name] = []
            bin_contents[bin_name].append(val)
        
        # Create descriptive labels based on bin edges
        descriptive_labels = {}
        for bin_name, values in bin_contents.items():
            bin_idx = int(bin_name.split('_')[1]) - 1
            if bin_idx == 0:
                descriptive_labels[bin_name] = f"{variable}≤{bin_edges[0]:.1f}"
            elif bin_idx == n_bins - 1:
                descriptive_labels[bin_name] = f"{variable}>{bin_edges[-2]:.1f}"
            else:
                descriptive_labels[bin_name] = f"{bin_edges[bin_idx-1]:.1f}<{variable}≤{bin_edges[bin_idx]:.1f}"
        
        return bin_mapping, descriptive_labels

def create_all_variable_bins(data, variables, n_bins=None):
    """
    Create bins for multiple variables.
    
    Args:
        data (pd.DataFrame): Input data containing the variables
        variables (list): List of variable names to bin
        n_bins (int, optional): Number of bins to create for each variable
        
    Returns:
        tuple: (bin_mappings, descriptive_labels) where:
            - bin_mappings: Dictionary mapping variable names to their bin mappings
            - descriptive_labels: Dictionary mapping variable names to their descriptive bin labels
    """
    bin_mappings = {}
    descriptive_labels = {}
    for var in variables:
        bin_mappings[var], descriptive_labels[var] = create_variable_bins(data, var, n_bins)
    return bin_mappings, descriptive_labels

def apply_bin_mappings(data, bin_mappings):
    """
    Apply bin mappings to create binned versions of variables.
    
    Args:
        data (pd.DataFrame): Input data
        bin_mappings (dict): Dictionary of bin mappings for each variable
        
    Returns:
        pd.DataFrame: New dataframe with binned variables
    """
    binned_data = data.copy()
    for var, mapping in bin_mappings.items():
        binned_data[f"{var}_binned"] = data[var].map(mapping)
    return binned_data
