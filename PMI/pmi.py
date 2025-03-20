import numpy as np
import pandas as pd

def _mask_zero_probabilities(probabilities, epsilon=1e-10):
    """
    Create a mask for probabilities that are effectively zero using numpy.
    
    Args:
        probabilities (pd.Series or np.ndarray): Series or array of probabilities
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        np.ndarray: Boolean mask where True indicates non-zero probabilities
    """
    if isinstance(probabilities, pd.Series):
        probabilities = probabilities.values
    return probabilities > epsilon

def get_probability(data, target, epsilon=1e-10):
    """
    Compute the unconditional probability P(y) for each outcome y in the target variable.
    Zero probabilities are filtered out. Vectorized implementation using numpy.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (str): The target variable column name.
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        pd.Series: A series mapping each outcome to its probability.
    """
    # Convert to numpy array if needed
    values = data[target].values
    # Get unique values and counts
    unique_vals, counts = np.unique(values, return_counts=True)
    # Compute probabilities
    probabilities = counts / len(values)
    # Create mask and filter
    mask = _mask_zero_probabilities(probabilities, epsilon)
    # Return as pandas Series with proper index
    return pd.Series(probabilities[mask], index=unique_vals[mask])

def get_conditional_probability(data, target, conditions, epsilon=1e-10):
    """
    Compute the conditional probability P(y | conditions) for each outcome y given the conditions.
    Zero probabilities are filtered out. Optimized implementation using numpy operations.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (str): The target variable column name.
        conditions (list): List of column names to condition on.
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        pd.Series: A Series with a MultiIndex (conditions + target) containing the conditional probabilities.
    """
    # Handle empty conditions case
    if not conditions:
        return get_probability(data, target, epsilon)
        
    # Convert conditions to numpy arrays for faster operations
    condition_arrays = [data[cond].values for cond in conditions]
    target_array = data[target].values
    
    # Create a structured array for faster unique operations
    dtype = [(f'cond_{i}', arr.dtype) for i, arr in enumerate(condition_arrays)]
    dtype.append(('target', target_array.dtype))
    
    # Combine all arrays into structured array
    combined = np.empty(len(target_array), dtype=dtype)
    for i, arr in enumerate(condition_arrays):
        combined[f'cond_{i}'] = arr
    combined['target'] = target_array
    
    # Get unique combinations and counts
    unique_combs, counts = np.unique(combined, return_counts=True)
    
    # Convert back to pandas for the groupby operation
    df_counts = pd.DataFrame(unique_combs)
    df_counts['count'] = counts
    
    # Group by conditions and normalize
    group_cols = [f'cond_{i}' for i in range(len(conditions))]
    cond_probs = df_counts.groupby(group_cols)['count'].transform(lambda x: x / x.sum())
    
    # Create proper index
    index = pd.MultiIndex.from_arrays(
        [unique_combs[col] for col in group_cols + ['target']],
        names=conditions + [target]
    )
    
    # Create Series with proper index
    result = pd.Series(cond_probs.values, index=index)
    
    # Apply zero probability masking
    mask = _mask_zero_probabilities(result, epsilon)
    return result[mask]

def pointwise_mutual_information(data, target, *conditions, epsilon=1e-10):
    """
    Compute pointwise mutual information (PMI) for each combination of condition outcomes and target outcomes.
    Zero probabilities are filtered out. Vectorized implementation using numpy operations.
    
    i(y; conditions) = ln( P(y|conditions) / P(y) )
    
    Note: Natural logarithm (base e) is used in the calculation.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (str): The target variable column name.
        *conditions (str): Arbitrary number of condition variable names.
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        pd.Series: A Series with a MultiIndex (conditions + target) containing the PMI values.
    """
    # Get probabilities
    p_y = get_probability(data, target, epsilon)
    cond_probs = get_conditional_probability(data, target, list(conditions), epsilon)
    
    # Convert to numpy for faster operations
    y_vals = cond_probs.index.get_level_values(-1)
    p_y_vals = p_y.reindex(y_vals).values
    
    # Compute PMI vectorized
    valid_mask = ~np.isnan(p_y_vals)
    pmi = pd.Series(index=cond_probs.index, dtype=float)
    pmi[valid_mask] = np.log(cond_probs[valid_mask] / p_y_vals[valid_mask])
    
    return pmi.dropna()

def conditional_pointwise_mutual_information(data, target, base_conditions, conditioned_conditions, epsilon=1e-10):
    """
    Compute the conditional pointwise mutual information:
    
    i(y; conditioned_conditions | base_conditions) = ln( P(y|base_conditions, conditioned_conditions) / P(y|base_conditions) )
    
    Zero probabilities are filtered out. Vectorized implementation using numpy operations.
    Note: Natural logarithm (base e) is used in the calculation.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (str): The target variable column name.
        base_conditions (list): List of column names that form the base condition (denominator).
        conditioned_conditions (list): List of additional condition names (numerator).
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        pd.Series: A Series with a MultiIndex (base_conditions, conditioned_conditions, target) containing the conditional PMI values.
        Index structure handles the hierarchical nature of conditioning.
    """
    # Get probabilities
    numerator = get_conditional_probability(data, target, base_conditions + conditioned_conditions, epsilon)
    denominator = get_conditional_probability(data, target, base_conditions, epsilon)
    
    # Create index mapping for efficient lookup
    num_base = len(base_conditions)
    
    # Extract base values and target values using numpy operations
    base_vals = [numerator.index.get_level_values(i) for i in range(num_base)]
    target_vals = numerator.index.get_level_values(-1)
    
    # Create lookup index
    lookup_idx = pd.MultiIndex.from_arrays([*base_vals, target_vals])
    
    # Vectorized division with proper indexing
    denom_vals = denominator.reindex(lookup_idx)
    
    # Ensure proper alignment
    valid_mask = ~np.isnan(denom_vals.values)
    numerator_vals = numerator.values
    denom_vals = denom_vals.values
    
    # Initialize result Series
    c_pmi = pd.Series(np.nan, index=numerator.index, dtype=float)
    
    # Compute conditional PMI where valid
    c_pmi.values[valid_mask] = np.log(numerator_vals[valid_mask] / denom_vals[valid_mask])
    
    return c_pmi.dropna()

def local_mutual_information(data, target, *conditions, epsilon=1e-10):
    """
    Compute the local mutual information (LMI) by averaging the pointwise mutual information over y:
    
    LMI(conditions) = sum_y P(y|conditions) * i(y; conditions)
    
    Zero probabilities are filtered out. Implementation uses vectorized operations where possible,
    with condition-wise grouping performed iteratively for proper aggregation.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (str): The target variable column name.
        *conditions (str): Condition variable names.
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        pd.Series: A series indexed by the condition variables containing the LMI values.
        Values below epsilon are filtered out from the final result.
    """
    # Get probabilities and PMI
    cond_probs = get_conditional_probability(data, target, list(conditions), epsilon)
    pmi = pointwise_mutual_information(data, target, *conditions, epsilon=epsilon)
    
    # Ensure alignment
    aligned_probs = cond_probs.reindex(pmi.index)
    
    # Compute LMI using numpy operations
    lmi = pd.Series(
        np.zeros(len(np.unique(pmi.index.droplevel(-1)))),
        index=np.unique(pmi.index.droplevel(-1))
    )
    
    # Group by conditions and compute sum
    for cond in lmi.index:
        mask = tuple(slice(None) if i == len(cond) else c for i, c in enumerate(cond))
        lmi[cond] = np.sum(aligned_probs[mask] * pmi[mask])
    
    return lmi[abs(lmi) > epsilon]

def conditional_local_mutual_information(data, target, base_conditions, conditioned_conditions, epsilon=1e-10):
    """
    Compute the conditional local mutual information (conditional LMI) by averaging the conditional PMI over y:
    
    conditional LMI = sum_y P(y|base_conditions, conditioned_conditions) * i(y; conditioned_conditions | base_conditions)
    
    Zero probabilities are filtered out. Implementation uses vectorized operations where possible,
    with condition-wise grouping performed iteratively for proper aggregation.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        target (str): The target variable column name.
        base_conditions (list): List of column names for the base condition (denominator).
        conditioned_conditions (list): List of additional conditioning variables.
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        pd.Series: A series indexed by the base_conditions and conditioned_conditions containing the conditional LMI values.
        Values below epsilon are filtered out from the final result.
    """
    # Get probabilities and conditional PMI
    cond_probs = get_conditional_probability(data, target, base_conditions + conditioned_conditions, epsilon)
    c_pmi = conditional_pointwise_mutual_information(data, target, base_conditions, conditioned_conditions, epsilon=epsilon)
    
    # Ensure alignment
    aligned_probs = cond_probs.reindex(c_pmi.index)
    
    # Compute conditional LMI using numpy operations
    conditions = base_conditions + conditioned_conditions
    cond_lmi = pd.Series(
        np.zeros(len(np.unique(c_pmi.index.droplevel(-1)))),
        index=np.unique(c_pmi.index.droplevel(-1))
    )
    
    # Group by conditions and compute sum
    for cond in cond_lmi.index:
        mask = tuple(slice(None) if i == len(cond) else c for i, c in enumerate(cond))
        cond_lmi[cond] = np.sum(aligned_probs[mask] * c_pmi[mask])
    
    return cond_lmi[abs(cond_lmi) > epsilon]

def mutual_information(data, target, *conditions, epsilon=1e-10):
    """
    Compute mutual information using joint probabilities:
    MI(Y; X1,...,Xn) = ∑∑ P(x,y) * log(P(y|x)/P(y))
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        target (str): Target variable column name
        *conditions: Variable number of condition column names
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        float: The mutual information value
    """
    # Handle empty conditions
    if not conditions:
        return 0.0
    
    # Get pointwise mutual information
    pmi = pointwise_mutual_information(data, target, *conditions, epsilon=epsilon)
    
    # For empty or invalid results, return 0
    if isinstance(pmi, pd.Series) and pmi.empty:
        return 0.0
    
    # Calculate joint probabilities P(x,y)
    # First get counts of each combination
    all_cols = list(conditions) + [target]
    value_counts = data[all_cols].value_counts()
    joint_probs = value_counts / len(data)
    
    # Ensure proper alignment of indices
    aligned_joint_probs = joint_probs.reindex(pmi.index).fillna(0)
    
    # Compute weighted average using joint probabilities
    mi = np.sum(pmi * aligned_joint_probs)
    return max(0, mi)  # Ensure non-negative

def conditional_mutual_information(data, target, base_conditions, conditioned_conditions, epsilon=1e-10):
    """
    Compute conditional mutual information by averaging conditional pointwise mutual information.
    This function handles arbitrary inputs for both base and conditioned conditions.
    
    CMI(Y; X1,...,Xn | Z1,...,Zm) = ∑∑ P(y,x,z) * log(P(y|x,z)/P(y|z))
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        target (str): Target variable column name
        base_conditions (list): List of column names for base conditions (Z1,...,Zm)
        conditioned_conditions (list): List of column names for conditioned variables (X1,...,Xn)
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        float: The conditional mutual information value
    """
    # Handle empty conditions
    if not base_conditions or not conditioned_conditions:
        return 0.0
        
    # Convert single strings to lists for consistency
    if isinstance(base_conditions, str):
        base_conditions = [base_conditions]
    if isinstance(conditioned_conditions, str):
        conditioned_conditions = [conditioned_conditions]
    
    # Get conditional pointwise mutual information
    cpmi = conditional_pointwise_mutual_information(
        data, target, base_conditions, conditioned_conditions, epsilon=epsilon
    )
    
    # For empty or invalid results, return 0
    if isinstance(cpmi, pd.Series) and cpmi.empty:
        return 0.0
    
    # Calculate joint probabilities P(y,x,z)
    all_cols = base_conditions + conditioned_conditions + [target]
    value_counts = data[all_cols].value_counts()
    joint_probs = value_counts / len(data)
    
    # Ensure proper alignment of indices
    aligned_joint_probs = joint_probs.reindex(cpmi.index).fillna(0)
    
    # Compute weighted average using joint probabilities
    cmi = np.sum(cpmi * aligned_joint_probs)
    return max(0, cmi)  # Ensure non-negative

def entropy(data, target, epsilon=1e-10):
    """
    Compute the entropy of a variable in the data.
    
    H(X) = -∑ P(x) log P(x)
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        target (str): Column name of the variable to compute entropy for
        epsilon (float): Threshold below which probabilities are considered zero
        
    Returns:
        float: The entropy value in nats (using natural logarithm)
    """
    # Get probability distribution using existing function
    probabilities = get_probability(data, target, epsilon)
    
    # Compute entropy using the formula H(X) = -∑ P(x) log P(x)
    entropy_value = -np.sum(probabilities * np.log(probabilities))
    
    return entropy_value

# Example usage:
if __name__ == "__main__":
    # Create a synthetic dataset for demonstration.
    df = pd.DataFrame({
        "Y": np.random.choice(["high", "low"], size=1000),
        "a": np.random.choice(["x", "y"], size=1000),
        "b": np.random.choice(["m", "n"], size=1000),
        "c": np.random.choice(["u", "v"], size=1000)
    })
    
    # Example 1: Compute PMI for Y given a, b, and c (joint input case).
    pmi_vals = pointwise_mutual_information(df, "Y", "a", "b", "c")
    print("Pointwise Mutual Information (PMI) for Y given a, b, and c:")
    print(pmi_vals.head())
    
    # Example 2: Compute LMI by averaging PMI over Y.
    lmi_vals = local_mutual_information(df, "Y", "a", "b", "c")
    print("\nLocal Mutual Information (LMI) for inputs a, b, and c:")
    print(lmi_vals.head())
    
    # Example 3: Compute conditional PMI for Y given base condition a and additional conditions b and c.
    cpmi_vals = conditional_pointwise_mutual_information(df, "Y", ["a"], ["b", "c"])
    print("\nConditional Pointwise Mutual Information (cPMI) for Y given a; additional conditions b and c:")
    print(cpmi_vals.head())
    
    # Example 4: Compute conditional LMI by averaging conditional PMI over Y.
    cond_lmi_vals = conditional_local_mutual_information(df, "Y", ["a"], ["b", "c"])
    print("\nConditional Local Mutual Information (conditional LMI) for a; additional conditions b and c:")
    print(cond_lmi_vals.head())

    # Create a simple test dataset
    test_data = pd.DataFrame({
        "Y": ["high", "low", "high", "medium", "low", "high"]
    })
    
    # Test get_probability
    print("Testing get_probability:")
    probs = get_probability(test_data, "Y")
    print("\nProbabilities:")
    print(probs)
    print("\nVerification:")
    print(f"Sum of probabilities: {probs.sum():.2f} (should be 1.0)")
    print(f"All probabilities >= 0: {all(probs >= 0)}")
    
    # Expected probabilities:
    # high: 3/6 = 0.5
    # low: 2/6 ≈ 0.33
    # medium: 1/6 ≈ 0.17
    expected = pd.Series({
        "high": 0.5,
        "low": 1/3,
        "medium": 1/6
    })
    print("\nExpected vs Actual differences:")
    print(abs(expected - probs).sum())
