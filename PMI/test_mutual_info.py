import numpy as np
import pandas as pd
from pmi import (
    pointwise_mutual_information,
    conditional_pointwise_mutual_information,
    mutual_information,
    conditional_mutual_information,
    get_conditional_probability,
    get_probability
)

def test_perfect_correlation():
    """Test mutual information with perfect correlation (should be log(2))"""
    print("\n=== Testing Perfect Correlation ===")
    
    # Create perfectly correlated variables
    data = pd.DataFrame({
        'Y': ['0', '1'] * 50,
        'X': ['0', '1'] * 50
    })
    
    # Theoretical value for perfect correlation is log(2)
    theoretical_mi = np.log(2)
    
    # Calculate MI using our function
    mi = mutual_information(data, 'Y', 'X')
    
    # Calculate MI using PMI manually for verification
    pmi = pointwise_mutual_information(data, 'Y', 'X')
    joint_probs = get_conditional_probability(data, 'Y', ['X'])
    manual_mi = np.sum(pmi * joint_probs)
    
    print(f"Theoretical MI: {theoretical_mi:.6f}")
    print(f"Calculated MI: {mi:.6f}")
    print(f"Manual MI from PMI: {manual_mi:.6f}")
    print(f"Difference: {abs(mi - theoretical_mi):.6f}")
    assert abs(mi - theoretical_mi) < 1e-10, "MI calculation incorrect for perfect correlation"

def test_independence():
    """Test mutual information with independent variables (should be 0)"""
    print("\n=== Testing Independence ===")
    
    # Create independent variables
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'Y': np.random.choice(['0', '1'], size=n),
        'X': np.random.choice(['0', '1'], size=n)
    })
    
    # Calculate MI
    mi = mutual_information(data, 'Y', 'X')
    
    # Calculate MI using PMI manually
    pmi = pointwise_mutual_information(data, 'Y', 'X')
    joint_probs = get_conditional_probability(data, 'Y', ['X'])
    manual_mi = np.sum(pmi * joint_probs)
    
    print(f"MI for independent variables: {mi:.6f}")
    print(f"Manual MI from PMI: {manual_mi:.6f}")
    assert mi < 0.1, "MI should be close to 0 for independent variables"

def test_conditional_perfect_correlation():
    """Test conditional mutual information with perfect correlation given condition"""
    print("\n=== Testing Conditional Perfect Correlation ===")
    
    # Create data where Y and X are perfectly correlated given Z=1
    data = pd.DataFrame({
        'Y': ['0', '1', '0', '1'] * 25,
        'X': ['0', '1', '0', '1'] * 25,
        'Z': ['1', '1', '0', '0'] * 25
    })
    
    # Calculate CMI
    cmi = conditional_mutual_information(data, 'Y', ['Z'], ['X'])
    
    # Calculate CMI using CPMI manually
    cpmi = conditional_pointwise_mutual_information(data, 'Y', ['Z'], ['X'])
    joint_probs = get_conditional_probability(data, 'Y', ['Z', 'X'])
    manual_cmi = np.sum(cpmi * joint_probs)
    
    print(f"CMI: {cmi:.6f}")
    print(f"Manual CMI from CPMI: {manual_cmi:.6f}")
    print(f"Difference: {abs(cmi - manual_cmi):.6f}")
    assert abs(cmi - manual_cmi) < 1e-10, "CMI calculation incorrect"

def test_conditional_independence():
    """Test conditional mutual information with conditional independence"""
    print("\n=== Testing Conditional Independence ===")
    
    # Create data where Y and X are independent given Z
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'Y': np.random.choice(['0', '1'], size=n),
        'X': np.random.choice(['0', '1'], size=n),
        'Z': np.random.choice(['0', '1'], size=n)
    })
    
    # Calculate CMI
    cmi = conditional_mutual_information(data, 'Y', ['Z'], ['X'])
    
    # Calculate CMI using CPMI manually
    cpmi = conditional_pointwise_mutual_information(data, 'Y', ['Z'], ['X'])
    joint_probs = get_conditional_probability(data, 'Y', ['Z', 'X'])
    manual_cmi = np.sum(cpmi * joint_probs)
    
    print(f"CMI for conditional independence: {cmi:.6f}")
    print(f"Manual CMI from CPMI: {manual_cmi:.6f}")
    assert cmi < 0.1, "CMI should be close to 0 for conditional independence"

def test_chain_rule():
    """Test the chain rule of mutual information"""
    print("\n=== Testing Chain Rule ===")
    
    # Create data with known dependencies
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'Y': np.random.choice(['0', '1'], size=n),
        'X1': np.random.choice(['0', '1'], size=n),
        'X2': np.random.choice(['0', '1'], size=n)
    })
    
    # Calculate individual and joint mutual information
    mi_1 = mutual_information(data, 'Y', 'X1')
    mi_2_given_1 = conditional_mutual_information(data, 'Y', ['X1'], ['X2'])
    mi_joint = mutual_information(data, 'Y', 'X1', 'X2')
    
    print(f"I(Y;X1) = {mi_1:.6f}")
    print(f"I(Y;X2|X1) = {mi_2_given_1:.6f}")
    print(f"I(Y;X1,X2) = {mi_joint:.6f}")
    print(f"I(Y;X1) + I(Y;X2|X1) = {mi_1 + mi_2_given_1:.6f}")
    assert abs(mi_joint - (mi_1 + mi_2_given_1)) < 0.1, "Chain rule not satisfied"

def test_arbitrary_inputs():
    """Test handling of arbitrary numbers of variables"""
    print("\n=== Testing Arbitrary Inputs ===")
    
    # Create data with multiple variables
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'Y': np.random.choice(['0', '1'], size=n),
        'X1': np.random.choice(['0', '1'], size=n),
        'X2': np.random.choice(['0', '1'], size=n),
        'X3': np.random.choice(['0', '1'], size=n),
        'X4': np.random.choice(['0', '1'], size=n)
    })
    
    # Test with increasing number of conditions
    for i in range(1, 5):
        conditions = [f'X{j}' for j in range(1, i+1)]
        mi = mutual_information(data, 'Y', *conditions)
        print(f"MI with {i} conditions: {mi:.6f}")
        assert not np.isnan(mi), f"MI calculation failed with {i} conditions"
    
    # Test conditional MI with multiple base and conditioned variables
    cmi = conditional_mutual_information(
        data, 'Y', 
        base_conditions=['X1', 'X2'],
        conditioned_conditions=['X3', 'X4']
    )
    print(f"CMI with multiple conditions: {cmi:.6f}")
    assert not np.isnan(cmi), "CMI calculation failed with multiple conditions"

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")
    
    # Create simple dataset
    data = pd.DataFrame({
        'Y': ['0', '1'] * 5,
        'X': ['0', '1'] * 5,
        'Z': ['0'] * 10  # Constant variable
    })
    
    # Test with constant variable
    mi_const = mutual_information(data, 'Y', 'Z')
    print(f"MI with constant variable: {mi_const:.6f}")
    assert mi_const == 0, "MI should be 0 for constant variable"
    
    # Test with empty conditions
    mi_empty = mutual_information(data, 'Y')
    print(f"MI with no conditions: {mi_empty:.6f}")
    assert mi_empty == 0, "MI should be 0 with no conditions"
    
    # Test CMI with empty conditions
    cmi_empty = conditional_mutual_information(data, 'Y', [], [])
    print(f"CMI with empty conditions: {cmi_empty:.6f}")
    assert cmi_empty == 0, "CMI should be 0 with empty conditions"
    
    # Test with single string input for conditions
    cmi_string = conditional_mutual_information(data, 'Y', 'X', 'Z')
    print(f"CMI with string conditions: {cmi_string:.6f}")
    assert not np.isnan(cmi_string), "CMI should handle string conditions"

if __name__ == "__main__":
    # Run all tests
    test_perfect_correlation()
    test_independence()
    test_conditional_perfect_correlation()
    test_conditional_independence()
    test_chain_rule()
    test_arbitrary_inputs()
    test_edge_cases()
    print("\nAll tests completed successfully!") 