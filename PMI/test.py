import numpy as np
import pandas as pd
from pmi import (
    get_probability,
    get_conditional_probability,
    pointwise_mutual_information,
    conditional_pointwise_mutual_information,
    local_mutual_information,
    conditional_local_mutual_information
)

def test_get_probability():
    """Test the get_probability function with known outcomes."""
    print("\n=== Testing get_probability ===")
    
    # Test Case 1: Simple balanced dataset
    test_data1 = pd.DataFrame({
        "Y": ["high", "low", "high", "low", "high", "low"]
    })
    probs1 = get_probability(test_data1, "Y")
    print("\nTest Case 1 - Balanced binary outcomes:")
    print(probs1)
    assert abs(probs1["high"] - 0.5) < 1e-10, "Expected P(high) = 0.5"
    assert abs(probs1["low"] - 0.5) < 1e-10, "Expected P(low) = 0.5"
    assert abs(probs1.sum() - 1.0) < 1e-10, "Probabilities must sum to 1"
    
    # Test Case 2: Unbalanced dataset with three classes
    test_data2 = pd.DataFrame({
        "Y": ["high", "low", "high", "medium", "low", "high"]
    })
    probs2 = get_probability(test_data2, "Y")
    print("\nTest Case 2 - Unbalanced three-class outcomes:")
    print(probs2)
    assert abs(probs2["high"] - 0.5) < 1e-10, "Expected P(high) = 0.5"
    assert abs(probs2["low"] - 1/3) < 1e-10, "Expected P(low) = 1/3"
    assert abs(probs2["medium"] - 1/6) < 1e-10, "Expected P(medium) = 1/6"
    
    # Test Case 3: Empty dataframe
    test_data3 = pd.DataFrame({"Y": []})
    try:
        get_probability(test_data3, "Y")
        assert False, "Expected error for empty dataframe"
    except Exception as e:
        print("\nTest Case 3 - Empty dataframe handled correctly:")
        print(f"Caught expected error: {str(e)}")
    
    print("\nAll get_probability tests passed!")

def test_get_conditional_probability():
    """Test the get_conditional_probability function with known outcomes."""
    print("\n=== Testing get_conditional_probability ===")
    
    # Test Case 1: Simple one-condition case
    test_data1 = pd.DataFrame({
        "Y": ["high", "low", "high", "low", "high", "low"],
        "A": ["x", "x", "x", "y", "y", "y"]
    })
    cond_probs1 = get_conditional_probability(test_data1, "Y", ["A"])
    print("\nTest Case 1 - Single condition:")
    print(cond_probs1)
    # Check P(Y|A=x)
    assert abs(cond_probs1["x", "high"] - 2/3) < 1e-10, "Expected P(high|x) = 2/3"
    assert abs(cond_probs1["x", "low"] - 1/3) < 1e-10, "Expected P(low|x) = 1/3"
    # Check P(Y|A=y)
    assert abs(cond_probs1["y", "high"] - 1/3) < 1e-10, "Expected P(high|y) = 1/3"
    assert abs(cond_probs1["y", "low"] - 2/3) < 1e-10, "Expected P(low|y) = 2/3"
    
    # Test Case 2: Two conditions
    test_data2 = pd.DataFrame({
        "Y": ["high", "low", "high", "low"] * 2,
        "A": ["x", "x", "y", "y"] * 2,
        "B": ["m", "m", "m", "m", "n", "n", "n", "n"]
    })
    cond_probs2 = get_conditional_probability(test_data2, "Y", ["A", "B"])
    print("\nTest Case 2 - Two conditions:")
    print(cond_probs2)
    # Verify probabilities sum to 1 for each condition combination
    for a in ["x", "y"]:
        for b in ["m", "n"]:
            subset = cond_probs2[a, b]
            assert abs(subset.sum() - 1.0) < 1e-10, f"Probabilities for A={a}, B={b} must sum to 1"
    
    print("\nAll get_conditional_probability tests passed!")

def test_pointwise_mutual_information():
    """Test the pointwise_mutual_information function with known outcomes."""
    print("\n=== Testing pointwise_mutual_information ===")
    
    # Test Case 1: Perfect correlation
    test_data1 = pd.DataFrame({
        "Y": ["high", "low"] * 3,
        "A": ["x", "y"] * 3
    })
    pmi1 = pointwise_mutual_information(test_data1, "Y", "A")
    print("\nTest Case 1 - Perfect correlation:")
    print(pmi1)
    # For perfect correlation, PMI should be log(2) ≈ 0.693
    expected_pmi = np.log(2)
    assert all(abs(pmi1 - expected_pmi) < 1e-10), f"Expected PMI = {expected_pmi} for perfect correlation"
    
    # Test Case 2: Independence
    test_data2 = pd.DataFrame({
        "Y": ["high", "high", "low", "low"],
        "A": ["x", "y", "x", "y"]
    })
    pmi2 = pointwise_mutual_information(test_data2, "Y", "A")
    print("\nTest Case 2 - Independence:")
    print(pmi2)
    # For independence, PMI should be 0
    assert all(abs(pmi2) < 1e-10), "Expected PMI = 0 for independence"
    
    print("\nAll pointwise_mutual_information tests passed!")

def test_conditional_pointwise_mutual_information():
    """Test the conditional_pointwise_mutual_information function with known outcomes."""
    print("\n=== Testing conditional_pointwise_mutual_information ===")
    
    # Test Case 1: Conditional independence
    test_data1 = pd.DataFrame({
        "Y": ["high", "low"] * 4,
        "A": ["x", "x", "y", "y"] * 2,
        "B": ["m", "m", "m", "m", "n", "n", "n", "n"]
    })
    cpmi1 = conditional_pointwise_mutual_information(test_data1, "Y", ["A"], ["B"])
    print("\nTest Case 1 - Conditional independence:")
    print(cpmi1)
    # When B is independent of Y given A, conditional PMI should be 0
    assert all(abs(cpmi1) < 1e-10), "Expected conditional PMI = 0 for conditional independence"
    
    print("\nAll conditional_pointwise_mutual_information tests passed!")

def test_local_mutual_information():
    """Test the local_mutual_information function with known outcomes."""
    print("\n=== Testing local_mutual_information ===")
    
    # Test Case 1: Perfect correlation
    test_data1 = pd.DataFrame({
        "Y": ["high", "low"] * 3,
        "A": ["x", "y"] * 3
    })
    lmi1 = local_mutual_information(test_data1, "Y", "A")
    print("\nTest Case 1 - Perfect correlation:")
    print(lmi1)
    # For perfect correlation, LMI should be log(2) ≈ 0.693
    expected_lmi = np.log(2)
    assert all(abs(lmi1 - expected_lmi) < 1e-10), f"Expected LMI = {expected_lmi} for perfect correlation"
    
    print("\nAll local_mutual_information tests passed!")

def test_conditional_local_mutual_information():
    """Test the conditional_local_mutual_information function with known outcomes."""
    print("\n=== Testing conditional_local_mutual_information ===")
    
    # Test Case 1: Conditional independence
    test_data1 = pd.DataFrame({
        "Y": ["high", "low"] * 4,
        "A": ["x", "x", "y", "y"] * 2,
        "B": ["m", "m", "m", "m", "n", "n", "n", "n"]
    })
    clmi1 = conditional_local_mutual_information(test_data1, "Y", ["A"], ["B"])
    print("\nTest Case 1 - Conditional independence:")
    print(clmi1)
    # When B is independent of Y given A, conditional LMI should be 0
    assert all(abs(clmi1) < 1e-10), "Expected conditional LMI = 0 for conditional independence"
    
    print("\nAll conditional_local_mutual_information tests passed!")

def test_high_dimensional_cases():
    """Test high-dimensional cases with known theoretical results."""
    print("\n=== Testing High-Dimensional Cases ===")
    
    # Test Case 1: Perfect Chain Dependency
    # Y -> A -> B -> C (Markov Chain)
    # In this case, Y⊥C|B (Y is independent of C given B)
    n_samples = 1000
    np.random.seed(42)
    
    # Generate perfect chain dependency
    y_vals = np.random.choice(["0", "1"], size=n_samples)
    a_vals = y_vals  # A perfectly depends on Y
    b_vals = a_vals  # B perfectly depends on A
    c_vals = b_vals  # C perfectly depends on B
    
    chain_data = pd.DataFrame({
        "Y": y_vals,
        "A": a_vals,
        "B": b_vals,
        "C": c_vals
    })
    
    print("\nTest Case 1: Perfect Chain Dependency")
    
    # Test 1.1: Direct dependencies should have maximum MI (log(2))
    pmi_y_a = pointwise_mutual_information(chain_data, "Y", "A")
    pmi_a_b = pointwise_mutual_information(chain_data, "A", "B")
    pmi_b_c = pointwise_mutual_information(chain_data, "B", "C")
    
    print("\nDirect dependencies (should be ≈ log(2) ≈ 0.693):")
    print(f"I(Y;A) = {pmi_y_a.mean():.3f}")
    print(f"I(A;B) = {pmi_a_b.mean():.3f}")
    print(f"I(B;C) = {pmi_b_c.mean():.3f}")
    
    # Test 1.2: Conditional independence Y⊥C|B should give zero
    cpmi_y_c_given_b = conditional_pointwise_mutual_information(chain_data, "Y", ["B"], ["C"])
    print("\nConditional independence Y⊥C|B (should be ≈ 0):")
    print(f"I(Y;C|B) = {abs(cpmi_y_c_given_b).mean():.3f}")
    
    # Test Case 2: XOR Relationship
    # Y = A ⊕ B (XOR)
    # This creates a case where Y depends on both A and B jointly
    # but has zero mutual information with each individually
    n_samples = 1000
    a_xor = np.random.choice(["0", "1"], size=n_samples)
    b_xor = np.random.choice(["0", "1"], size=n_samples)
    y_xor = np.array([str(int(a) != int(b)) for a, b in zip(a_xor, b_xor)])
    
    xor_data = pd.DataFrame({
        "Y": y_xor,
        "A": a_xor,
        "B": b_xor
    })
    
    print("\nTest Case 2: XOR Relationship")
    
    # Test 2.1: Individual mutual information should be zero
    pmi_y_a_xor = pointwise_mutual_information(xor_data, "Y", "A")
    pmi_y_b_xor = pointwise_mutual_information(xor_data, "Y", "B")
    
    print("\nIndividual mutual information (should be ≈ 0):")
    print(f"I(Y;A) = {abs(pmi_y_a_xor).mean():.3f}")
    print(f"I(Y;B) = {abs(pmi_y_b_xor).mean():.3f}")
    
    # Test 2.2: Joint mutual information should be maximum (1 bit)
    lmi_y_ab = local_mutual_information(xor_data, "Y", "A", "B")
    print("\nJoint mutual information (should be ≈ log(2) ≈ 0.693):")
    print(f"I(Y;A,B) = {lmi_y_ab.mean():.3f}")
    
    # Test Case 3: High-Dimensional Synergy
    # Create a case where multiple variables interact in a way that creates
    # higher-order dependencies
    n_samples = 1000
    a_syn = np.random.choice(["0", "1"], size=n_samples)
    b_syn = np.random.choice(["0", "1"], size=n_samples)
    c_syn = np.random.choice(["0", "1"], size=n_samples)
    # Y is 1 if majority of A,B,C are 1, creating a synergistic relationship
    y_syn = np.array([str(int((int(a) + int(b) + int(c)) > 1)) 
                      for a, b, c in zip(a_syn, b_syn, c_syn)])
    
    synergy_data = pd.DataFrame({
        "Y": y_syn,
        "A": a_syn,
        "B": b_syn,
        "C": c_syn
    })
    
    print("\nTest Case 3: High-Dimensional Synergy")
    
    # Test 3.1: Pairwise mutual information should be less than three-way
    lmi_y_ab = local_mutual_information(synergy_data, "Y", "A", "B")
    lmi_y_abc = local_mutual_information(synergy_data, "Y", "A", "B", "C")
    
    print("\nComparing pairwise vs three-way mutual information:")
    print(f"I(Y;A,B) = {lmi_y_ab.mean():.3f}")
    print(f"I(Y;A,B,C) = {lmi_y_abc.mean():.3f}")
    print(f"Difference = {(lmi_y_abc.mean() - lmi_y_ab.mean()):.3f} (should be > 0)")

def test_five_dimensional_cases():
    """Test all mutual information measures with 5 inputs."""
    print("\n=== Testing Five-Dimensional Cases ===")
    
    # Create dataset with 5 inputs and complex dependencies
    n_samples = 1000
    np.random.seed(42)
    
    # Generate base variables
    a_vals = np.random.choice(["0", "1"], size=n_samples)
    b_vals = np.random.choice(["0", "1"], size=n_samples)
    c_vals = np.random.choice(["0", "1"], size=n_samples)
    d_vals = np.random.choice(["0", "1"], size=n_samples)
    e_vals = np.random.choice(["0", "1"], size=n_samples)
    
    # Create different types of dependencies:
    # 1. Y1 = A XOR B XOR C (complex 3-way interaction)
    # 2. Y2 = majority(A,B,C,D,E) (5-way voting)
    # 3. Y3 = (A AND B) OR (C AND D) (hierarchical dependency)
    # 4. Y4 = A → B → C → D → E (Markov chain)
    # 5. Y5 = parity(A,B,C,D,E) (all inputs matter)
    
    y1 = np.array([str(int(a) ^ int(b) ^ int(c)) 
                   for a, b, c in zip(a_vals, b_vals, c_vals)])
    
    y2 = np.array([str(int((int(a) + int(b) + int(c) + int(d) + int(e)) > 2))
                   for a, b, c, d, e in zip(a_vals, b_vals, c_vals, d_vals, e_vals)])
    
    y3 = np.array([str(int((int(a) and int(b)) or (int(c) and int(d))))
                   for a, b, c, d in zip(a_vals, b_vals, c_vals, d_vals)])
    
    y4 = a_vals  # Perfect chain A → B → C → D → E
    
    y5 = np.array([str(sum(map(int, [a, b, c, d, e])) % 2)
                   for a, b, c, d, e in zip(a_vals, b_vals, c_vals, d_vals, e_vals)])
    
    # Create dataframes for each type
    data_dict = {
        "XOR3": pd.DataFrame({"Y": y1, "A": a_vals, "B": b_vals, "C": c_vals, 
                            "D": d_vals, "E": e_vals}),
        "MAJ5": pd.DataFrame({"Y": y2, "A": a_vals, "B": b_vals, "C": c_vals, 
                            "D": d_vals, "E": e_vals}),
        "HIER": pd.DataFrame({"Y": y3, "A": a_vals, "B": b_vals, "C": c_vals, 
                            "D": d_vals, "E": e_vals}),
        "CHAIN": pd.DataFrame({"Y": y4, "A": a_vals, "B": b_vals, "C": c_vals, 
                             "D": d_vals, "E": e_vals}),
        "PARITY": pd.DataFrame({"Y": y5, "A": a_vals, "B": b_vals, "C": c_vals, 
                              "D": d_vals, "E": e_vals})
    }
    
    # Test each mutual information measure
    for name, data in data_dict.items():
        print(f"\n=== Testing {name} Relationship ===")
        
        # 1. Test PMI with increasing number of inputs
        print("\nPointwise Mutual Information:")
        for i in range(1, 6):
            vars_subset = ["A", "B", "C", "D", "E"][:i]
            pmi = pointwise_mutual_information(data, "Y", *vars_subset)
            print(f"I(Y;{','.join(vars_subset)}) = {abs(pmi).mean():.3f}")
        
        # 2. Test Conditional PMI with different condition sets
        print("\nConditional Pointwise Mutual Information:")
        test_cases = [
            (["A"], ["B"]),
            (["A", "B"], ["C"]),
            (["A", "B"], ["C", "D"]),
            (["A", "B", "C"], ["D", "E"])
        ]
        for base, cond in test_cases:
            cpmi = conditional_pointwise_mutual_information(data, "Y", base, cond)
            print(f"I(Y;{','.join(cond)}|{','.join(base)}) = {abs(cpmi).mean():.3f}")
        
        # 3. Test LMI with increasing number of inputs
        print("\nLocal Mutual Information:")
        for i in range(1, 6):
            vars_subset = ["A", "B", "C", "D", "E"][:i]
            lmi = local_mutual_information(data, "Y", *vars_subset)
            print(f"I(Y;{','.join(vars_subset)}) = {lmi.mean():.3f}")
        
        # 4. Test Conditional LMI with different condition sets
        print("\nConditional Local Mutual Information:")
        for base, cond in test_cases:
            clmi = conditional_local_mutual_information(data, "Y", base, cond)
            if not clmi.empty:
                print(f"I(Y;{','.join(cond)}|{','.join(base)}) = {clmi.mean():.3f}")
            else:
                print(f"I(Y;{','.join(cond)}|{','.join(base)}) = 0.000")

if __name__ == "__main__":
    # Run all tests
    test_get_probability()
    test_get_conditional_probability()
    test_pointwise_mutual_information()
    test_conditional_pointwise_mutual_information()
    test_local_mutual_information()
    test_conditional_local_mutual_information()
    test_high_dimensional_cases()
    test_five_dimensional_cases()
    print("\nAll tests completed successfully!") 