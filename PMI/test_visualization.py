import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualizationCode import (
    analyze_three_variable_decomposition,
    plot_feature_importance_by_level,
    plot_feature_importance_by_ordering,
    plot_feature_importance_by_level_and_ordering
)

def create_test_data():
    """Create synthetic test data with known relationships."""
    np.random.seed(42)
    n_samples = 100  # Reduced sample size for testing
    
    # Create simple categorical variables
    age = np.random.choice(['young', 'old'], n_samples)
    education = np.random.choice(['low', 'high'], n_samples)
    income = np.random.choice(['poor', 'rich'], n_samples)
    gender = np.random.choice(['m', 'f'], n_samples)
    city = np.random.choice(['A', 'B'], n_samples)
    
    # Create DataFrame with simple categorical values
    return pd.DataFrame({
        'income': income,
        'age': age,
        'education': education,
        'gender': gender,
        'city': city
    })

def test_analyze_three_variable_decomposition():
    """Test three-variable decomposition analysis."""
    print("\nTesting analyze_three_variable_decomposition...")
    
    # Create test data
    data = create_test_data()
    
    # Test case 1: Basic functionality
    results = analyze_three_variable_decomposition(
        data, 'income', 'age', 'education', 'gender'
    )
    
    # Verify results structure
    assert isinstance(results, pd.DataFrame), "Results should be a DataFrame"
    assert all(col in results.columns for col in ['Ordering', 'Level', 'Variable', 'Information']), \
        "Results should have required columns"
    
    # Verify number of results (6 orderings * 3 levels = 18 rows)
    assert len(results) == 18, f"Expected 18 rows, got {len(results)}"
    
    # Verify levels
    assert set(results['Level']) == {1, 2, 3}, "Should have exactly 3 levels"
    
    print("Three-variable decomposition tests passed!")

def test_plot_feature_importance_by_level():
    """Test feature importance by level plotting."""
    print("\nTesting plot_feature_importance_by_level...")
    
    # Create test data
    data = create_test_data()
    feature_cols = ['age', 'education', 'gender', 'city']
    
    # Test case 1: Basic functionality
    fig = plot_feature_importance_by_level(
        data, 'income', feature_cols
    )
    
    # Verify figure properties
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    assert len(fig.axes) == 1, "Should have one axis"
    
    # Verify plot elements
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Features', "X-axis label should be 'Features'"
    assert 'Information Content' in ax.get_ylabel(), "Y-axis label should contain 'Information Content'"
    assert len(ax.patches) > 0, "Should have bars in the plot"
    
    # Clean up
    plt.close(fig)
    print("Feature importance by level plotting tests passed!")

def test_plot_feature_importance_by_ordering():
    """Test feature importance by ordering plotting."""
    print("\nTesting plot_feature_importance_by_ordering...")
    
    # Create test data
    data = create_test_data()
    feature_cols = ['age', 'education', 'gender']  # Using fewer features for clarity
    
    # Test case 1: Basic functionality
    fig = plot_feature_importance_by_ordering(
        data, 'income', feature_cols
    )
    
    # Verify figure properties
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    assert len(fig.axes) == 1, "Should have one axis"
    
    # Verify plot elements
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Features', "X-axis label should be 'Features'"
    assert 'Information Content' in ax.get_ylabel(), "Y-axis label should contain 'Information Content'"
    assert len(ax.patches) > 0, "Should have bars in the plot"
    
    # Clean up
    plt.close(fig)
    print("Feature importance by ordering plotting tests passed!")

def test_plot_feature_importance_by_level_and_ordering():
    """Test feature importance by level and ordering plotting."""
    print("\nTesting plot_feature_importance_by_level_and_ordering...")
    
    # Create test data
    data = create_test_data()
    feature_cols = ['age', 'education', 'gender']  # Using fewer features for clarity
    
    # Test case 1: Basic functionality
    fig = plot_feature_importance_by_level_and_ordering(
        data, 'income', feature_cols
    )
    
    # Verify figure properties
    assert isinstance(fig, plt.Figure), "Should return a matplotlib Figure"
    assert len(fig.axes) == 2, "Should have two facets (one for each level)"
    
    # Verify plot elements for each facet
    for ax in fig.axes:
        assert 'Information Content' in ax.get_ylabel(), "Y-axis label should contain 'Information Content'"
        assert len(ax.patches) > 0, "Should have bars in the plot"
    
    # Clean up
    plt.close(fig)
    print("Feature importance by level and ordering plotting tests passed!")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Create test data
    data = create_test_data()
    
    # Test case 1: Single feature
    try:
        fig = plot_feature_importance_by_level(data, 'income', ['age'])
        plt.close(fig)
        print("Single feature case handled")
    except Exception as e:
        print(f"Single feature case failed: {str(e)}")
    
    # Test case 2: Empty feature list
    try:
        plot_feature_importance_by_level(data, 'income', [])
        print("ERROR: Empty feature list should raise an error")
    except ValueError:
        print("Empty feature list correctly raises ValueError")
    
    # Test case 3: Invalid target column
    try:
        plot_feature_importance_by_level(data, 'invalid_column', ['age'])
        print("ERROR: Invalid target column should raise an error")
    except KeyError:
        print("Invalid target column correctly raises KeyError")
    
    print("Edge cases testing completed!")

if __name__ == "__main__":
    # Run all tests
    test_analyze_three_variable_decomposition()
    test_plot_feature_importance_by_level()
    test_plot_feature_importance_by_ordering()
    test_plot_feature_importance_by_level_and_ordering()
    test_edge_cases()
    print("\nAll tests completed!") 