# Test function to compare implementations
def test_mutual_info_implementations():
    # Create test cases
    test_cases = [
        (data['Gender'], data['profession_4']),
        (data['Gender'], data['education']),
        (pd.qcut(data['income'], q=10), data['Gender'])
    ]
    
    print("Testing mutual information implementations:")
    for i, (x, y) in enumerate(test_cases, 1):
        result = calculate_mutual_info(x, y)
        print(f"\nTest case {i}:")
        print(f"Result: {result}")
        
        # Test with sklearn's mutual_info_score for validation
        from sklearn.metrics import mutual_info_score
        sklearn_result = mutual_info_score(x, y)
        print(f"Sklearn result: {sklearn_result}")
        print(f"Difference: {abs(result - sklearn_result)}")

def test_analyze_feature_combinations():
    print("\nTesting analyze_feature_combinations implementations...")
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 500
    
    dummy_data = pd.DataFrame({
        'target': np.random.normal(0, 1, n_samples),
        'feat1': ['A', 'B'] * (n_samples // 2),
        'feat2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'feat3': np.random.uniform(0, 10, n_samples),
        'feat4': pd.qcut(np.random.normal(0, 1, n_samples), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    })
    
    feature_cols = ['feat1', 'feat2', 'feat3', 'feat4']
    
    print("\nTest case 1: Basic functionality (2 features)")
    result = analyze_feature_combinations(dummy_data, 'target', feature_cols[:2], max_features=2)
    print("Shape:", result.shape)
    print("Contains expected columns:", all(col in result.columns 
                                         for col in ['Features', 'Num_Features', 'Total_Information']))
    
    print("\nTest case 2: All combinations up to 3 features")
    result_all = analyze_feature_combinations(dummy_data, 'target', feature_cols, max_features=3)
    print("Number of combinations:", len(result_all))
    print("Sorted by information:", result_all['Total_Information'].is_monotonic_decreasing)
    
    print("\nTest case 3: Mixed data types handling")
    mixed_features = ['feat1', 'feat3']  # Categorical and continuous
    result_mixed = analyze_feature_combinations(dummy_data, 'target', mixed_features)
    print("Successfully handled mixed types:", len(result_mixed) > 0)

def test_analyze_feature_importance():
    print("\nTesting analyze_feature_importance...")
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 500
    
    dummy_data = pd.DataFrame({
        'target': np.random.normal(0, 1, n_samples),
        'feat1': ['A', 'B'] * (n_samples // 2),  # Categorical
        'feat2': np.random.choice(['X', 'Y', 'Z'], n_samples),  # Categorical
        'feat3': np.random.uniform(0, 10, n_samples),  # Continuous
        'feat4': np.random.normal(5, 2, n_samples)  # Continuous
    })
    
    feature_cols = ['feat1', 'feat2', 'feat3', 'feat4']
    
    result = analyze_feature_importance(dummy_data, 'target', feature_cols)
    
    print("\nTest case 1: Basic functionality")
    print("Shape:", result.shape)
    print("Contains expected columns:", all(col in result.columns 
                                         for col in ['Feature', 'Mutual_Information']))
    
    print("\nTest case 2: Mixed data types")
    mixed_features = ['feat1', 'feat3']  # One categorical, one continuous
    result_mixed = analyze_feature_importance(dummy_data, 'target', mixed_features)
    print("Successfully handled mixed types:", len(result_mixed) > 0)
    
    print("\nTest case 3: Discretization")
    continuous_features = ['feat3', 'feat4']
    result_cont = analyze_feature_importance(dummy_data, 'target', continuous_features, num_bins=5)
    print("Proper discretization:", all('discretized' in f for f in result_cont['Feature']))

def test_plot_feature_combinations():
    print("\nTesting plot_feature_combinations...")
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 500
    
    dummy_data = pd.DataFrame({
        'target': np.random.normal(0, 1, n_samples),
        'feat1': ['A', 'B'] * (n_samples // 2),  # Categorical
        'feat2': np.random.choice(['X', 'Y', 'Z'], n_samples),  # Categorical
        'feat3': np.random.uniform(0, 10, n_samples),  # Continuous
        'feat4': np.random.normal(5, 2, n_samples)  # Continuous
    })
    
    feature_cols = ['feat1', 'feat2', 'feat3', 'feat4']
    
    print("\nTest case 1: Basic plotting functionality")
    fig = plot_feature_combinations(dummy_data, 'target', feature_cols[:3])
    print("Plot created:", fig is not None)
    
    print("\nTest case 2: Plot elements")
    ax = fig.axes[0]
    print("Has title:", bool(ax.get_title()))
    print("Has x-label:", bool(ax.get_xlabel()))
    print("Has y-label:", bool(ax.get_ylabel()))
    
    print("\nTest case 3: Custom figure size")
    fig_custom = plot_feature_combinations(dummy_data, 'target', feature_cols[:2], figsize=(12, 8))
    print("Custom size applied:", fig_custom.get_size_inches().tolist() == [12.0, 8.0])
    
    # Clean up plots
    plt.close('all')

def test_plot_multilevel_info():
    print("\nTesting plot_multilevel_info...")
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 500
    
    dummy_data = pd.DataFrame({
        'target': np.random.normal(0, 1, n_samples),
        'feat1': ['A', 'B'] * (n_samples // 2),  # Categorical
        'feat2': np.random.choice(['X', 'Y', 'Z'], n_samples),  # Categorical
        'feat3': np.random.uniform(0, 10, n_samples),  # Continuous
        'feat4': np.random.normal(5, 2, n_samples)  # Continuous
    })
    
    feature_cols = ['feat1', 'feat2', 'feat3']  # Using 3 features for clearer visualization
    
    print("\nTest case 1: Basic plotting functionality")
    fig = plot_multilevel_info(dummy_data, 'target', feature_cols)
    print("Plot created:", fig is not None)
    
    print("\nTest case 2: Plot elements")
    ax = fig.axes[0]
    print("Has title:", bool(ax.get_title()))
    print("Has x-label:", bool(ax.get_xlabel()))
    print("Has y-label:", bool(ax.get_ylabel()))
    
    print("\nTest case 3: Top N filtering")
    fig_top = plot_multilevel_info(dummy_data, 'target', feature_cols, top_n=2)
    print("Top N filtering applied:", len(fig_top.axes[0].get_xticklabels()) <= 2)
    
    print("\nTest case 4: Custom figure size")
    fig_custom = plot_multilevel_info(dummy_data, 'target', feature_cols, figsize=(12, 8))
    print("Custom size applied:", fig_custom.get_size_inches().tolist() == [12.0, 8.0])
    
    # Clean up plots
    plt.close('all')

def test_plot_feature_importance_by_level():
    print("\nTesting plot_feature_importance_by_level...")
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 500
    
    dummy_data = pd.DataFrame({
        'target': np.random.normal(0, 1, n_samples),
        'feat1': ['A', 'B'] * (n_samples // 2),  # Categorical
        'feat2': np.random.choice(['X', 'Y', 'Z'], n_samples),  # Categorical
        'feat3': np.random.uniform(0, 10, n_samples),  # Continuous
        'feat4': np.random.normal(5, 2, n_samples)  # Continuous
    })
    
    feature_cols = ['feat1', 'feat2', 'feat3']  # Using 3 features for clearer visualization
    
    print("\nTest case 1: Basic plotting functionality")
    fig = plot_feature_importance_by_level(dummy_data, 'target', feature_cols)
    print("Plot created:", fig is not None)
    
    print("\nTest case 2: Plot elements")
    ax = fig.axes[0]
    print("Has title:", bool(ax.get_title()))
    print("Has x-label:", bool(ax.get_xlabel()))
    print("Has y-label:", bool(ax.get_ylabel()))
    print("Has legend:", bool(ax.get_legend()))
    
    print("\nTest case 3: Data aggregation")
    # Get the plotted data
    plot_data = ax.containers[0].datavalues
    print("Has data points:", len(plot_data) > 0)
    print("Proper aggregation:", all(not np.isnan(x) for x in plot_data))
    
    print("\nTest case 4: Custom figure size")
    fig_custom = plot_feature_importance_by_level(dummy_data, 'target', feature_cols, figsize=(12, 8))
    print("Custom size applied:", fig_custom.get_size_inches().tolist() == [12.0, 8.0])
    
    # Clean up plots
    plt.close('all')

def test_plot_feature_importance_by_ordering():
    print("\nTesting plot_feature_importance_by_ordering...")
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 500
    
    dummy_data = pd.DataFrame({
        'target': np.random.normal(0, 1, n_samples),
        'feat1': ['A', 'B'] * (n_samples // 2),  # Categorical
        'feat2': np.random.choice(['X', 'Y', 'Z'], n_samples),  # Categorical
        'feat3': np.random.uniform(0, 10, n_samples),  # Continuous
        'feat4': np.random.normal(5, 2, n_samples)  # Continuous
    })
    
    feature_cols = ['feat1', 'feat2', 'feat3']  # Using 3 features for clearer visualization
    
    print("\nTest case 1: Basic plotting functionality")
    fig = plot_feature_importance_by_ordering(dummy_data, 'target', feature_cols)
    print("Plot created:", fig is not None)
    
    print("\nTest case 2: Plot elements")
    ax = fig.axes[0]
    print("Has title:", bool(ax.get_title()))
    print("Has x-label:", bool(ax.get_xlabel()))
    print("Has y-label:", bool(ax.get_ylabel()))
    print("Has legend:", bool(ax.get_legend()))
    
    print("\nTest case 3: Data aggregation")
    # Get the plotted data
    plot_data = ax.containers[0].datavalues
    print("Has data points:", len(plot_data) > 0)
    print("Proper aggregation:", all(not np.isnan(x) for x in plot_data))
    
    print("\nTest case 4: Custom figure size")
    fig_custom = plot_feature_importance_by_ordering(dummy_data, 'target', feature_cols, figsize=(12, 8))
    print("Custom size applied:", fig_custom.get_size_inches().tolist() == [12.0, 8.0])
    
    # Clean up plots
    plt.close('all')

def test_plot_feature_importance_by_level_and_ordering():
    print("\nTesting plot_feature_importance_by_level_and_ordering...")
    
    # Create dummy dataset
    np.random.seed(42)
    n_samples = 500
    
    dummy_data = pd.DataFrame({
        'target': np.random.normal(0, 1, n_samples),
        'feat1': ['A', 'B'] * (n_samples // 2),  # Categorical
        'feat2': np.random.choice(['X', 'Y', 'Z'], n_samples),  # Categorical
        'feat3': np.random.uniform(0, 10, n_samples),  # Continuous
        'feat4': np.random.normal(5, 2, n_samples)  # Continuous
    })
    
    feature_cols = ['feat1', 'feat2', 'feat3']  # Using 3 features for clearer visualization
    
    print("\nTest case 1: Basic plotting functionality")
    fig = plot_feature_importance_by_level_and_ordering(dummy_data, 'target', feature_cols)
    print("Plot created:", fig is not None)
    
    print("\nTest case 2: Plot elements")
    # Check facet grid elements
    print("Has multiple subplots:", len(fig.axes) > 1)
    print("Has title:", bool(fig._suptitle))
    
    # Check individual subplot elements
    ax = fig.axes[0]  # Check first subplot
    print("Has x-label:", bool(ax.get_xlabel()))
    print("Has y-label:", bool(ax.get_ylabel()))
    print("Has legend:", bool(ax.get_legend()))
    
    print("\nTest case 3: Data aggregation")
    # Get the plotted data from first subplot
    plot_data = ax.containers[0].datavalues
    print("Has data points:", len(plot_data) > 0)
    print("Proper aggregation:", all(not np.isnan(x) for x in plot_data))
    
    print("\nTest case 4: Custom figure size")
    fig_custom = plot_feature_importance_by_level_and_ordering(dummy_data, 'target', feature_cols, figsize=(12, 8))
    print("Custom size applied:", fig_custom.get_size_inches().tolist() == [12.0, 8.0])
    
    # Clean up plots
    plt.close('all')