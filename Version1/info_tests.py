import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from itertools import permutations, combinations
import mutualinfo as mi

#===============================
# SECTION 3: DATA LOADING AND ANALYSIS
#===============================
# Load the mock data
data = pd.read_csv("mock_data/small_data.csv")

# Create different levels of profession detail
data['profession'] = data['profession'].astype(str)
data['profession_3'] = data['profession'].str[:3]
data['profession_2'] = data['profession'].str[:2]
data['profession_1'] = data['profession'].str[:1]
data = data.rename(columns={'profession': 'profession_4'})

# Create gender variables
data['female'] = (data['gender'] == 'f').astype(int)
data['Gender'] = data['gender'].map({'m': 'Male', 'f': 'Female'})

# Process income variables
data['income_log'] = np.log(data['income'] + 1)
data['income'] = pd.to_numeric(data['income'])

# Print data validation
print("Data validation:")
print(f"Number of records: {len(data)}")
print(f"Missing values in income: {data['income'].isna().sum()}")
print(f"Missing values in gender: {data['gender'].isna().sum()}")

# Define feature columns for analysis
feature_cols = ['profession_4', 'gender', 'education', 'city', 'uni_id', 'ethnicity', 'age']

# # Perform mutual information analysis
# print("\nAnalyzing feature importance...")
# importance_results = mi.analyze_feature_importance(data, 'income', feature_cols)
# print("\nFeature Importance Results:")
# print(importance_results)

# # Plot feature importance
# plt.figure(figsize=(10, 6))
# mi.plot_feature_importance(data, 'income', feature_cols)
# plt.savefig('feature_importance.pdf')
# plt.close()

# # Analyze feature combinations
# print("\nAnalyzing feature combinations...")
# combinations_results = mi.analyze_feature_combinations(data, 'income', feature_cols[:4])
# print("\nFeature Combinations Results:")
# print(combinations_results)

# # Plot feature combinations
# plt.figure(figsize=(12, 6))
# mi.plot_feature_combinations(data, 'income', feature_cols[:4])
# plt.savefig('feature_combinations.pdf')
# plt.close()

def generate_3VD_plots():
# Perform three-variable decomposition analysis for all combinations
    print("\nAnalyzing three-variable decomposition for all combinations...")
    all_triplets = list(combinations(feature_cols, 3))
    print(f"Total number of combinations to analyze: {len(all_triplets)}")

    # Calculate number of rows and columns needed for subplots
    n_combinations = len(all_triplets)
    n_cols = 5  # We'll use 5 columns
    n_rows = (n_combinations + n_cols - 1) // n_cols  # Ceiling division

    # Create figure and axes grid first
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 20))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Iterate through combinations and corresponding axes
    for i, ((var1, var2, var3), ax) in enumerate(zip(all_triplets, axes)):
        print(f"\nAnalyzing combination {i+1}/{len(all_triplets)}: {var1}, {var2}, {var3}")
        results_df = mi.analyze_three_variable_decomposition(
            data,
            'income',
            var1,
            var2,
            var3
        )
        print(f"\nResults for {var1}, {var2}, {var3}:")
        
        # Create a color map for variables using Set3 palette
        var_colors = {
                var1: '#8DD3C7',  # Soft teal
                var2: '#FFFFB3',  # Light yellow
                var3: '#BEBADA'   # Soft purple
            }

        # For each ordering, plot bars in the correct stacking order
        orderings = results_df['Ordering'].unique()
        x = np.arange(len(orderings))
        
        # Plot each ordering
        for i, ordering in enumerate(orderings):
            ordering_data = results_df[results_df['Ordering'] == ordering]
            bottom = 0
            
            # Get the variables in their order of appearance
            variables_in_order = ordering.split(" → ")

            # Plot each variable's contribution
            for var in variables_in_order:
                height = ordering_data[ordering_data['Variable'] == var]['Information'].values[0]
                ax.bar(
                    i,
                    height,
                    bottom=bottom,
                    color=var_colors[var],
                    label=str(var)
                )
                bottom += height
            
        
        # ax.set_xticks(x, ["" for _ in orderings], rotation=45, ha='right', fontsize=14)
        # ax.set_yticks(fontsize=14)
        # ax.set_title(f'Info Dec: {var1}, {var2}, {var3}', fontsize=18)
        # ax.set_xlabel('Combinations', fontsize=18)
        ax.set_ylabel('Information (bits)', fontsize=18)
        
        # Get legend handles and labels from the specific axis
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)




    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined figure
    plt.savefig('results/all_3VD_combinations1.pdf', bbox_inches='tight')
    plt.close()
    print("\nAnalysis complete. All combinations have been plotted in 'all_3VD_combinations.pdf'")

generate_3VD_plots()