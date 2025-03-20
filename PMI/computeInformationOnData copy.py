import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from visualizationCode import analyze_n_variable_decomposition
import pmi
from binning import create_variable_bins, create_all_variable_bins, apply_bin_mappings

# Load and preprocess the data
print("Loading and preprocessing data...")
filename = "small_data"
data = pd.read_csv("mock_data/"+filename+".csv")

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

def generate_nVD_plots(k=3):
    """
    Generate n-variable decomposition plots for all combinations of features.
    
    Args:
        k (int): Number of variables to decompose (default: 3)
    """
    print(f"\nAnalyzing {k}-variable decomposition for all combinations...")
    
    # Get all possible k-tuplets of features
    all_k_tuplets = list(combinations(feature_cols, k))
    print(f"Total number of combinations to analyze: {len(all_k_tuplets)}")
    
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Calculate grid dimensions
    n_combinations = len(all_k_tuplets)
    n_cols = 5  # We'll use 5 columns
    n_rows = (n_combinations + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and axes grid
    fig = plt.figure(figsize=(7*n_cols, 5*n_rows), facecolor='white')
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.2)  # Reduced spacing
    axes = gs.subplots()
    if n_rows == 1:
        axes = np.array([axes])  # Ensure axes is always 2D
    axes = axes.flatten()
    
    # Create a beautiful color palette - using seaborn's color palettes
    colors = sns.color_palette("husl", k)
    
    # Iterate through combinations and corresponding axes
    for combo_idx, (variables, ax) in enumerate(zip(all_k_tuplets, axes)):
        print(f"\nAnalyzing combination {combo_idx+1}/{len(all_k_tuplets)}: {', '.join(variables)}")
        
        # Set background color for this subplot
        ax.set_facecolor('#f8f9fa')
        
        # Create a color map for variables
        var_colors = {var: colors[j % len(colors)] for j, var in enumerate(variables)}
        
        # Analyze n-variable decomposition
        results_df = analyze_n_variable_decomposition(
            data,
            'income',
            *variables
        )
        
        # Implement tree-based plotting
        # First, plot each root variable (direct information)
        x_positions = {}
        group_spacing = 1.8  # Slightly increased spacing between variable groups
        
        for root_idx, root_var in enumerate(variables):
            # Position for this root variable
            x_pos = root_idx * group_spacing
            x_positions[root_var] = x_pos
            
            # Get the information for the root variable
            root_data = results_df[(results_df['Level'] == 1) & (results_df['Variable'] == root_var)]
            root_height = root_data['Information'].values[0]
            
            # Plot root variable
            bar = ax.bar(
                x_pos,
                root_height,
                bottom=0,
                color=var_colors[root_var],
                width=0.9,
                label=f'I(Y;{root_var})',
                edgecolor='white',
                linewidth=1.2,
                alpha=0.85
            )
            
            # Add value label if significant
            if root_height > 0.1:
                ax.text(
                    x_pos,
                    root_height/2,
                    f"I(Y;{root_var})\n{root_height:.2f}",
                     ha='center', va='center',
                    color='white', fontsize=9,
                    fontweight='bold'
                )
            
            # Store the stack height and width for children
            stack_height = root_height
            
            # Now plot children recursively for this root
            def plot_children(parent_var, parent_context, level, x_center, width, bottom):
                """
                Recursively plot children of a variable.
                
                Args:
                    parent_var: The parent variable
                    parent_context: The conditioning context (list of previous variables)
                    level: Current level in the tree
                    x_center: Center x position for parent
                    width: Width of the parent bar
                    bottom: Bottom position to start stacking
                """
                if level > len(variables):
                    return
                
                # Find all variables that can appear at this level (those not in parent_context)
                remaining_vars = [v for v in variables if v not in parent_context]
                
                if not remaining_vars:
                    return
                
                # Calculate width for each child
                child_width = width / len(remaining_vars)
                
                # Special case: If there's only one remaining variable, keep the parent's width
                if len(remaining_vars) == 1:
                    child_width = width
                    
                # Calculate starting position for first child
                start_x = x_center - (width / 2) + (child_width / 2)
                
                # Plot each child
                for i, child_var in enumerate(remaining_vars):
                    # Calculate position for this child
                    child_x = start_x + (i * child_width)
                    
                    # Get data for this child and context
                    child_data = results_df[
                        (results_df['Level'] == level) & 
                        (results_df['Variable'] == child_var)
                    ]
                    
                    # Filter for the correct ordering context
                    # We need to find rows where the ordering starts with our parent context
                    if len(child_data) > 0:
                        filtered_data = []
                        
                        for _, row in child_data.iterrows():
                            ordering_parts = row['Ordering'].split(" → ")
                            
                            # Check if our parent context matches the start of this ordering
                            matches_context = False
                            if len(ordering_parts) >= len(parent_context):
                                matches_context = ordering_parts[:len(parent_context)] == parent_context
                            
                            # And the next element is our child_var
                            next_is_child = False
                            if len(ordering_parts) > len(parent_context):
                                next_is_child = ordering_parts[len(parent_context)] == child_var
                            
                            if matches_context and next_is_child:
                                filtered_data.append(row)
                        
                        if filtered_data:
                            child_data = pd.DataFrame(filtered_data)
                        else:
                            child_data = pd.DataFrame()
                    
                    if len(child_data) > 0:
                        child_height = child_data['Information'].values[0]
                        
                        # Calculate fade factor for aesthetic gradient effect
                        fade_factor = 1.0 - (0.1 * (level - 1))
                        
                        # Get base color and adjust alpha
                        child_color = var_colors[child_var]
                        
                        # Plot child
                        bar = ax.bar(
                            child_x,
                            child_height,
                            bottom=bottom,
                            color=child_color,
                            width=child_width,
                            label=f'I(Y;{child_var}|{",".join(parent_context)})',
                            edgecolor='white',
                            linewidth=0.8,
                            alpha=fade_factor * 0.85
                        )
                        
                        # If height is significant, add a label
                        if child_height > 0.1:
                            font_size = min(9, max(7, 8 - (level-1)))
                            ax.text(
                                child_x,
                                bottom + child_height/2,
                                f"I(Y;{child_var}|{','.join(parent_context)})\n{child_height:.2f}",
                                ha='center', va='center',
                                color='white', fontsize=font_size,
                                fontweight='bold'
                            )
                        
                        # Recursively plot this child's children
                        new_context = parent_context + [child_var]
                        plot_children(
                            child_var,
                            new_context,
                            level + 1,
                            child_x,
                            child_width,
                            bottom + child_height
                        )
            
            # Start recursive plotting for this root variable
            plot_children(root_var, [root_var], 2, x_pos, 0.9, stack_height)
        
        # Customize subplot
        ax.set_ylabel('Mutual Information (bits)', fontsize=12, fontweight='bold')
        ax.set_title(', '.join(variables), fontsize=13, fontweight='bold', pad=15)
        
        # Remove x-ticks and labels (redundant with the legend)
        ax.set_xticks([])
        
        # Improve grid and spines
        ax.yaxis.grid(True, alpha=0.3, color='gray', linestyle='-')
        ax.set_axisbelow(True)
        
        # Make y-axis labels more readable
        ax.tick_params(axis='y', labelsize=10)
        
        # Set reasonable x-axis limits
        min_x = min(x_positions.values()) - 1
        max_x = max(x_positions.values()) + 1
        ax.set_xlim(min_x, max_x)
        
        # Create legend with improved styling
        handles = [plt.Rectangle((0,0),1,1, facecolor=var_colors[var], alpha=0.85, 
                               edgecolor='white', linewidth=1) 
                  for var in variables]
        
        # Place legend at the bottom outside the axes, in a horizontal orientation
        ax.legend(handles, variables, fontsize=8, 
                 loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 frameon=True, framealpha=0.9, edgecolor='lightgray',
                 title=None, ncol=len(variables),
                 borderpad=0.4, labelspacing=0.5,
                 handlelength=1.0, handletextpad=0.4,
                 columnspacing=1.0)
    
    # Hide any unused subplots
    for ax in axes[len(all_k_tuplets):]:
        ax.set_visible(False)
    
    # Adjust layout - use subplots_adjust instead of tight_layout to accommodate the legend outside the axes
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
    plt.savefig(f'results/all_{k}VD_combinations.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'results/all_{k}VD_combinations.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nAnalysis complete. Results saved in 'results/all_{k}VD_combinations.pdf' and .png")

def generate_conditional_nVD_plots(conditioning_var, *variables, k=3):
    """
    Generate n-variable decomposition plots conditioned on bins of specified variables.
    
    Args:
        conditioning_var (str): First variable to condition on (will be binned)
        *variables: Variables to use for conditioning in each subplot
        k (int): Number of variables to decompose (default: 3)
    """
    print(f"\nAnalyzing {k}-variable decomposition conditioned on multiple variables...")
    
    # Create bins for all variables that will be used for conditioning
    all_conditioning_vars = [conditioning_var] + list(variables)
    bin_mappings = {}
    descriptive_labels = {}
    
    for var in all_conditioning_vars:
        bin_mapping, desc_labels = create_variable_bins(data, var, n_bins=5)
        bin_mappings[var] = bin_mapping
        descriptive_labels[var] = desc_labels
    
    # Set up the information trees for each conditioning variable
    var_info_trees = {}
    
    # For each variable, build its information tree for each bin
    for var_idx, var in enumerate(all_conditioning_vars):
        print(f"\nAnalyzing conditioning variable: {var}")
        
        # Get unique bins for this variable
        binned_data = apply_bin_mappings(data, {var: bin_mappings[var]})
        unique_bins = binned_data[f"{var}_binned"].unique()
        n_bins = len(unique_bins)
        print(f"Created {n_bins} bins for {var}")
        
        # Calculate entropy of target variable for the full dataset
        from pmi import entropy
        full_entropy = entropy(data, 'income')
        
        # Build information trees for each bin of this variable
        info_trees = {}
        
        for bin_val in unique_bins:
            # Filter data for this bin
            bin_data = data[data[var].map(bin_mappings[var]) == bin_val]
            n_samples = len(bin_data)
            print(f"  Analyzing bin {bin_val}: {descriptive_labels[var][bin_val]} with {n_samples} samples")
            
            # Create tree structure for this bin
            bin_tree = {}
            
            # First level: I(Y;A=bin) = H(Y) - H(Y|A=bin)
            bin_entropy = entropy(bin_data, 'income')
            first_level_info = full_entropy - bin_entropy
            bin_tree['root'] = {
                'info': first_level_info,
                'label': f"{var}={descriptive_labels[var][bin_val]}",
                'children': {}
            }
            
            # Get other variables for this tree (excluding the current conditioning variable)
            other_vars = [v for v in all_conditioning_vars if v != var]
            
            # Build the tree recursively for higher order information
            def build_info_tree(parent_node, parent_context, level, remaining_vars):
                """Recursively build the information tree for this bin."""
                if level > k or not remaining_vars:
                    return
                
                for other_var in remaining_vars:
                    # Calculate conditional mutual information for this variable
                    if level == 1:
                        # Direct mutual information between target and this variable for this bin
                        mi = pmi.mutual_information(bin_data, 'income', other_var)
                    else:
                        # Conditional mutual information
                        mi = pmi.conditional_mutual_information(
                            bin_data, 
                            'income',
                            parent_context,  # Already conditioned on parent variables
                            [other_var]      # Conditioning on this variable
                        )
                    
                    # Add to tree if significant
                    if mi > 0.001:
                        parent_node['children'][other_var] = {
                            'info': mi,
                            'label': f"I(Y;{other_var}|{','.join(parent_context)})",
                            'children': {}
                        }
                        
                        # Recurse for this variable's children
                        new_parent_context = parent_context + [other_var]
                        new_remaining_vars = [v for v in remaining_vars if v != other_var]
                        build_info_tree(
                            parent_node['children'][other_var],
                            new_parent_context,
                            level + 1,
                            new_remaining_vars
                        )
            
            # Start recursive tree building for this bin
            build_info_tree(bin_tree['root'], [], 1, other_vars)
            
            # Store the completed tree for this bin
            info_trees[bin_val] = bin_tree
        
        # Store all trees for this conditioning variable
        var_info_trees[var] = info_trees
    
    # 2. Visualize the information trees
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Create figure - one subplot per conditioning variable
    n_vars = len(all_conditioning_vars)
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = fig.add_gridspec(1, n_vars, hspace=0.3, wspace=0.5)
    axes = gs.subplots()
    if n_vars == 1:
        axes = np.array([axes])  # Ensure axes is always array-like
    
    # Create color palettes for variables
    all_var_colors = {}
    color_palettes = ["Blues", "Reds", "Greens", "Purples", "Oranges"]
    for i, var in enumerate(all_conditioning_vars):
        palette_idx = i % len(color_palettes)
        all_var_colors[var] = sns.color_palette(color_palettes[palette_idx], 3)[1]  # Use middle color from each palette
    
    # Plot each conditioning variable in a separate subplot
    for var_idx, var in enumerate(all_conditioning_vars):
        ax = axes[var_idx]
        ax.set_facecolor('#f8f9fa')
        
        # Get the trees for this conditioning variable
        info_trees = var_info_trees[var]
        unique_bins = list(info_trees.keys())
        n_bins = len(unique_bins)
        
        # Create color palette for bins
        bin_colors = sns.color_palette("husl", n_bins)
        
        # Set up position for bin groups
        group_spacing = 1.0  # Space between bin groups
        bar_width = 0.8  # Width of the bars
        x_positions = {}
        
        # Plot each bin in this subplot
        for bin_idx, bin_val in enumerate(unique_bins):
            # Get the tree for this bin
            tree = info_trees[bin_val]
            
            # Position for this bin
            x_pos = bin_idx * group_spacing
            x_positions[bin_val] = x_pos
            
            # Plot the root level (conditioning variable information)
            root_height = tree['root']['info']
            ax.bar(
                x_pos,
                root_height,
                width=bar_width,
                color=bin_colors[bin_idx],
                edgecolor='white',
                linewidth=1.2,
                alpha=0.85,
                label=f"{tree['root']['label']}" if var_idx == 0 else ""
            )
            
            # Add label for the root if significant
            if root_height > 0.1:
                ax.text(
                    x_pos,
                    root_height/2,
                    f"{root_height:.2f}",
                    ha='center', va='center',
                    color='white', fontsize=8,
                    fontweight='bold'
                )
            
            # Store height for children
            stack_height = root_height
            
            # Define recursive function to plot children
            def plot_children(node, parent_vars, level, x_center, width, bottom):
                """Recursively plot children of the tree."""
                if level > k:
                    return
                
                # Find variables that can appear at this level (not in parent_vars)
                remaining_vars = [v for v in all_conditioning_vars if v != var and v not in parent_vars]
                
                if not remaining_vars:
                    return
                
                # Calculate width for each child
                child_width = width / len(remaining_vars)
                
                # If only one remaining variable, keep parent's width
                if len(remaining_vars) == 1:
                    child_width = width
                
                # Calculate starting position for first child
                start_x = x_center - (width/2) + (child_width/2)
                
                # Plot each child
                for i, child_var in enumerate(remaining_vars):
                    # Skip if not in children dictionary
                    if child_var not in node['children']:
                        continue
                    
                    # Calculate position for this child
                    child_x = start_x + (i * child_width)
                    
                    # Get information for this child
                    child_info = node['children'][child_var]['info']
                    
                    # Calculate fade factor for aesthetics
                    fade_factor = 1.0 - (0.1 * (level - 1))
                    
                    # Get color for this child
                    child_color = all_var_colors[child_var]
                    
                    # Plot child
                    ax.bar(
                        child_x,
                        child_info,
                        bottom=bottom,
                        color=child_color,
                        width=child_width,
                        edgecolor='white',
                        linewidth=0.8,
                        alpha=fade_factor * 0.85
                    )
                    
                    # Add label if significant
                    if child_info > 0.1:
                        font_size = min(8, max(6, 7 - (level-1)))
                        ax.text(
                            child_x,
                            bottom + child_info/2,
                            f"{child_info:.2f}",
                            ha='center', va='center',
                            color='white', fontsize=font_size,
                            fontweight='bold'
                        )
                    
                    # Recursively plot this child's children
                    new_parents = parent_vars + [child_var]
                    plot_children(
                        node['children'][child_var],
                        new_parents,
                        level + 1,
                        child_x,
                        child_width,
                        bottom + child_info
                    )
            
            # Start recursive plotting for this bin
            plot_children(tree['root'], [], 1, x_pos, bar_width, stack_height)
        
        # Customize subplot
        ax.set_ylabel('Mutual Information (bits)', fontsize=12, fontweight='bold')
        ax.set_title(f'Conditioned on {var}', fontsize=13, fontweight='bold', pad=15)
        
        # Add x-ticks and labels
        ax.set_xticks([x for x in x_positions.values()])
        ax.set_xticklabels([descriptive_labels[var][bin_val] for bin_val in unique_bins], rotation=45, fontsize=8, ha='right')
        
        # Improve grid and spines
        ax.yaxis.grid(True, alpha=0.3, color='gray', linestyle='-')
        ax.set_axisbelow(True)
        
        # Make y-axis labels more readable
        ax.tick_params(axis='y', labelsize=10)
        
        # Set reasonable x-axis limits
        min_x = min(x_positions.values()) - 0.5
        max_x = max(x_positions.values()) + 0.5
        ax.set_xlim(min_x, max_x)
    
    # Create legend for variable colors
    var_handles = [plt.Rectangle((0,0),1,1, facecolor=all_var_colors[var], alpha=0.85, 
                              edgecolor='white', linewidth=1) 
                 for var in all_conditioning_vars]
    var_labels = [f"Variable: {var}" for var in all_conditioning_vars]
    
    # Place legend at the bottom
    fig.legend(var_handles, var_labels, fontsize=10, 
              loc='lower center', bbox_to_anchor=(0.5, 0.01),
              frameon=True, framealpha=0.9, edgecolor='lightgray',
              ncol=len(var_handles), borderpad=0.4, labelspacing=0.5,
              handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.2)
    plt.suptitle(f'Information Decomposition Conditioned on Different Variables', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figures
    plt.savefig(f'results/conditional_{k}VD_multiple.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'results/conditional_{k}VD_multiple.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nAnalysis complete. Results saved in 'results/conditional_{k}VD_multiple.pdf' and .png")

def plotInformationLandscape():
    """
    Plot the information landscape showing:
    1. A horizontal line for the entropy of income distribution
    2. Scattered points for mutual information of all variable combinations
    """
    print("\nAnalyzing information landscape...")
    
    # Calculate entropy of income
    income_entropy = pmi.entropy(data, 'income')
    print(f"Income entropy: {income_entropy:.3f} bits")
    
    # Prepare to collect mutual information values
    info_values = []  # Will store (n_vars, mutual_info) pairs
    
    # For each possible cardinality
    max_vars = min(8, len(feature_cols))
    for k in range(1, max_vars + 1):
        print(f"\nAnalyzing {k}-variable combinations...")
        
        # Get all possible k-combinations of features
        for variables in combinations(feature_cols, k):
            # Calculate mutual information directly
            mi = pmi.mutual_information(data, 'income', *variables)
            info_values.append((k, mi))
            print(f"MI({', '.join(variables)}) = {mi:.3f} bits")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot entropy line
    plt.axhline(y=income_entropy, color='r', linestyle='--', alpha=0.5, 
                label=f'H(income) = {income_entropy:.3f}')
    
    # Plot mutual information points
    x_vals, y_vals = zip(*info_values)
    plt.scatter(x_vals, y_vals, alpha=0.6, c='blue')
    
    # Set plot bounds and labels
    plt.xlim(0.5, max_vars + 0.5)
    plt.ylim(0, income_entropy + 1)
    plt.xlabel('Number of Variables', fontsize=12)
    plt.ylabel('Information (bits)', fontsize=12)
    plt.title('Information Landscape: Variable Combinations vs. Income Entropy', 
              fontsize=14, pad=20)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig('results/information_landscape.pdf', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\nAnalysis complete. Results saved in 'results/information_landscape_"+filename+".pdf'")

if __name__ == "__main__":
    # Test binning functionality
    print("\nTesting variable binning...")
    
    # Create bins for all feature columns
    bin_mappings, descriptive_labels = create_all_variable_bins(data, feature_cols, n_bins=5)
    
    # Print detailed bin information for each variable
    for var in feature_cols:
        print(f"\n{'='*50}")
        print(f"Variable: {var}")
        print(f"Data type: {data[var].dtype}")
        print(f"Number of unique values: {len(data[var].unique())}")
        
        # Get value counts for this variable
        value_counts = data[var].value_counts()
        
        # Get the mapping for this variable
        mapping = bin_mappings[var]
        
        # Create reverse mapping (bin -> list of values)
        bin_contents = {}
        for val, bin_label in mapping.items():
            if bin_label not in bin_contents:
                bin_contents[bin_label] = []
            bin_contents[bin_label].append(val)
        
        print(f"\nNumber of bins created: {len(bin_contents)}")
        print("\nBin contents:")
        for bin_label, values in bin_contents.items():
            # Calculate how many data points fall into this bin
            count = sum(value_counts[val] for val in values if val in value_counts)
            print(f"\nBin: {bin_label}")
            print(f"Count: {count} data points")
            print("Values:", end=" ")
            if len(values) > 10:
                print(f"{values[:10]} ... ({len(values)} total values)")
            else:
                print(values)
    
    # Apply binning to create binned dataset
    binned_data = apply_bin_mappings(data, bin_mappings)
    
    # Test the new conditional plotting function with multiple conditioning variables
    print("\nTesting conditional nVD plots...")
    # Each subplot will condition on a different variable
    generate_conditional_nVD_plots('education', 'profession_4', 'gender', 'city', k=3)
    
    # Uncomment to continue with existing analysis
    # for k in [3,4,5,6,7]:
    #     print(f"\nTesting with k={k} variables...")
    #     generate_nVD_plots(k=k)
    # plotInformationLandscape()

