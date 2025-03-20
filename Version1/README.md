# Income Predictors Analysis using Mutual Information

## Overview
This R script performs a comprehensive mutual information analysis to understand which factors best predict income levels. It uses information theory concepts to quantify the predictive power of various demographic and professional variables.

## Dependencies
The script requires the following R packages:
- entropy (for entropy calculations)
- dplyr (for data manipulation)
- ggplot2 (for visualization)
- tidyr (for data reshaping)
- data.table (for efficient data handling)
- gtools (for permutations)

## Key Functions

### 1. calculate_mutual_info(x, y)
Calculates the mutual information between two variables using:
- Joint probability tables
- Marginal probabilities
- Log2-based mutual information calculation

### 2. calculate_cond_mutual_info(x, y, z)
Computes conditional mutual information I(X;Y|Z) by:
- Iterating through unique values of the conditioning variable
- Calculating mutual information for each condition
- Weighting by probability of each condition

### 3. discretize_variable(x, num_bins = 100)
Converts continuous variables into discrete bins using:
- Quantile-based binning
- Configurable number of bins

### 4. multilevel_info_decomposition(data, target_col, feature_cols, ...)
Main analysis function that:
- Handles both continuous and categorical variables
- Computes information decomposition for different feature orderings
- Generates visualizations of results
- Returns detailed statistics and plotting functions

### 5. analyze_feature_combinations(data, target_col, feature_cols, max_features)
Analyzes different feature subset combinations to find optimal predictors by:
- Testing various feature combinations
- Computing total information for each combination
- Ranking feature sets by predictive power

## Data Processing
The script processes the following key variables:
- Income (total_income)
- Gender
- Profession (at different levels of detail)
- Education level (Sun2000niva)
- City of residence
- University ID

## Analysis Workflow
1. Data preprocessing and cleaning
2. Feature discretization where needed
3. Multiple levels of analysis:
   - 3-feature combinations
   - 4-feature combinations
   - 5-feature combinations
4. Generation of visualizations showing:
   - Feature ordering importance
   - Top feature combinations
   - Information contribution of each feature

## Output
The script generates several PDF visualizations:
- feature_combinations.pdf (overall feature combinations)
- feature_combinationsvars_3.pdf (3-variable combinations)
- feature_combinationsvars_4.pdf (4-variable combinations)
- feature_combinationsvars_5.pdf (5-variable combinations)

## Usage Notes
- The script is configured for a specific cohort (1982) and year (2022)
- Only positive income values are included in the analysis
- Missing values are excluded
- The analysis can be computationally intensive for large feature sets

## Limitations
- Full permutation analysis is limited to 6 features to avoid computational explosion
- For larger feature sets, only a subset of permutations is analyzed
- Results depend on the quality of discretization for continuous variables 