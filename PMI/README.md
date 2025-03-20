# Information Theory Calculations: PMI and LMI Implementation

This directory contains implementations of Pointwise Mutual Information (PMI) and Local Mutual Information (LMI) calculations, along with their conditional variants.

## Mathematical Foundation

### Pointwise Mutual Information (PMI)
PMI measures the association between a raw outcome and when conditioned on variables:
\[i(y;a) = \ln \frac{P(y|a)}{P(y)}\]

For multiple variables:
\[i(y;a,b,c,...) = \ln \frac{P(y|a,b,c,...)}{P(y)}\]

### Conditional PMI
Measures the association when further conditioning:
\[i(y;b,c|a) = \ln \frac{P(y|a,b,c)}{P(y|a)}\]

Where a represents base conditions and b,c represent additional conditions.

### Local Mutual Information (LMI)
Averages PMI over the target variable Y:
\[i(Y;a,b,c,...) = \sum_y P(y|a,b,c,...)i(y;a,b,c,...)\]

### Conditional LMI
\[i(Y;b,c|a) = \sum_y P(y|a,b,c)i(y;b,c|a)\]

## Implementation Details

The code in `pmi.py` implements these calculations with the following structure:

### Core Probability Functions
- `get_probability(data, target)`: Calculates P(y)
- `get_conditional_probability(data, target, conditions)`: Calculates P(y|conditions)

### Information Measures
1. **PMI Calculations**
   - `pointwise_mutual_information(data, target, *conditions)`: Calculates i(y;a,b,...)
   - `conditional_pointwise_mutual_information(data, target, base_conditions, conditioned_conditions)`: Calculates i(y;b,c|a)

2. **LMI Calculations**
   - `local_mutual_information(data, target, *conditions)`: Implements i(Y;a,b,...)
   - `conditional_local_mutual_information(data, target, base_conditions, conditioned_conditions)`: Implements i(Y;b,c|a)

### Key Features
- Handles arbitrary number of conditioning variables
- Separates base conditions from additional conditions in conditional measures
- Uses pandas MultiIndex for efficient computation
- Implements vectorized operations where possible
- Provides clear separation between probability calculations and information measures

## Usage

```python
# Example usage with multiple variables
from pmi import pointwise_mutual_information, local_mutual_information, conditional_pointwise_mutual_information, conditional_local_mutual_information

# Calculate PMI with multiple conditions
pmi_vals = pointwise_mutual_information(data, "Y", "a", "b", "c")

# Calculate LMI with multiple conditions
lmi_vals = local_mutual_information(data, "Y", "a", "b", "c")

# Calculate conditional variants
# Base condition 'a', additional conditions 'b' and 'c'
cpmi_vals = conditional_pointwise_mutual_information(
    data, 
    target="Y", 
    base_conditions=["a"], 
    conditioned_conditions=["b", "c"]
)

# Calculate conditional LMI
clmi_vals = conditional_local_mutual_information(
    data, 
    target="Y", 
    base_conditions=["a"], 
    conditioned_conditions=["b", "c"]
)
```

## Implementation Notes
- All functions return pandas Series with appropriate MultiIndex structure
- Handles edge cases (zero probabilities, missing values)
- Uses efficient pandas operations for grouping and aggregation
- Base conditions and additional conditions are clearly separated in conditional calculations
- Supports arbitrary number of variables in both joint and conditional calculations

## Mathematical Interpretation

In the context of data analysis:
- PMI shows the association between specific outcomes and multiple conditions
- LMI provides the average information content for combinations of variables
- Conditional variants help understand incremental information gain from new variables given existing ones
- These measures can help identify important feature combinations and hierarchical relationships in the data 