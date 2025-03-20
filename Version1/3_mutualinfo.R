# Load necessary libraries
library(entropy)# For entropy calculations
library(dplyr) # For data manipulation
library(ggplot2) # For visualization
library(tidyr) # For data reshaping
library(data.table) # For data reshaping



setwd("//micro.intra/Projekt/P0515$/P0515_Gem/Laura/")

load(file="Distributions/data/subset.rda")

data=as.data.table(data)




# different details of profession
data[, profession_3:=substr(profession, 1,3)]
data[, profession_2:=substr(profession, 1,2)]
data[, profession_1:=substr(profession, 1,1)]
setnames(data, 'profession', 'profession_4')


data[Kon==1, female:=0]
data[Kon==2, female:=1]

data[Kon==1, Gender:='Male']
data[Kon==2, Gender:='Female']



nrow(data[is.na(total_income)])
nrow(data[is.na(earned_income)])
nrow(data[is.na(disp_income)])
nrow(data[is.na(female)])


data[,income:=as.numeric(total_income)]

# Ensure it's a data.table
data <- as.data.table(data)


# Log income for analysis
data[, income_log := log(total_income+ 1)] # +1 to handle zeros

# Function to calculate mutual information between two variables
calculate_mutual_info <- function(x, y) {
  # Create joint probability table
  joint_prob <- table(x, y) / length(x)
  
  # Calculate marginal probabilities
  p_x <- rowSums(joint_prob)
  p_y <- colSums(joint_prob)
  
  # Calculate mutual information
  mi <- 0
  for(i in 1:nrow(joint_prob)) {
    for(j in 1:ncol(joint_prob)) {
      if(joint_prob[i,j] > 0) {
        mi <- mi + joint_prob[i,j] * log2(joint_prob[i,j] / (p_x[i] * p_y[j]))
      }
    }
  }
  
  return(mi)
}

# Function to calculate conditional mutual information: I(X;Y|Z)
calculate_cond_mutual_info <- function(x, y, z) {
  # Initialize
  cond_mi <- 0
  
  # For each value of the conditioning variable
  for(z_val in unique(z)) {
    # Filter for this value
    idx <- which(z == z_val)
    if(length(idx) <= 1) {
      next
    }
    
    # Calculate P(Z = z)
    p_z <- sum(z == z_val) / length(z)
    
    # Calculate I(X;Y|Z=z)
    x_given_z <- x[idx]
    y_given_z <- y[idx]
    
    # Skip if not enough variation
    if(length(unique(x_given_z)) <= 1 || length(unique(y_given_z)) <= 1) {
      next
    }
    
    mi_given_z <- calculate_mutual_info(x_given_z, y_given_z)
    
    # Add to weighted sum
    cond_mi <- cond_mi + p_z * mi_given_z
  }
  
  return(cond_mi)
}

# Function to discretize continuous variables
discretize_variable <- function(x, num_bins = 100) {
  if(is.numeric(x)) {
    x_binned <- cut(x, 
                    breaks = quantile(x, probs = seq(0, 1, 1/num_bins), na.rm = TRUE),
                    labels = paste0("B", 1:num_bins),
                    include.lowest = TRUE)
    return(x_binned)
  } else {
    return(x)
  }
}

# Generalized multilevel information decomposition function
multilevel_info_decomposition <- function(data, target_col, feature_cols, discretize = TRUE, num_bins = 10) {
  # Make a copy of the data
  df <- data.frame(data)
  
  # Discretize target variable if needed
  if(discretize && is.numeric(df[[target_col]])) {
    df$target_discretized <- discretize_variable(df[[target_col]], num_bins)
    target_var <- "target_discretized"
  } else {
    target_var <- target_col
  }
  
  # Discretize feature variables if needed
  if(discretize) {
    for(col in feature_cols) {
      if(is.numeric(df[[col]])) {
        df[[paste0(col, "_discretized")]] <- discretize_variable(df[[col]], num_bins)
        feature_cols[feature_cols == col] <- paste0(col, "_discretized")
      }
    }
  }
  
  # Number of features
  n_features <- length(feature_cols)
  
  # Create all possible permutations of feature ordering
  feature_orders <- list()
  if(n_features <= 6) { # Limit full permutation to 6 features to avoid explosion
    feature_orders <- as.list(data.frame(t(permutations(n_features, n_features, feature_cols))))
  } else {
    # Just use the original order and a few random ones
    feature_orders[[1]] <- feature_cols
    set.seed(123)
    for(i in 2:5) {
      feature_orders[[i]] <- sample(feature_cols)
    }
  }
  
  # Store results
  results <- list()
  
  # For each feature ordering
  for(order_idx in 1:length(feature_orders)) {
    feature_order <- feature_orders[[order_idx]]
    
    # Create a name for this ordering
    order_name <- paste(feature_order, collapse = " → ")
    
    # Initialize information components
    info_components <- numeric(n_features)
    names(info_components) <- feature_order
    
    # Calculate first level information: I(Y; X1)
    info_components[1] <- calculate_mutual_info(df[[target_var]], df[[feature_order[1]]])
    
    # Calculate each subsequent level: I(Y; Xi | X1, X2, ..., X_(i-1))
    if(n_features > 1) {
      for(i in 2:n_features) {
        # Variables we're conditioning on
        cond_vars <- feature_order[1:(i-1)]
        
        # Create a combined conditioning variable
        df$combined_cond <- apply(df[, cond_vars, drop = FALSE], 1, function(x) paste(x, collapse = "|"))
        
        # Calculate conditional information
        info_components[i] <- calculate_cond_mutual_info(
          df[[target_var]], df[[feature_order[i]]], df$combined_cond
        )
      }
    }
    
    # Calculate total information
    total_info <- sum(info_components)
    
    # Calculate proportions
    info_proportions <- info_components / total_info * 100
    
    # Store results for this ordering
    results[[order_name]] <- list(
      order = feature_order,
      components = info_components,
      total = total_info,
      proportions = info_proportions
    )
  }
  
  # Create a summary data frame for all orderings
  summary_df <- data.frame()
  
  for(order_name in names(results)) {
    order_results <- results[[order_name]]
    
    for(i in 1:length(order_results$components)) {
      feature <- order_results$order[i]
      
      if(i == 1) {
        level_name <- paste("I(", target_col, ";", feature, ")")
      } else {
        cond_vars <- paste(order_results$order[1:(i-1)], collapse = ",")
        level_name <- paste("I(", target_col, ";", feature, "|", cond_vars, ")")
      }
      
      summary_df <- rbind(summary_df, data.frame(
        Ordering = order_name,
        Level = i,
        Feature = feature,
        Description = level_name,
        Information = order_results$components[i],
        Proportion = order_results$proportions[i]
      ))
    }
  }
  
  # Return results
  return(list(
    detailed_results = results,
    summary = summary_df,
    
    # Function to plot results
    plot = function(top_n = NULL) {
      plot_data <- summary_df
      
      # Limit to top_n orderings if specified
      if(!is.null(top_n) && top_n < length(unique(plot_data$Ordering))) {
        # Calculate total information for each ordering
        ordering_totals <- aggregate(Information ~ Ordering, data = plot_data, FUN = sum)
        ordering_totals <- ordering_totals[order(ordering_totals$Information, decreasing = TRUE), ]
        top_orderings <- ordering_totals$Ordering[1:top_n]
        plot_data <- plot_data[plot_data$Ordering %in% top_orderings, ]
      }
      
      # Create plot
      ggplot(plot_data, aes(x = Ordering, y = Information, fill = factor(Level))) +
        geom_bar(stat = "identity") +
        labs(
          title = paste("Multilevel Information Decomposition for", target_col),
          #subtitle = "Based on equation 6.10 from textbook",
          x = "Feature Ordering",
          y = "Mutual Information",
          fill = "Decomposition Level"
        ) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        scale_fill_brewer(palette = "Set3")
    }
  ))
}

# Load necessary library for permutations
if(!require(gtools)) {
  install.packages("gtools")
  library(gtools)
}


data[,income:=total_income]
income_data=data[cohort==1982 & year==2022 & income>0, c("income", "Gender", 
                                                "profession_4", "Sun2000niva", 
                                                "sector","bost_kommun","city_uni",
                                                "city_res", "uni_id")]
income_data=income_data[!is.na(income)]
income_data[,income:=as.numeric(income)]
income_data[,education:=as.factor(Sun2000niva)]
income_data[,profession:=as.factor(profession_4)]
income_data[,city:=as.factor(city_res)]
income_data[,university:=as.factor(uni_id)]
# Example usage for multiple variables:
# 1. Simple example with 3 features
vars_3 <- c("Gender", "profession", "education")
results_3 <- multilevel_info_decomposition(income_data, "income", vars_3)

# Display summary
head(results_3$summary, 10)

# Plot the results
results_3$plot()

# 2. Example with 4 features
vars_4 <- c("Gender", "profession", "education", "city")
results_4 <- multilevel_info_decomposition(income_data, "income", vars_4)

# Display summary
head(results_4$summary, 10)

# Plot the results
results_4$plot()

# 3. Example with all 5 features
vars_5 <- c("Gender", "profession", "education", "city","uni_id")
results_5 <- multilevel_info_decomposition(income_data, "income", vars_5)

# Plot top 5 orderings if there are many permutations
results_5$plot(top_n = 5)
ggsave("Distributions/graphs/feature_combinations.pdf")


# Extract most informative ordering
most_informative <- aggregate(Information ~ Ordering, data = results_5$summary, FUN = sum)
most_informative <- most_informative[order(most_informative$Information, decreasing = TRUE), ]
top_ordering <- most_informative$Ordering[1]

cat("Most informative feature ordering:", top_ordering, "\n")
cat("Total information captured (bits):", most_informative$Information[1], "\n\n")

# Show detailed breakdown of the top ordering
top_results <- results_5$summary[results_5$summary$Ordering == top_ordering, ]
top_results <- top_results[order(top_results$Level), ]
print(top_results[, c("Feature", "Information", "Proportion")])

# Function to analyze all feature subsets
analyze_feature_combinations <- function(data, target_col, feature_cols, max_features = 3) {
  results <- data.frame()
  
  # Create all possible combinations of features
  for(k in 1:min(length(feature_cols), max_features)) {
    combos <- combn(feature_cols, k, simplify = FALSE)
    
    for(features in combos) {
      # Calculate information for this feature set
      info_decomp <- multilevel_info_decomposition(data, target_col, features)
      
      # Get total information
      total_info <- sum(info_decomp$summary$Information[info_decomp$summary$Ordering == paste(features, collapse = " → ")])
      
      # Add to results
      results <- rbind(results, data.frame(
        FeatureSet = paste(features, collapse = ", "),
        NumFeatures = length(features),
        TotalInformation = total_info
      ))
    }
  }
  
  # Return sorted results
  return(results[order(results$TotalInformation, decreasing = TRUE), ])
}

# Example usage analyzing different feature combinations
feature_combos <- analyze_feature_combinations(income_data, "income", vars_3, max_features = 3)
head(feature_combos, 10)

# Plot top feature combinations
ggplot(head(feature_combos, 10), aes(x = reorder(FeatureSet, TotalInformation), y = TotalInformation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Top Feature Combinations for Predicting Income",
    x = "Feature Set",
    y = "Total Mutual Information"
  ) +
  theme_minimal() +
  coord_flip()


ggsave("Distributions/graphs/feature_combinationsvars_3.pdf")


# Example usage analyzing different feature combinations
feature_combos <- analyze_feature_combinations(income_data, "income", vars_4, max_features = 3)
head(feature_combos, 10)

# Plot top feature combinations
ggplot(head(feature_combos, 10), aes(x = reorder(FeatureSet, TotalInformation), y = TotalInformation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Top Feature Combinations for Predicting Income",
    x = "Feature Set",
    y = "Total Mutual Information"
  ) +
  theme_minimal() +
  coord_flip()


ggsave("Distributions/graphs/feature_combinationsvars_4.pdf")

# Example usage analyzing different feature combinations
feature_combos <- analyze_feature_combinations(income_data, "income", vars_5, max_features = 3)
head(feature_combos, 10)

# Plot top feature combinations
ggplot(head(feature_combos, 10), aes(x = reorder(FeatureSet, TotalInformation), y = TotalInformation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Top Feature Combinations for Predicting Income",
    x = "Feature Set",
    y = "Total Mutual Information"
  ) +
  theme_minimal() +
  coord_flip()


ggsave("Distributions/graphs/feature_combinationsvars_5.pdf")
