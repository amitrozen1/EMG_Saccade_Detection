# Load necessary libraries
library(lme4)
library(tidyverse)



normalize_and_centralize <- function(data, colname) {
  ## usage example:
  #  data_normalized <- normalize_and_centralize(data, "scores")
  
  
  # Check if the specified column exists in the dataframe
  if (!colname %in% names(data)) {
    stop("Column not found in the dataframe")
  }
  
  # Extract the column from the dataframe
  column_data <- data[[colname]]
  
  # Calculate mean and standard deviation of the column
  mean_data <- mean(column_data, na.rm = TRUE)
  sd_data <- sd(column_data, na.rm = TRUE)
  
  # Centralize and normalize the column
  normalized_data <- (column_data - mean_data) / sd_data
  
  # Replace the original column with the normalized data
  data[[colname]] <- normalized_data
  
  # Return the modified dataframe
  return(data)
}

# Load the CSV file into a data frame
# Load the CSV file into a data frame
# Double backslash method
data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_readnig_for_R_new_hirarchy.csv")

data$session <- as.factor(data$session)
data$story <- as.factor(data$story)
data$pair_id <- as.factor(data$pair_id)

#data$story <- as.factor(data$story)
data$ListenerChoice <- as.factor(data$ListenerChoice)
data$ChoseTogether <- as.factor(data$ChoseTogether)

data$order <- as.factor(data$order)
data$ListenerChoiceOrdered <- as.factor(data$ListenerChoiceOrdered)
data$ListenerChoicePairedOrdered <- as.factor(data$ListenerChoicePairedOrdered)


# Define a vector of column names to process
corr_columns <- c( "chosen_comps_mean_corr", "chosen_comps_max_corr",
                   "chosen_max_comp_mean_corr", "chosen_max_comp_max_corr")

lag_columns <- c("chosen_comps_mean_lag","chosen_max_comp_mean_lag")

muscle_cols <- c("Reader_Nose", "Listener_Nose", "Reader_Eye", "Listener_Eye", "Reader_Zygomaticus", "Listener_Zygomaticus", "Reader_Corrugator", "Listener_Corrugator")

# Loop through each column name and apply the normalize_and_centralize function
for (col_name in corr_columns) {
  data[[col_name]] <- psych::fisherz(data[[col_name]])
  data <- normalize_and_centralize(data, col_name)
}

for (col_name in lag_columns) {
  data <- normalize_and_centralize(data, col_name)
}

for (col_name in muscle_cols) {
  data <- normalize_and_centralize(data, col_name)
}


# might need to code again the choice.

# for sum_choice I should use poisson as it's the distribution for count. poisson assumption is that the mean equals to the variance of the outcome var
# it this is not the case - use negative binomial.

# ordered logistic is also an option - need to see if viable when there are only 2 - maybe it's not. it's more for "low" "medium" "high" or 1st, 2nd,3rd, 4th...


# check model random and fixed effects:

baseline <- glmer(ListenerChoicePairedOrdered ~ 1 + chosen_comps_mean_corr + chosen_comps_max_corr + chosen_comps_mean_lag
                 +order + (1|session) + (1|pair_id),
                 data = data, 
                 family = binomial(link = "logit"))

summary(baseline)

random_effects <- glmer(ListenerChoicePairedOrdered ~ 1 + chosen_max_comp_mean_corr
                  + order + (1 + chosen_comps_mean_corr |session) + (1|pair_id),
                 data = data, 
                 family = binomial(link = "logit"))

summary(random_effects)





model <- glmer(ListenerChoicePairedOrdered ~ 1 + chosen_comps_mean_corr + chosen_comps_max_corr + chosen_comps_mean_lag
                    + Listener_Nose + Listener_Eye + Listener_Zygomaticus + Listener_Corrugator
                    + Reader_Nose + Reader_Eye + Reader_Zygomaticus + Reader_Corrugator
                    +order + (1|session) + (1|pair_id),
               data = data, 
               family = binomial(link = "logit"))


# Summarize the model
performance::icc(model)
summary(model)

# To examine model diagnostics and ensure the model fits well, consider plotting:
# AIC of the model for model comparison
print(AIC(model))

# Checking for convergence issues
print(convergence(model))



data$newcorr <- data$chosen_comps_mean_corr / data$chosen_comps_mean_lag

data %>% 
  ggplot(aes(x = ListenerChoice, y = all_max_comp_mean_corr)) +
  geom_boxplot()


library(brms)

bayesian_model <- brm(
  formula = ListenerChoicePairedOrdered ~ 1 + chosen_max_comp_mean_corr + chosen_max_comp_max_corr + chosen_max_comp_mean_lag
  + order + (1|session) + (1|pair_id), # check if the order matters.
  data = data, 
  family = bernoulli(link = "logit"),
  prior = c(
    prior(normal(0, 5), class = "b"),         # Default normal priors for coefficients
    prior(student_t(3, 0, 10), class = "sd")  # Default student-t priors for group-level standard deviations
  ),
  chains = 4,
  warmup = 1000,
  iter = 2000,
  cores = parallel::detectCores()
)

# Print model summary
print(summary(bayesian_model))


