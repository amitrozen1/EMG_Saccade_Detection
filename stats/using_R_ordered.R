library(lme4)
library(dplyr)

calculate_ratio_and_centralize_base <- function(data, colname, group_name) {
  # Create new column names for a1, a2, and the ratio
  colname1 <- paste0(colname, "_1")
  colname2 <- paste0(colname, "_2")
  ratio_colname <- paste0(colname, "_ratio")
  
  # Calculate the ratio
  data[[ratio_colname]] <- (data[[colname2]] - data[[colname1]]) / (data[[colname2]] + data[[colname1]])
  
  # Centralize the ratio within groups
  # We calculate the mean of the ratio for each group and then subtract it from each group member
  means <- ave(data[[ratio_colname]], data[[group_name]], FUN = mean)
  data[[ratio_colname]] <- data[[ratio_colname]] - means
  
  # Return the updated dataframe
  return(data)
}




create_minus_column <- function(data, col1, col2)
{
  return (data[,col1] - data[,col2]) / (data[,col1]+data[,col2])
  #return (data[,col1] - data[,col2])
}


convert_to_factors <- function(data, colnames) {
  for (colname in colnames) {
    data[[colname]] <- as.factor(data[[colname]])
  }
  return(data)
}

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


normalize_and_centralize_col <- function(column) {
  normalized_column <- (column - mean(column, na.rm = TRUE)) / sd(column, na.rm = TRUE)
  return(normalized_column)
}






#data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_readnig_for_R_paired_ordered.csv")

data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_readnig_for_R_paired_ordered.csv")

data <- convert_to_factors(data, c("session", "whoseReading","ListenerChoicePairedOrdered", "order_type", "pair_id"))

corr_columns <- c( "chosen_comps_mean_corr", "chosen_comps_max_corr",
                   "chosen_max_comp_mean_corr", "chosen_max_comp_max_corr",
                   "all_max_comp_mean_corr", "all_max_comp_max_corr")

lag_columns <- c("chosen_comps_mean_lag","chosen_max_comp_mean_lag", "all_max_comp_mean_lag")

muscle_cols <- c("Reader_mean_Nose", "Listener_mean_Nose", "Reader_mean_Eye", "Listener_mean_Eye", "Reader_mean_Zygomaticus", "Listener_mean_Zygomaticus", "Reader_mean_Corrugator", "Listener_mean_Corrugator")



data <- calculate_ratio_and_centralize(data, "chosen_max_comp_mean_corr", "session")


# for (col_name in c(corr_columns, lag_columns))
# {  
#   for (story in c("_1", "_2"))
#   {
#     colname_for_story <- paste(col_name, story, sep = "")
#     data <- normalize_and_centralize(data, colname_for_story)
#   }
# }


for(col_name in c(corr_columns))
{
  for (story in c("_1", "_2"))
  {
    colname_for_story <- paste(col_name, story, sep = "")
    data[,colname_for_story] = psych::fisherz(data[,colname_for_story])
  }
}

for(col_name in c(muscle_cols))
{
  for (story in c("_1", "_2"))
  {
    colname_for_story <- paste(col_name, story, sep = "")
    data[,colname_for_story] = normalize_and_centralize_col(data[,colname_for_story])
  }
}

for (col_name in c(corr_columns, lag_columns)) {
  # Combine the "_1" and "_2" data into a single column for normalization
  combined_data <- c(data[, paste(col_name, "_1", sep = "")], data[, paste(col_name, "_2", sep = "")])

  # Normalize the combined column
  combined_data <- normalize_and_centralize_col(combined_data)

  # Separate the normalized data back into the original "_1" and "_2" columns
  n <- nrow(data)
  data[, paste(col_name, "_1", sep = "")] <- combined_data[1:n]
  data[, paste(col_name, "_2", sep = "")] <- combined_data[(n + 1):(2*n)]
}


baseline <- glmer(ListenerChoice ~ 1 + chosen_max_comp_mean_corr_1 + chosen_max_comp_mean_corr_2 
                  + chosen_max_comp_mean_lag_1 + chosen_max_comp_mean_lag_2 + (-1 + order_type|pair_id) + 
                     + (1 + chosen_max_comp_mean_corr_1 + chosen_max_comp_mean_corr_2 |session),
                  data = data, 
                  family = binomial(link = "logit"))

summary(baseline)


baseline <- glmer(ListenerChoice ~ 1 + chosen_comps_mean_corr_1 + chosen_comps_mean_corr_2 
                   + (0 + order_type||pair_id) + 
                    + (0 + chosen_comps_mean_corr_1 + chosen_comps_mean_corr_2  || session),
                  data = data, 
                  family = binomial(link = "logit"))

summary(baseline)


data$minus_corr <- create_minus_column(data,"chosen_max_comp_mean_corr_1","chosen_max_comp_mean_corr_2")
data$minus_lag <- create_minus_column(data,"chosen_max_comp_mean_lag_1","chosen_max_comp_mean_lag_2")
data$minus_all_corr <- create_minus_column(data, "chosen_comps_mean_corr_1", "chosen_comps_mean_corr_2")





baseline <- glmer(ListenerChoice ~ 1 + minus_corr + minus_lag
                   + (0 + order_type || pair_id)
                    + (0 + minus_corr ||session),
                  data = data, 
                  family = binomial(link = "logit"))

summary(baseline)


data$Reader_Zygomaticus_Minus <- create_minus_column(data, "Reader_mean_Zygomaticus_1","Reader_mean_Zygomaticus_2")
data$Listener_Zygomaticus_Minus <- create_minus_column(data, "Listener_mean_Zygomaticus_1","Listener_mean_Zygomaticus_2")

data$Reader_Eye_Minus <- create_minus_column(data, "Reader_mean_Eye_1","Reader_mean_Eye_2")
data$Listener_Eye_Minus <- create_minus_column(data, "Listener_mean_Eye_1","Listener_mean_Eye_2")

data$Reader_Nose_Minus <- create_minus_column(data, "Reader_mean_Nose_1","Reader_mean_Nose_2")
data$Listener_Nose_Minus <- create_minus_column(data, "Listener_mean_Nose_1","Listener_mean_Nose_2")

data$Reader_Corrugator_Minus <- create_minus_column(data, "Reader_mean_Corrugator_1","Reader_mean_Corrugator_2")
data$Listener_Corrugator_Minus <- create_minus_column(data, "Listener_mean_Corrugator_1","Listener_mean_Corrugator_2")



data$minus_corr <- normalize_and_centralize_col(data$minus_corr)
data$minus_all_corr <- normalize_and_centralize_col(data$minus_all_corr)
data$Listener_Eye_Minus <- normalize_and_centralize_col(data$Listener_Eye_Minus)
data$Reader_Eye_Minus <- normalize_and_centralize_col(data$Reader_Eye_Minus)
data$Reader_Corrugator_Minus <- normalize_and_centralize_col(data$Reader_Corrugator_Minus)


data$minus_corr <- psych::fisherz(data$minus_corr)


muscles_model <- glmer(ChoseTogether ~ 1
                       + minus_corr + Listener_Eye_Minus + Reader_Eye_Minus
                       + (1 + Listener_Eye_Minus + Reader_Eye_Minus || session),
                       data = data, 
                       family = binomial(link = "logit"))

summary(muscles_model)

################# BEST #################

muscles_model <- glmer(ListenerChoice ~ 
                       minus_corr 
                       + (1 + minus_corr | session),
                       data = data, 
                       family = binomial(link = "logit"))

summary(muscles_model)

data$Listener_Eye_Minus <- normalize_and_centralize_col(data$Listener_Eye_Minus)
data$Listener_Corrugator_Minus <- normalize_and_centralize_col(data$Listener_Corrugator_Minus)
data$Listener_Nose_Minus <- normalize_and_centralize_col(data$Listener_Nose_Minus)
data$Listener_Zygomaticus_Minus <- normalize_and_centralize_col(data$Listener_Zygomaticus_Minus)


data$Reader_Eye_Minus <- normalize_and_centralize_col(data$Reader_Eye_Minus)
data$Reader_Corrugator_Minus <- normalize_and_centralize_col(data$Reader_Corrugator_Minus)
data$Reader_Nose_Minus <- normalize_and_centralize_col(data$Reader_Nose_Minus)
data$Reader_Zygomaticus_Minus <- normalize_and_centralize_col(data$Reader_Zygomaticus_Minus)


listener_model <- glmer(ListenerChoice ~ 1
                       + Listener_Corrugator_Minus + Listener_Nose_Minus + Listener_Eye_Minus + Listener_Zygomaticus_Minus
                       + (1 | session),
                       data = data, 
                       family = binomial(link = "logit"))

summary(listener_model)



Reader_model <- glmer(ReaderChoice ~ 1
                      + Reader_Corrugator_Minus + Reader_Nose_Minus + Reader_Eye_Minus + Reader_Zygomaticus_Minus
                      + (1 + order_type | pair_id)
                      + (1 | session),
                      data = data, 
                      family = binomial(link = "logit"))

summary(Reader_model)



muscles_model <- glmer(ListenerChoice ~ 1
                       + minus_corr 
                       + (1 + order_type | pair_id)
                       + (1 + minus_corr | session),
                       data = data, 
                       family = binomial(link = "logit"))

summary(muscles_model)






ListenerChoice ~ 1 + sync1 + sync2 + (1+sync1 + sync2| session) + (1+order_type|story_pair_id)


baseline <- glmer(ListenerChoicePairedOrdered ~ 1 + (1 |pair_id) 
                  ,data = data
                  ,family = binomial(link = "logit"))

summary(baseline)



baseline <- glmer(ListenerChoice ~ 1 + chosen_max_comp_mean_corr_1 + chosen_max_comp_mean_corr_2
                  + (1|order_type/pair_id) +    
                  + (1 + chosen_max_comp_mean_corr_1 + chosen_max_comp_mean_corr_2   | session) , 
                  data = data, 
                  family = binomial(link = "logit"))

summary(baseline)


# read lme4 documentation.
# (1|order_type/pair_id)
# (1|order_type:pair_id)
#(1 + pair_id||order_type)
#(0 + pair_id||order_type)
read: 
1. dale bar 2013 - keep it maximal.
2. matuschek  - ~2017 random structure



#continue with removing variables that explains less variability

# there's a lib named : buildmer, 

# try it again with probabilities.











baseline <- glmer(ListenerChoice ~ 1 + minus*order_type * pair_id 
                    + (1  |session),
                  data = data, 
                  family = binomial(link = "logit"))

summary(baseline)



baseline <- glmer(ListenerChoice ~ 1 + all_max_comp_mean_corr_1 + all_max_comp_mean_corr_2 
                  + all_max_comp_mean_lag_1 + all_max_comp_mean_lag_2 + (-1 + order_type|pair_id) + 
                    + (1 + all_max_comp_mean_corr_1 + all_max_comp_mean_corr_2  |session),
                  data = data, 
                  family = binomial(link = "logit"))

summary(baseline)



# should I combine pairid and reverse order into one parameter ? 

library(brms)

bayesian_model <- brm(
  formula = ListenerChoice ~ 1 + chosen_comps_mean_corr_1 + chosen_comps_mean_corr_2 
            + (-1 + order_type|pair_id) + 
            + (1 + chosen_comps_mean_corr_1 + chosen_comps_mean_corr_2 |session),
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

