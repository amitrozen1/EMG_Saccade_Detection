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


data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_readnig_for_R_paired_ordered.csv")

data <- convert_to_factors(data, c("session", "whoseReading","ListenerChoice", "order_type", "pair_id", "ReaderChoice", "chose_same"))




data$minus_corr <- create_minus_column(data,"chosen_max_comp_mean_corr_1","chosen_max_comp_mean_corr_2")
data$minus_lag <- create_minus_column(data,"chosen_max_comp_mean_lag_1","chosen_max_comp_mean_lag_2")
data$minus_all_corr <- create_minus_column(data, "chosen_comps_mean_corr_1", "chosen_comps_mean_corr_2")

data$minus_listener_eye <- create_minus_column(data, "Listener_mean_Eye_1", "Listener_mean_Eye_2")
data$minus_listener_zyg <- create_minus_column(data, "Listener_mean_Zygomaticus_1", "Listener_mean_Zygomaticus_2")
data$minus_listener_nose <- create_minus_column(data, "Listener_mean_Nose_1", "Listener_mean_Nose_2")
data$minus_listener_corrugator <- create_minus_column(data, "Listener_mean_Corrugator_1", "Listener_mean_Corrugator_2")

data$minus_reader_eye <- create_minus_column(data, "Reader_mean_Eye_1", "Reader_mean_Eye_2")
data$minus_reader_zyg <- create_minus_column(data, "Reader_mean_Zygomaticus_1", "Reader_mean_Zygomaticus_2")
data$minus_reader_nose <- create_minus_column(data, "Reader_mean_Nose_1", "Reader_mean_Nose_2")
data$minus_reader_corrugator <- create_minus_column(data, "Reader_mean_Corrugator_1", "Reader_mean_Corrugator_2")




listener_muscles_glm <- glm(ListenerChoice ~ 1 + minus_listener_eye + minus_listener_zyg + minus_listener_nose + minus_listener_corrugator, data=data, family=binomial(link = "logit"))
summary(listener_muscles_glm)
plot(allEffects(listener_muscles_glm))

reader_muscles_glm <- glm(ListenerChoice ~ 1 + minus_reader_eye + minus_reader_zyg + minus_reader_nose + minus_reader_corrugator, data=data, family=binomial(link = "logit"))
summary(reader_muscles_glm)

paired <- glm(ListenerChoice ~ 1 + minus_reader_eye + minus_reader_zyg + minus_reader_nose + minus_reader_corrugator
              +minus_listener_eye + minus_listener_zyg + minus_listener_nose + minus_listener_corrugator
              + minus_corr + minus_lag, data=data, family=binomial(link = "logit"))

summary(paired)
plot(allEffects(paired))


sync_glm <- glm(ListenerChoice ~ 1 + minus_corr + minus_lag, data=data, family=binomial(link = "logit"))
summary(sync_glm)
plot(allEffects(sync_glm))


synchrony_model <- glmer(ListenerChoice ~ 1 +
                           minus_corr
                       + (1 + minus_corr | session),
                       data = data, 
                       family = binomial(link = "logit"))

summary(synchrony_model)


library(effects)

# Plot conditional effects
plot(allEffects(synchrony_model))

