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

create_minus_column_single_name <- function(data, col_name)
{
    col1 <- paste0(col_name, "_1")
    col2 <- paste0(col_name, "_2")
    return(create_minus_column(data, col1, col2))
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

data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_RMS_of_corr_comp_VAR_readnig_for_R_paired_ordered.csv")
#data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_20_window_3_maxlag_2_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_mean_of_corr_comp_VAR_readnig_for_R_paired_ordered.csv")

data <- convert_to_factors(data, c("session", "whoseReading","ListenerChoice", "order_type", "pair_id", "ReaderChoice", "chose_same"))


data$minus_corr <- create_minus_column(data,"chosen_max_comps_mean_corr_1","chosen_max_comps_mean_corr_2")
data$minus_lag <- create_minus_column(data,"chosen_max_comps_mean_lag_1","chosen_max_comps_mean_lag_2")

data$minus_sum_corr <- create_minus_column_single_name(data, "chosen_sum_comps_mean_corr")
data$minus_sum_lag <- create_minus_column_single_name(data, "chosen_sum_comps_mean_lag")

data$minus_listener_chosen_mean <- create_minus_column_single_name(data, "Listener_mean_chosen_max")
data$minus_reader_chosen_mean <- create_minus_column_single_name(data, "Reader_mean_chosen_max")

data$minus_listener_sum_mean <- create_minus_column_single_name(data, "Listener_mean_sum")
data$minus_reader_sum_mean <- create_minus_column_single_name(data, "Reader_mean_sum")

sum_corr_glm <- glm(ListenerChoice ~ 1 + minus_sum_corr + minus_listener_sum_mean + minus_reader_sum_mean + minus_sum_lag, data=data, family=binomial(link = "logit"))
summary(sum_corr_glm)
plot(allEffects(sum_corr_glm))

max_corr_glm <- glm(ListenerChoice ~ 1 + minus_corr + minus_lag+ minus_listener_chosen_mean + minus_reader_chosen_mean , data=data, family=binomial(link = "logit"))
summary(max_corr_glm)
plot(allEffects(max_corr_glm))

hir_model <- glmer(ListenerChoice ~ 1 + minus_corr + ( 1 + minus_corr | session)
                   , data=data, family=binomial(link = "logit"))
summary(hir_model)
plot(allEffects(hir_model))

#load inertial:

#data_inertial <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_accelerometer_pearson_correlations_readnig_for_R_paired_ordered.csv")
data_inertial <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_400_downsample_20_window_2_maxlag_3_gyro_pearson_correlations_no_artifact_remove_readnig_for_R_paired_ordered.csv")

data_inertial <- convert_to_factors(data_inertial, c("session", "whoseReading","ListenerChoice", "order_type", "pair_id", "ReaderChoice", "chose_same"))

data_inertial$minus_max_comp_corr <- create_minus_column_single_name(data_inertial, "inertial_max_comp_mean_corr")
data_inertial$minus_max_comp_lag <- create_minus_column_single_name(data_inertial, "inertial_max_comp_mean_lag")

data_inertial$minus_rms_comp_corr <- create_minus_column_single_name(data_inertial, "inertial_rms_comp_mean_corr")
data_inertial$minus_rms_comp_lag <- create_minus_column_single_name(data_inertial, "inertial_rms_comp_mean_lag")

data_inertial$minus_listener_max_comp <- create_minus_column_single_name(data_inertial, "Listener_mean_max_comp")
data_inertial$minus_reader_max_comp <- create_minus_column_single_name(data_inertial, "Reader_mean_max_comp")

data_inertial$minus_listener_rms_comp <- create_minus_column_single_name(data_inertial, "Listener_mean_rms_comp")
data_inertial$minus_reader_rms_comp <- create_minus_column_single_name(data_inertial, "Reader_mean_rms_comp")

max_comp_glm <- glm(ListenerChoice ~ 1 + minus_max_comp_corr + minus_max_comp_lag + minus_listener_max_comp + minus_reader_max_comp, data=data_inertial, family=binomial(link = "logit"))
summary(max_comp_glm)
plot(allEffects(max_comp_glm))

rms_comp_glm <- glm(ListenerChoice ~ 1 + minus_rms_comp_corr + minus_rms_comp_lag + minus_listener_rms_comp + minus_reader_rms_comp, data=data_inertial, family=binomial(link = "logit"))
summary(rms_comp_glm)
plot(allEffects(rms_comp_glm))

data_inertial$minus_max_comp_lag <- normalize_and_centralize_col(data_inertial$minus_max_comp_lag)

rms_comp_glmer <- glmer(ListenerChoice ~ 1 + minus_rms_comp_corr + minus_rms_comp_lag
                        + (1+minus_rms_comp_corr + minus_rms_comp_lag | session)
                        , data=data_inertial, family=binomial(link = "logit"))

summary(rms_comp_glmer)
plot(allEffects(rms_comp_glmer))


muscle_data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_corrEachComp_readnig_for_R_paired_ordered.csv")

muscle_data <- convert_to_factors(muscle_data, c("session", "whoseReading","ListenerChoice", "order_type", "pair_id", "ReaderChoice", "chose_same"))

for (muscle in c("Nose", "Eye", "Zygomaticus", "Corrugator"))
{
    muscle_data[[paste0("minus_",muscle,"_corr")]] <- create_minus_column_single_name(muscle_data, paste0(muscle,"_mean_corr"))
    muscle_data[[paste0("minus_",muscle,"_lag")]] <- create_minus_column_single_name(muscle_data, paste0(muscle,"_mean_lag"))
    muscle_data[[paste0("minus_Listener_",muscle)]] <- create_minus_column_single_name(muscle_data, paste0("Listener_mean_",muscle))
    muscle_data[[paste0("minus_Reader_",muscle)]] <- create_minus_column_single_name(muscle_data, paste0("Reader_mean_",muscle))
}

# Nose model:
nose_glm <- glm(ListenerChoice ~ 1 + minus_Nose_corr + minus_Nose_lag + minus_Listener_Nose + minus_Reader_Nose, data=muscle_data, family=binomial(link = "logit"))
summary(nose_glm)
plot(allEffects(nose_glm))

# Eye model:
eye_glm <- glm(ListenerChoice ~ 1 + minus_Eye_corr + minus_Eye_lag + minus_Listener_Eye + minus_Reader_Eye, data=muscle_data, family=binomial(link = "logit"))
summary(eye_glm)
plot(allEffects(eye_glm))

# Zygomaticus model:
zygomaticus_glm <- glm(ListenerChoice ~ 1 + minus_Zygomaticus_corr + minus_Zygomaticus_lag + minus_Listener_Zygomaticus + minus_Reader_Zygomaticus, data=muscle_data, family=binomial(link = "logit"))
summary(zygomaticus_glm)
plot(allEffects(zygomaticus_glm))

# Corrugator model:
corrugator_glm <- glm(ListenerChoice ~ 1 + minus_Corrugator_corr + minus_Corrugator_lag + minus_Listener_Corrugator + minus_Reader_Corrugator, data=muscle_data, family=binomial(link = "logit"))
summary(corrugator_glm)
plot(allEffects(corrugator_glm))

#all corr model:
all_corr_glm <- glm(ListenerChoice ~ 1 + minus_Nose_corr + minus_Eye_corr + minus_Zygomaticus_corr + minus_Corrugator_corr ,data=muscle_data, family=binomial(link = "logit"))
summary(all_corr_glm)
plot(allEffects(all_corr_glm))





#models with var:
data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_mean_of_corr_comp_VAR_readnig_for_R_paired_ordered.csv")

data <- convert_to_factors(data, c("session", "whoseReading","ListenerChoice", "order_type", "pair_id", "ReaderChoice", "chose_same"))

data$minus_corr <- create_minus_column_single_name(data, "chosen_max_comps_mean_corr")
data$minus_lag <- create_minus_column_single_name(data, "chosen_max_comps_mean_lag")
data$minus_cor_var <- create_minus_column_single_name(data, "chosen_max_comps_var_corr")
data$minus_lag_var <- create_minus_column_single_name(data, "chosen_max_comps_var_lag")

data$minus_sum_corr <- create_minus_column_single_name(data, "chosen_sum_comps_mean_corr")
data$minus_sum_lag <- create_minus_column_single_name(data, "chosen_sum_comps_mean_lag")
data$minus_sum_cor_var <- create_minus_column_single_name(data, "chosen_sum_comps_var_corr")
data$minus_sum_lag_var <- create_minus_column_single_name(data, "chosen_sum_comps_var_lag")


data$minus_corr <- normalize_and_centralize_col(data$minus_corr)
data$minus_cor_var <- normalize_and_centralize_col(data$minus_cor_var)
data$minus_lag <- normalize_and_centralize_col(data$minus_lag)
data$minus_lag_var <- normalize_and_centralize_col(data$minus_lag_var)



#model of mean and var corr:
corr_glm <- glm(ListenerChoice ~ 1 + minus_corr + minus_cor_var + minus_lag + minus_lag_var, data=data, family=binomial(link = "logit"))
summary(corr_glm)
plot(allEffects(corr_glm))

corr_glm <- glm(ListenerChoice ~ 1 + minus_corr, data=data, family=binomial(link = "logit"))
summary(corr_glm)
plot(allEffects(corr_glm))




