library(lme4)
library(dplyr)
library(effects)


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

data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_RMS_of_corr_comp_VAR_listening_for_R_paired_ordered.csv")

data <- convert_to_factors(data, c("session", "choice_a","choice_b", "order_type", "pair_id", "chose_same", "is_other"))


data$minus_corr <- create_minus_column(data,"chosen_max_comps_mean_corr_1","chosen_max_comps_mean_corr_2")
data$minus_lag <- create_minus_column(data,"chosen_max_comps_mean_lag_1","chosen_max_comps_mean_lag_2")

data$minus_sum_corr <- create_minus_column_single_name(data, "chosen_rms_comps_mean_corr")
data$minus_sum_lag <- create_minus_column_single_name(data, "chosen_rms_comps_mean_lag")

data$minus_A_chosen_mean <- create_minus_column_single_name(data, "mean_chosen_max_A")
data$minus_B_chosen_mean <- create_minus_column_single_name(data, "mean_chosen_max_B")

data$minus_A_sum_mean <- create_minus_column_single_name(data, "mean_chosen_rms_A")
data$minus_B_sum_mean <- create_minus_column_single_name(data, "mean_chosen_rms_B")

sum_corr_glm <- glm(chose_same ~ 1 + minus_sum_corr + minus_A_sum_mean + minus_B_sum_mean + minus_sum_lag, data=data, family=binomial(link = "logit"))
summary(sum_corr_glm)
plot(allEffects(sum_corr_glm))


max_corr_glm <- glm(chose_same ~ 1 + minus_corr + minus_lag+ minus_A_chosen_mean + minus_B_chosen_mean + is_other, data=data, family=binomial(link = "logit"))
summary(max_corr_glm)
plot(allEffects(max_corr_glm))

hir_model <- glmer(chose_same ~ 1 + minus_corr + (1 + order_type | pair_id)
                   , data=data, family=binomial(link = "logit"))
summary(hir_model)
plot(allEffects(hir_model))

max_corr_glm <- glm(chose_same ~1 + chosen_max_comps_mean_corr_1 + chosen_max_comps_mean_corr_2  
                    , data=data, family=binomial(link = "logit"))
summary(max_corr_glm)
plot(allEffects(max_corr_glm))


a_model <- glm(choice_a ~ 1 + minus_corr + minus_lag + minus_A_chosen_mean, data=data, family=binomial(link = "logit"))
summary(a_model)

b_model <- glm(choice_a ~ 1 + minus_corr + minus_lag + minus_B_chosen_mean, data=data, family=binomial(link = "logit"))
summary(b_model)



data$minus_corr <- abs(data$minus_corr)
data$minus_lag <- abs(data$minus_lag)

abs_model <- glm(chose_same ~ 1 + minus_corr + minus_lag, data=data, family=binomial(link = "logit"))
summary(abs_model)


for (muscle in c("Nose", "Eye", "Zygomaticus", "Corrugator"))
{
  data[[paste0("minus_",muscle,"_A")]] <- create_minus_column_single_name(data, paste0("mean_", muscle, "_A"))
  data[[paste0("minus_",muscle,"_B")]] <- create_minus_column_single_name(data, paste0("mean_", muscle, "_B"))
}

a_model <- glm(choice_a ~ 1 + minus_Nose_A + minus_Eye_A + minus_Zygomaticus_A + minus_Corrugator_A
               , data=data, family=binomial(link = "logit"))
summary(a_model)
plot(allEffects(a_model))


b_model <- glm(choice_b ~ 1 + minus_Nose_B + minus_Eye_B + minus_Zygomaticus_B + minus_Corrugator_B
               , data=data, family=binomial(link = "logit"))
summary(b_model)
plot(allEffects(b_model))

# ביק


#in this model nose a is more indicative for choiceA and nose b more for choiceB
comb_model <- glm(choice_b ~ 1 + minus_Nose_A + minus_Nose_B, 
                  data=data, family=binomial(link = "logit"))
summary(comb_model)
plot(allEffects(comb_model))

#should I load the original and try predicting sum choice?




data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_RMS_of_corr_comp_VAR_listening_for_R_split_for_particiapnt.csv")
data <- convert_to_factors(data, c("session", "choice", "order_type", "pair_id", "chose_same", "is_other"))

data$minus_corr <- create_minus_column_single_name(data,"chosen_max_comps_mean_corr")
data$minus_lag <- create_minus_column_single_name(data,"chosen_max_comps_mean_lag")

data$minus_sum_corr <- create_minus_column_single_name(data, "chosen_rms_comps_mean_corr")
data$minus_sum_lag <- create_minus_column_single_name(data, "chosen_rms_comps_mean_lag")

data$minus_chosen_mean <- create_minus_column_single_name(data, "mean_chosen_max")
data$minus_sum_mean <- create_minus_column_single_name(data, "mean_chosen_rms")

for (muscle in c("Nose", "Eye", "Zygomaticus", "Corrugator"))
{
  data[[paste0("minus_",muscle)]] <- create_minus_column_single_name(data, paste0("mean_", muscle))
}

choice_model <- glm(choice ~ 1 + minus_corr + minus_lag + minus_chosen_mean, data=data, family=binomial(link = "logit"))
summary(choice_model)

muscle_model <- glm(choice ~ 1 + minus_Nose + minus_Eye + minus_Zygomaticus + minus_Corrugator, data=data, family=binomial(link = "logit"))
summary(muscle_model)

muscle_sync_model <- glm(choice ~ 1 + minus_Nose + minus_Eye + minus_Zygomaticus + minus_Corrugator + minus_corr + minus_lag, data=data, family=binomial(link = "logit"))
summary(muscle_sync_model)

nose_model <- glm(choice ~ 1 + minus_Nose, data=data, family=binomial(link = "logit"))
summary(nose_model)
plot(allEffects(nose_model))



#todo: remove prediction outlayers - 5 less.
cooksd <- cooks.distance(nose_model)
plot(cooksd, type="h", main="Cook's Distance", ylab="Cook's Distance")  # 'type="h"' creates a stem plot
abline(h = 4/(nrow(data) - length(nose_model$coefficients) - 1), col="red")  # Common threshold line

threshold <- 4/(nrow(data) - length(nose_model$coefficients) - 1)
outliers <- as.numeric(names(cooksd)[(cooksd > threshold)])
clean_data <- data[-outliers, ]

nose_model_clean <- glm(choice ~ 1 + minus_Nose, data=clean_data, family=binomial(link = "logit"))
summary(nose_model_clean)
plot(allEffects(nose_model_clean))

#improves p value to 0.000985


# clean by minus_Nose_value:
Q1 <- quantile(data$minus_Nose, 0.25)
Q3 <- quantile(data$minus_Nose, 0.75)

# Calculate the Interquartile Range (IQR)
IQR <- Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Identify and remove outliers - removed 24 observations.
clean_data_iqr <- data[data$minus_Nose >= lower_bound & data$minus_Nose <= upper_bound, ]

nose_model_clean_iqr <- glm(choice ~ 1 + minus_Nose + minus_Eye + minus_Zygomaticus + minus_Corrugator + minus_corr, data=clean_data_iqr, family=binomial(link = "logit"))
summary(nose_model_clean_iqr)
plot(allEffects(nose_model_clean_iqr))

#pvalue is no longer significant.

#using zscore method:
mean_minus_Nose <- mean(data$minus_Nose)
sd_minus_Nose <- sd(data$minus_Nose)

# Calculate Z-scores
data$z_score <- (data$minus_Nose - mean_minus_Nose) / sd_minus_Nose

# Define a threshold for outliers (e.g., 3 or -3)
threshold <- 3

# Identify and remove outliers - removed 5 obs
clean_data_z <- data[abs(data$z_score) <= threshold, ]

nose_model_clean_z <- glm(choice ~ 1 + minus_Nose + minus_Eye + minus_Zygomaticus + minus_Corrugator, data=clean_data_z, family=binomial(link = "logit"))
summary(nose_model_clean_z)
plot(allEffects(nose_model_clean_z))

# pvalue is no longer significant.






data <- data[data$is_other == 1,]

choice_model <- glm(choice ~ 1 + minus_corr + minus_lag + minus_chosen_mean, data=data, family=binomial(link = "logit"))
summary(choice_model)

muscle_model <- glm(choice ~ 1 + minus_Nose + minus_Eye + minus_Zygomaticus + minus_Corrugator, data=data, family=binomial(link = "logit"))
summary(muscle_model)

muscle_sync_model <- glm(choice ~ 1 + minus_Nose + minus_Eye + minus_Zygomaticus + minus_Corrugator + minus_corr + minus_lag, data=data, family=binomial(link = "logit"))
summary(muscle_sync_model)

nose_model <- glm(choice ~ 1 + minus_Nose, data=data, family=binomial(link = "logit"))
summary(nose_model)
plot(allEffects(nose_model))


data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_corrEachComp_listening_for_R_split_for_particiapnt.csv")
data <- convert_to_factors(data, c("session", "choice", "order_type", "pair_id", "chose_same", "is_other"))

for (muscle in c("Nose", "Eye", "Zygomaticus", "Corrugator"))
{
  data[[paste0("minus_corr_",muscle)]] <- create_minus_column_single_name(data, paste0(muscle, "_mean_corr"))
  data[[paste0("minus_lag_",muscle)]] <- create_minus_column_single_name(data, paste0(muscle, "_mean_lag"))
  data[[paste0("minus_", muscle)]] <- create_minus_column_single_name(data, paste0("mean_", muscle))
}

choice_model <- glm(choice ~ 1 + minus_corr_Nose + minus_corr_Eye + minus_corr_Zygomaticus + minus_corr_Corrugator
                     + minus_lag_Nose + minus_lag_Eye + minus_lag_Zygomaticus + minus_lag_Corrugator
                     + minus_Nose + minus_Eye + minus_Zygomaticus + minus_Corrugator, data=data, family=binomial(link = "logit"))

summary(choice_model)


# sum choice model:
data <- read.csv("C:\\Liron\\DataEmg\\Try\\results_2024\\RMS_100_downsample_20_window_3_maxlag_2_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_mean_of_corr_comp_VAR.csv")
data <- convert_to_factors(data, c("session", "isOther"))

data = data[data$action == "Listening",]
data$mult_choice <- data$isChoiceA * data$isChoiceB
data$mult_choice <- as.factor(data$mult_choice)

glm_mult_choice <- glm(mult_choice ~ 1 + chosen_max_comps_mean_corr + chosen_max_comps_mean_lag
                      # + mean_Nose_A + mean_Nose_B
                      # + mean_Zygomaticus_A + mean_Zygomaticus_B
                      # + mean_Eye_A + mean_Eye_B
                      # + mean_Corrugator_A + mean_Corrugator_B
                      , data=data, family=binomial(link = "logit"))
summary(glm_mult_choice)
plot(allEffects(glm_mult_choice))


mean_lag <- mean(data$chosen_max_comps_mean_lag)
sd_lag <- sd(data$chosen_max_comps_mean_lag)

# Calculate Z-scores
data$z_score <- (data$chosen_max_comps_mean_lag - mean_lag) / sd_lag

# Define a threshold for outliers (e.g., 3 or -3)
threshold <- 3

# Identify and remove outliers - removed 5 obs
clean_data_z <- data[abs(data$z_score) <= threshold, ]

glm_mult_choice <- glm(mult_choice ~ 1 + chosen_max_comps_mean_corr + chosen_max_comps_mean_lag
                       # + mean_Nose_A + mean_Nose_B
                       # + mean_Zygomaticus_A + mean_Zygomaticus_B
                       # + mean_Eye_A + mean_Eye_B
                       # + mean_Corrugator_A + mean_Corrugator_B
                       , data=clean_data_z, family=binomial(link = "logit"))
summary(glm_mult_choice)
plot(allEffects(glm_mult_choice))





data$sum_choice <- data$isChoiceA + data$isChoiceB
data$sum_choice <- as.factor(data$sum_choice)

# change ordinal meaning.
glm_sum_choice <- glmer(sum_choice ~ 1 + chosen_max_comps_mean_corr + chosen_max_comps_mean_lag
                        #+ mean_Nose_A + mean_Nose_B
                        #+ mean_Zygomaticus_A + mean_Zygomaticus_B
                        #+ mean_Eye_A + mean_Eye_B
                        #+ mean_Corrugator_A + mean_Corrugator_B
                        + (1 + chosen_max_comps_mean_corr + chosen_max_comps_mean_lag | session)
                       , data=data, family=poisson(link="log"))
summary(glm_sum_choice)
plot(allEffects(glm_sum_choice))
