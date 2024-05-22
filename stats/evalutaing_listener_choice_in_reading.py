from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

# Load the data from the uploaded Excel file
file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_200_downsample_40_window_3_maxlag_1_after_decision_and_remove_artifacts_with_callibration.csv"
data = pd.read_csv(file_path)
data = data[data['action'] == 'Reading']
data['story'] = data['key'].str.extract('Reading_(\d+).png')[0].astype(int)


# Adjust the dataset to reflect the listener's choice correctly
data['ListenerChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 1 else row['isChoiceA'], axis=1)

# formula = 'ListenerChoice ~ chosen_comps_mean_corr + chosen_comps_mean_lag'

# # We use a Mixed Linear Model with a binomial family since smf.mixedlm does not directly support logistic regression
# # This workaround involves using GLMM from another package, but for simplicity, we'll demonstrate with a fixed effects logistic model here
# model = smf.logit(formula, data=data, groups=data["session"], re_formula='1').fit()

# # Summary of the model to check the fit and coefficients
# print(model.summary())

# # Plotting the effect of chosen_comps_mean_corr on Listener's Choice probability
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


# # Generate data for prediction
# pred_data = pd.DataFrame({
#     'chosen_comps_mean_corr': np.linspace(data['chosen_comps_mean_corr'].min(), data['chosen_comps_mean_corr'].max(), 100),
#     'whoseReading': 1,  # Predict for when A is reading first
# })

# # Predicting probabilities
# pred_data['prob_choice'] = model.predict(pred_data)

# # Plot
# sns.lineplot(x='chosen_comps_mean_corr', y='prob_choice', data=pred_data)
# plt.xlabel('chosen_comps_mean_corr')
# plt.ylabel('Probability of Choosing A')
# plt.show()


model_formula = 'ListenerChoice ~ chosen_comps_mean_corr'

# Define a dict where keys are covariance structure names and
# values are formulas for the random effects
vc_formula = {'session': '0 + C(session)'
              ,'story': '0 + C(story)'}

# Fit the model
mixed_model = smf.mixedlm(model_formula, data, groups=data['session'], 
                          vc_formula=vc_formula, family=sm.families.Binomial()).fit()

# Print the model's summary
print(mixed_model.summary())