import os
import pandas as pd
from scipy.stats import zscore


def zscore_values_over_couple(df, session, column):
    sessions = df["session"].unique()
    zscored = zscore(df[df["session"] == sessions[0]][column])
    # update the df with the zscored values:
    for session in sessions:
        df.loc[df["session"] == session, column] = zscore(df[df["session"] == session][column])
    return df


filename = "RMS_100_downsample_20_window_1000_step_25_pixels_no_reader_assumption_no_zyg_on_read_new_comps.csv"
path = os.path.join("C:\\Liron\\DataEmg\\Try\\results_2024", filename)

measure_col = "chosen_comps_mean_" #chosen_max_comp_mean_"


df = pd.read_csv(path)
df = df[df["action"] == "Listening"]

sessions =df["session"].unique()

# for session in sessions:
#     zscore_values_over_couple(df, session, measure_col + "corr")
#     zscore_values_over_couple(df, session, measure_col + "lag")


#take only trials when isChoiceA == isChoiceB:
df = df[df["isChoiceA"] == df["isChoiceB"]]

unique_stories = df["key"].unique()

stories_sum_counter = dict()

for story in unique_stories:
    stories_sum_counter[story] = [0,0]


for story in unique_stories:
    df_story = df[df["key"] == story]
    a_choses = df_story["isChoiceA"].sum()
    b_choses = df_story["isChoiceB"].sum()
    stories_sum_counter[story][0] = a_choses + b_choses
    stories_sum_counter[story][1] = df_story[measure_col + "corr"].sum()

#print(stories_sum_counter)
if False:
    unique_pairs = [(unique_stories[i], unique_stories[i+1]) for i in range(0, len(unique_stories), 2)]
    for pair in unique_pairs:
        print(pair)
        story0_df = df[df["key"] == pair[0]]
        story1_df = df[df["key"] == pair[1]]
        chosen_story0 = story0_df[story0_df["isChoiceA"]==1][measure_col]
        chosen_story1 = story1_df[story1_df["isChoiceA"]==1][measure_col]
        unchosen_story0 = story0_df[story0_df["isChoiceA"]==0][measure_col]
        unchosen_story1 = story1_df[story1_df["isChoiceA"]==0][measure_col]

        #combine lists:
        chosen = list(chosen_story0) + list(chosen_story1)
        unchosen = list(unchosen_story1) + list(unchosen_story0)

        #paired t-test:
        from scipy.stats import ttest_rel
        t, p = ttest_rel(chosen, unchosen)
        print(f"t: {t}, p: {p}")

        #reg t-test:
        from scipy.stats import ttest_ind
        t, p = ttest_ind(chosen, unchosen)
        print(f"t: {t}, p: {p}")

import statsmodels.api as sm

print(sm.GLM(df["isChoiceA"], df[measure_col + "corr"], family=sm.families.Binomial()).fit().summary())
#glm by meaure_col action and session:
print(sm.GLM(df["isChoiceA"], df[[measure_col + "corr", measure_col + "lag"]], family=sm.families.Binomial()).fit().summary())

#plot:
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=df, x=measure_col + "corr", y="isChoiceA")
plt.show()
sns.scatterplot(data=df, x=measure_col + "lag", y="isChoiceA")
plt.show()

# turn into mix model.


# showing this path:
# first - maybe we could not predict listener choice 
# so we chose continue with listening and blinks


# הקשר בין סנכרון לקבלת החלטות. מתחילים בהאם אפשר לנבא החלטה של מקשיב על ידי סנכרון לדובר.
# מיליון השוואות. אולי לא יצליח. גם קלסיפייר גם מיקס מודל אנובה. של זוג ושני סיפורים.
# ונגיד שקלו הקיצים - ולא הצלחנו.
# עוברים הלאה, זה יכול לנבוע מ-כך וכך.
# האם הייתה בחירה משותפת בהנתן שני הסיפורים.
# אם לא - אולי הקראה.
# צד אחד - מצמוצים
# צד שני - הלכנו לשמיעה.
# 