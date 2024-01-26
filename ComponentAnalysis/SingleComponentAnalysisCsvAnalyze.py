import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

A=0
B=1

def prepareSetfromDF(df):
    keys = ["mean_nose", "mean_eye", "mean_zygomaticus", "mean_corrugator", "var_nose", "var_eye", "var_zygomaticus", "var_corrugator"] #, "is_reading"]
    X = []
    Y = []
    for p in [A,B]:
        for i in range(p, len(df), 4):
            story1 = df.iloc[i]
            story2 = df.iloc[i+2]

            x_i = []
            x_i += [(story1[key] - story2[key]) for key in keys]
            X.append(x_i)
            y_i = int(story2["my_choice"]) # 0 if story1 is chosen, and 1 if story2 is chosen
            Y.append(y_i)
        
    return X,Y

data_path = r"C:\Liron\DataEmg\Done"
csv_path = os.path.join(data_path, "four_components_mean_and_choices_ReverseabsZscore.csv")

csvFile = pd.read_csv(csv_path)
df = pd.DataFrame(csvFile)

indexes = df["my_rt"] != 0
df = df[indexes]


X,Y = prepareSetfromDF(df)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=40)

classifier = SVC(kernel="poly")
classifier.fit(x_train, y_train)

y_test_predicted = classifier.predict(x_test)

print(classification_report(y_test, y_test_predicted))


# NOTHING!!!!