import pandas as pd

# רשימת הערכים שאנחנו רוצים לבדוק
window_options = [5, 7, 9]
# רשימת הקבצים/הקריאות שלך (תעדכני לפי השמות אצלך)
readings = ["PA_Reading_2", "PA_Reading_13", "PB_Reading_14", "PB_Reading_15"]

results_list = []

for reading in readings:
    for win in window_options:
        # כאן את קוראת לפונקציית העיבוד הקיימת שלך עם הפרמטר המשתנה
        # למשל: saccades = process_reading(reading, smooth_win=win)
        
        # לצורך הדוגמה, נניח שהפונקציה מחזירה ספירה של סוגי סקאדות:
        h_count = get_horizontal_count(reading, win) 
        v_count = get_vertical_count(reading, win)
        total = h_count + v_count
        
        # חישוב היחס - ככל שהוא גבוה יותר, הזיהוי אופקי יותר (מתאים לקריאה)
        hv_ratio = h_count / v_count if v_count > 0 else h_count
        
        results_list.append({
            "Reading": reading,
            "Window": win,
            "Horizontal": h_count,
            "Vertical": v_count,
            "H:V Ratio": round(hv_ratio, 2),
            "Total": total
        })

# הצגת התוצאות בטבלה ברורה
df_results = pd.DataFrame(results_list)
print(df_results.sort_values(by=["Reading", "Window"]))