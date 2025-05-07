#from analysis_decisions_py.read_gyroscope_info import main
#from Analysis.cross_corr_components import main, add_mean_component_headers
#from Analysis.cross_corr_inertial import main
#from Analysis.plot_heatmap import main
# from Analysis.blinking_for_shoval import main
#from blinking.blinking_for_shoval import extract_main
#from Analysis.blink_detection import main
#from Analysis.mark_blinks import main
#from blinking.blink_mains import main, main_all_sessions, main_1, run_over_all_sessions, generate_simple_reading_excel, run_over_all_sessions_listening
#from General.init import *
#from Analysis.analysis_runner import compute_neg_pos_components_histogram_csv
#from analysis_decisions_py.surrogate_decisions_2 import main, gen_heatmap
from revision_analysis.data_loading import validate_classification, load_dyad_data, load_actress_data, check_actress_smile_corr
from revision_analysis.run_analysis import run_analysis
from revision_analysis.convert_excel_to_r import export_reading_to_r, export_listening_to_r, export_both_listeners_to_r, export_reading_for_combinations_to_r
from General.init import *
#from revision_analysis.check_actress_comps_against_vid import main
from revision_analysis.dyads_from_previous_version import main
import pandas as pd

if __name__ == '__main__':
   # main_all_sessions(data_path)
   # run_over_all_sessions(data_path)
   # main()
   # main_1()
   # vector, chunks = extract_main()
   # generate_simple_reading_excel(data_path)
   #run_over_all_sessions_listening(data_path)
   #compute_neg_pos_components_histogram_csv()
   #main()
   #gen_heatmap()
   #validate_classification("123")
   #res = load_actress_data(50,5)
   #both_comps, both_chunks = load_dyad_data("04122022_1545", 5, 50)
   #exit(0)
   rms = 50
   ds = 5
   window = 3000
   max_lag = 3000
   action = "reading"
   #csv_out_path = fr"C:\Liron\DataEmg\revision_analysis\{action}_rms_{rms}_ds_{ds}_window_{window}_max_lag_{max_lag}_moreActivationInfo_batch{current_batch}.csv"
   
   for window in [1000,2000]:
      for max_lag in [4000, 5000]:
         dfs = []
         csv_out_path = fr"C:\Liron\DataEmg\revision_analysis\{action}_rms_{rms}_ds_{ds}_window_{window}_max_lag_{max_lag}_no_pre500_tInc5_combined.csv"
         for i in [1,2]:
            set_current_batch(i)
            dfs.append(run_analysis(rms, ds, window, max_lag, csv_out_path=None, action=action))
         df = pd.concat(dfs, ignore_index=True)
         df.to_csv(csv_out_path, index=False)
         export_reading_to_r(csv_out_path)

   #export_reading_to_r(r"C:\Liron\DataEmg\revision_analysis\final_files\reading_rms_50_ds_5_window_3000_max_lag_3000_combined - replace_2004.csv")
   #export_listening_to_r(r"C:\Liron\DataEmg\revision_analysis\final_files\listening_rms_50_ds_5_window_3000_max_lag_3000_with_smile_corr_combined.csv")

   #export_both_listeners_to_r(r"C:\Liron\DataEmg\revision_analysis\listening_participants_rms_50_ds_5_window_3000_max_lag_3000_combined.csv")
   #check_actress_smile_corr()
   #main()

   