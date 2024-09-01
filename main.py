#from analysis_decisions_py.read_gyroscope_info import main
#from Analysis.cross_corr_components import main, add_mean_component_headers
#from Analysis.cross_corr_inertial import main
#from Analysis.plot_heatmap import main
# from Analysis.blinking_for_shoval import main
from blinking.blinking_for_shoval import extract_main
#from Analysis.blink_detection import main
#from Analysis.mark_blinks import main
from blinking.blink_mains import main, main_all_sessions, main_1, run_over_all_sessions, generate_simple_reading_excel, run_over_all_sessions_listening
from General.init import data_path


if __name__ == '__main__':
   # main_all_sessions(data_path)
   # run_over_all_sessions(data_path)
   # main()
   # main_1()
   # vector, chunks = extract_main()
   # generate_simple_reading_excel(data_path)
   run_over_all_sessions_listening(data_path)