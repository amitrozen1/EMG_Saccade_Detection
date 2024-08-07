#from analysis_decisions_py.read_gyroscope_info import main
#from Analysis.cross_corr_components import main
#from Analysis.cross_corr_inertial import main
#from Analysis.plot_heatmap import main
# from Analysis.blinking_for_shoval import main
from Analysis.blinking_for_shoval import extract_main
#from Analysis.blink_detection import main
#from Analysis.mark_blinks import main
from Analysis.blink_mains import main, main_all_sessions, main_1
from General.init import data_path


if __name__ == '__main__':
   # main_all_sessions(data_path)
   main()
   # main_1()
   # vector, chunks = extract_main()
