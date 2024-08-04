#from analysis_decisions_py.read_gyroscope_info import main
#from Analysis.cross_corr_components import main, add_mean_component_headers
#from Analysis.cross_corr_inertial import main
#from Analysis.plot_heatmap import main
#from Analysis.blinking_for_shoval import main
#from analysis_decisions_py.save_new_w_and_plot_chosen_comps import go_manually_and_edit_chosen_components, main, create_html, do_single_session, plot_selected_components
from General.init import *
#from analysis_decisions_py.surrogate_decisions import main, main_one_dyad
#from analysis_decisions_py.validation_of_correlation_via_smile_anger import compute_pseudo_and_real_dyads_2,plot_boxplot, plot_sessions, check_best_window_lag, check_best_rms_ds
#from analysis_decisions_py.which_components_to_change import plot_sessions_and_decide_on_components, print_problematic_sessions
#from analysis_decisions_py.removing_artifacts import plot_couples_graph_and_xcorr
#from Analysis.cross_corr_comp_with_decisions import main
#from analysis_decisions_py.plot_chosen_comps_in_reading import main
# from Analysis.blink_detection import main
#from Analysis.cross_corr_inertial import main_accelerometer, main_gryo
#from Analysis.cross_corr_comp_with_decisions import main_2, main_rms
#from analysis_decisions_py.original_actress_readings import export_actress_chunked_data
from Analysis.cross_corr_comp_with_decisions import main_with_actress_compare

if __name__ == '__main__':
    main_with_actress_compare()



#generate random "left" or "right" string:



