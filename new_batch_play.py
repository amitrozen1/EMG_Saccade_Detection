#from analysis_decisions_py.save_new_w_and_plot_chosen_comps import main, create_html, create_html_for_new_batch, create_html_batch2
from Analysis.cross_corr_comp_with_decisions import main_batch2, main, main_3, main_corr_with_actress_dwt, get_main_corrolation_of_single_comps_headers
from General.init import *
#from analysis_decisions_py.save_new_w_and_plot_chosen_comps import do_single_session_no_review, main, create_html_after_dwt

from analysis_decisions_py.original_actress_readings import export_actress_chunked_data_with_dwt, export_actress_chunked_data

from analysis_decisions_py.plot_chosen_comps_in_reading import main_plot_read, main_nose_corrs



#main_with_actress_compare(True)
#main_batch2()
#main(data_path="C:\\Liron\\DataEmg\\Try")


#create_html_batch2()

#create_html_after_dwt()

#main_batch2()

#main()

#main_3()

#print(get_main_corrolation_of_single_comps_headers(True))
# #main_corr_with_actress_dwt()

# from analysis_decisions_py.plot_chosen_comps_in_reading import main_plot_read, main_check_pearson_to_comp
#main_plot_read()

# try with lowess, see if the correct 1-direct was applied or not to previous.

#main()
#create_html_after_dwt(data_path)

# export_actress_chunked_data()
# export_actress_chunked_data_with_dwt(5,50,False)

#todo - remove 8 coresponding electroe (i think 7 here)

#main_plot_read()

#main_3()
main_nose_corrs()

#main_corr_with_actress_dwt()