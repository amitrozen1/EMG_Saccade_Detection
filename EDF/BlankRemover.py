import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, ttest_ind_from_stats, mannwhitneyu
from EDF.SharedIcaRunner import LoadComponents, LoadComponentsWithRegularRms
from General.PlotFunctions import plotHeatMapsForComponents, clearPlot, savePlot   
from General import HtmlTablePlot
from General.HelperFunctions import particiapnt_to_char, RemoveRowFromMatrix
from EDF.ComponentValidator import GetNoiseComponentsBySort
from General.PlotFunctions import plotDataFromBothChunks
from scipy.stats import f

from scipy.stats import chi2


def shouldAnalyzeCurrentWindow(window):
    max_window_val = np.max(window, axis=1)
    min_window_val = np.min(window, axis=1)
    return np.std(window, axis=1) > 0.1*(max_window_val-min_window_val)


def compare_variances_chi(array1, array2, alpha=0.05):
    # Calculate the variances of the two arrays
    v1 = np.var(array1, ddof=1)
    v2 = np.var(array2, ddof=1)
    
    # Calculate the ratio of the larger variance to the smaller variance
    R = max(v1, v2) / min(v1, v2)
    
    # Calculate the F-statistic
    F = v1 / v2 if v1 > v2 else v2 / v1
    
    # Calculate the degrees of freedom
    df1 = len(array1) - 1
    df2 = len(array2) - 1
    
    # Calculate the critical value of chi-squared
    crit_val = chi2.ppf(1 - alpha/2, df1, df2)
    
    # Compare the calculated chi-squared statistic to the critical value of chi-squared
    return F<=crit_val


def compare_comp_variance_chi(analyzerComponents, windowComponents):
    return [compare_variances_chi(analyzerComponents[i], windowComponents[i]) for i in range(analyzerComponents.shape[0])]

def compare_standard_deviations(array1, array2, alpha=0.05):
    # Calculate the standard deviations of the two arrays
    s1 = np.std(array1, ddof=1)
    s2 = np.std(array2, ddof=1)
    
    # Calculate the ratio of the larger standard deviation to the smaller standard deviation
    R = max(s1, s2) / min(s1, s2)
    
    # Calculate the F-statistic
    F = (s1 ** 2) / (s2 ** 2)
    
    # Calculate the degrees of freedom
    df1 = len(array1) - 1
    df2 = len(array2) - 1
    
    # Calculate the critical value of F
    crit_val = f.ppf(1 - alpha/2, df1, df2)
    
    # Compare the calculated F-statistic to the critical value of F
    return F<=crit_val


def compare_components_std_F(analyzerComponents, windowComponents):
    return [compare_standard_deviations(analyzerComponents[i], windowComponents[i]) for i in range(analyzerComponents.shape[0])]
            

A = 0 
B = 1

def does_components_have_close_std(analyzerComponents, windowComponents):
    std_a = np.std(analyzerComponents, axis = 1)
    std_w = np.std(windowComponents, axis = 1)
    return np.abs(std_a-std_w)/std_a < 0.1


def AreFarFromMeanByMoreThanStd(analyzerComps, windowComps):
    m_a = np.mean(analyzerComps, axis = 1)
    m_w = np.mean(windowComps, axis = 1)
    std_a = np.std(analyzerComps, axis = 1)
    return np.abs(m_a - m_w) < std_a


def isCompEqual_permutation(comp1, comp2, alpha = 0.05):
    # Compute the observed difference in means
    obs_diff = np.mean(comp1) - np.mean(comp2)

    # Concatenate the two populations
    concat_comps = np.concatenate([comp1, comp2])

    # Set the number of bootstrap samples
    num_bootstraps = 10000

    # Initialize an empty array to store the differences in means
    boot_diffs = np.zeros(num_bootstraps)

    # Perform the bootstraps
    for i in range(num_bootstraps):
        # Resample from the concatenated population with replacement
        resample = np.random.choice(concat_comps, size=len(concat_comps), replace=True)
        
        # Split the resampled population into two groups
        boot_comp1 = resample[:len(comp1)]
        boot_comp2 = resample[len(comp1):]
        
        # Compute the difference in means for the resampled groups
        boot_diffs[i] = np.mean(boot_comp1) - np.mean(boot_comp2)

    # Compute the p-value
    p = np.sum(np.abs(boot_diffs) >= np.abs(obs_diff)) / num_bootstraps

    return p

def AreCompsEqual_permutation(comps1, comps2):
    p_vals = []
    for i in range(comps1.shape[0]):
        p_vals.append(isCompEqual_permutation(comps1[i], comps2[i]))
    return np.array(p_vals)


def AreComponentsEqual(components, baselines):
    _, p_vals = mannwhitneyu(components, baselines, axis=1 )
    return p_vals

def AreComponentsCloseTpMean_t(components, meanVector):
    t_statistic_array, p_value_array = ttest_1samp(components, meanVector, axis=1)
    return p_value_array

def AreAllComponentsCloseToMean_T(components, meanVector, alpha=0.01):
    return np.all(AreComponentsCloseTpMean_t(components, meanVector) > alpha )



class EmgBlankRemover: 
    def  __init__(self, independentComponents, chunks):
        self.blank_mean = []
        self.blank_std = []
        self.initBlank(chunks, independentComponents)

    def initBlank(self,chunks, independentComponents):
        blink_tag = "blink__{0}"
        for participant in [A,B]:
            blinkComponentsValues = None
            for i in range(0,3):
                chunk = chunks[participant][str.format(blink_tag, i)]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                chunk_data = independentComponents[participant][: , chunk_start:chunk_end]
                if ( i == 0 ):
                    blinkComponentsValues = chunk_data
                else:
                    blinkComponentsValues = np.append(blinkComponentsValues, chunk_data, axis=1)

            self.blank_mean.append(np.mean(blinkComponentsValues, axis=1))
            self.blank_std.append(np.std(blinkComponentsValues, axis=1))
    
    def shouldAnalyzeCurrentWindow(self, window_to_check): 
        AreComponentsCloseToBlink = self.shouldAnalyzeCurrentWindow_byParticiapnt(window_to_check)
        return np.any(AreComponentsCloseToBlink)
    
    def shouldAnalyzeCurrentWindow_byParticiapnt(self, window_to_check):
        # this function gets a window to analyze and returns a list of booleans
        # if the i_th place is True , than for the i_th component, the window is significantly
        # different than our "zero" activity - which is just blinking.
        # there are two cases that I can think of - returning which component to omit
        # or, omitting all compoenents when all components are not significant.

        AreComponentsCloseToBlink = []
        for p in [A,B]:
            distFromBlinkMean = np.abs(np.mean(window_to_check[p], axis=1) - self.blank_mean[p]) < self.blank_std[p]
            AreComponentsCloseToBlink.append(np.all(distFromBlinkMean))
        return not AreComponentsCloseToBlink

    def shouldAnalyzeCurrentWindow_byComponent(self, one_array_to_check, i):
        areBothCloseToBlink = []
        for p in [A,B]:
            check = np.abs(np.mean(one_array_to_check[p]) - self.blank_mean[p][i]) < self.blank_std[p][i]
            check2 = np.abs(np.mean(one_array_to_check[p]) / self.blank_mean[p][i]) < 1.1

            areBothCloseToBlink.append(check2)
        return not np.all(areBothCloseToBlink)

    def shouldAnalyzeCurrentWindow_singleP_allComponents(self, window_to_check, particiapnt):
        np.abs(np.mean(window_to_check, axis=1) - self.blank_mean[particiapnt]) >= self.blank_std[particiapnt]

    def shouldAnalyzeCurrentWindowSingleParticiapnt(self, window_to_check, particiapnt):
        return not np.all(np.abs(np.mean(window_to_check, axis=1) - self.blank_mean[particiapnt]) < self.blank_std[particiapnt])

    def shouldAnalyzeCurrentWindowTtest(self, window_to_check, alpha = 0.01): 
        AreComponentsCloseToBlink = [AreAllComponentsCloseToMean_T(window_to_check[p], self.blank_mean[p]) for p in [A,B]]
        return not np.all(AreComponentsCloseToBlink)

    def shouldAnalyzeCurrentWindow_stdCheck(self, window_to_check, alpha = 0.01): 
        AreComponentsCloseToBlink = [AreFarFromMeanByMoreThanStd(window_to_check[p], self.blank_mean[p]) for p in [A,B]]
        return not np.all(AreComponentsCloseToBlink)

def loadSession(session, rmsMsWindow = 50):
    data_path = r"C:\Liron\DataEmg\Done" 
    independentComponents, chunks = LoadComponents(data_path, session, rmsMsWindow) # 20HZ
    rows_to_remove = GetNoiseComponentsBySort(data_path, session)
    for p in [A,B]:
        rows_removed = 0
        for row in rows_to_remove:
            # comes sorted ? 
            independentComponents[p] = RemoveRowFromMatrix(independentComponents[p] , row - rows_removed)
            rows_removed = rows_removed + 1 #so you delete the right row.
    
    return independentComponents, chunks


if __name__ == "__main__":
    rmsMsWindow = 10
    independentComponents, chunks = loadSession("20122022_0830", rmsMsWindow)
    br = EmgBlankRemover(independentComponents, chunks)
    titles = [x + "__{0}" for x in ["smile", "angry", "blink"]]
    components = [[],[]]
    triggers_times = [[],[]]
    for p in [A,B]:
        for i in range(3):
            for title in titles:
                chunk = chunks[p][str.format(title,i)]
                triggers_times[p].append(int(chunk.Start.Time))
                triggers_times[p].append(int(chunk.End.Time))
    
    s = [triggers_times[A][0], triggers_times[B][0]]
    e = [triggers_times[A][-1], triggers_times[B][-1]]
    #plotDataFromBothChunks(independentComponents[A][:, s_a:e_a], independentComponents[B][:, s_b:e_b], triggers_times[A], triggers_times[B], "comps", 15)
    #exit()

    for p in [A,B]:
        for window in range(s[p], e[p], 1000//rmsMsWindow):
            analyze_bools = br.shouldAnalyzeCurrentWindow_singleP_allComponents(independentComponents[p][:, window:window+(1000//rmsMsWindow)],p)
            #shouldAnalyzeCurrentWindow(independentComponents[p][:, window:window+(1000//rmsMsWindow)])

            for i in range(len(analyze_bools)):
                if ( not analyze_bools[i]):
                    independentComponents[p][i, window:window+1000//rmsMsWindow] = np.zeros_like(independentComponents[p][i, window:window+(1000//rmsMsWindow)])

    plotDataFromBothChunks(independentComponents[A][:, s[A]:e[A]], independentComponents[B][:, s[B]:e[B]], triggers_times[A], triggers_times[B], "comps", 15)
    exit()

    #zero what's close to blink.
    # for p in [A,B]:
    #     start = triggers_times[p][0]
    #     end = triggers_times[p][-1]
    #     numPixelInSec = 1000//rmsMsWindow
    #     for i in range(start, end, numPixelInSec):
    #         window = independentComponents[p][:, i:i+numPixelInSec]
    #         if (not br.shouldAnalyzeCurrentWindowSingleParticiapnt(window, p)):
    #             independentComponents[p][:, i:i+numPixelInSec] = np.zeros_like(window)
    
    key = "Listening_6.ogg_2"
    chunkA = chunks[A][key]
    chunkB = chunks[B][key]

    numPixelInSec = 1000//rmsMsWindow
    s_a = int(chunkA.Start.Time)
    s_b = int(chunkB.Start.Time)
    e_a = s_a + numPixelInSec
    e_b = s_b + numPixelInSec

    components[A] = independentComponents[A][:, s_a : e_a]
    components[B] = independentComponents[B][:, s_b : e_b]
    # num_components = independentComponents[0].shape[0]
    # plotDataFromBothChunks(independentComponents[A][:, start:end], independentComponents[B][:, start:end], triggers_times[A], triggers_times[B], "RemoveBlank", num_components)
    # plotDataFromBothChunks(independentComponents[A][:, chunkA.Start.Time:chunkA.End.Time], independentComponents[B][:, chunkB.Start.Time:chunkB.End.Time], [],[], "RemoveBlank", num_components)

    
    # print(br.shouldAnalyzeCurrentWindow(components))
    # print(br.shouldAnalyzeCurrentWindowTtest(components))




    
    analyzer_comps = []
    for i in range(0,3):
        for participant in [A,B]:
            chunk = chunks[participant][f"blink__{i}"]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data = independentComponents[participant][: , chunk_start:chunk_end]
            if ( i == 0):
                analyzer_comps.append(chunk_data)
            else:
                analyzer_comps[participant] = np.append(analyzer_comps[participant], chunk_data, axis=1)



    # p= A
    # for i in 

    d = HtmlTablePlot.initHtmlDictionary(["particiapnt","method_0", "stats_0","method_1", "stats_1","method_2", "stats_2",  "hist_analyzer", "hist_comps"])

    for p in [A,B]:
        bool_tbl_U = AreComponentsEqual(analyzer_comps[p], components[p]) > 0.05
        bool_tbl_perm = AreCompsEqual_permutation(analyzer_comps[p], components[p]) > 0.05
        stdist_tbl = AreFarFromMeanByMoreThanStd(analyzer_comps[p], components[p])
        std_equal_tbl = does_components_have_close_std(analyzer_comps[p], components[p])
        var_chi = compare_comp_variance_chi(analyzer_comps[p], components[p])
        std_F = compare_components_std_F(analyzer_comps[p], components[p])
        title_analyzer_comps = f"analyzer_comps_{particiapnt_to_char(p)}.jpg"
        plotHeatMapsForComponents(analyzer_comps[p], title_analyzer_comps)
        savePlot(title_analyzer_comps)
        clearPlot()
        title_comps = f"components_{particiapnt_to_char(p)}.jpg"
        plotHeatMapsForComponents(components[p], title_comps)
        savePlot(title_comps)
        HtmlTablePlot.addRow(d, [HtmlTablePlot.toTextHtmlCol(particiapnt_to_char(p)), HtmlTablePlot.toTextHtmlCol("std_f"), HtmlTablePlot.toTextHtmlCol(std_F),
                                 HtmlTablePlot.toTextHtmlCol("var_chi"), HtmlTablePlot.toTextHtmlCol(var_chi),
                                 #HtmlTablePlot.toTextHtmlCol("stdist"), HtmlTablePlot.toTextHtmlCol(stdist_tbl),
                                 HtmlTablePlot.toTextHtmlCol("std_equal"), HtmlTablePlot.toTextHtmlCol(std_equal_tbl), 
                                 HtmlTablePlot.toImageHtmlColFromPath(title_analyzer_comps), HtmlTablePlot.toImageHtmlColFromPath(title_comps)])
        clearPlot()

    HtmlTablePlot.SaveHtmlFile(d, "blankRemoverTest_blink_U+perm_equal_std.html")