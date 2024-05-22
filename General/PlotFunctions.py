import matplotlib.pyplot as plt
import numpy as np

A=0
B=1

def fixTicks(ticks):
    if ( len(ticks) == 0 ):
        return ticks
    s = ticks[0]
    return [ x - s for x in ticks]

def plot(y_val, X, totalSize, index, title, color, ticks): #, maximun_val):
    plot = plt.subplot2grid(totalSize, index, rowspan=2, colspan=2)
    plot.plot(X,y_val, color = color)
    plot.set_title(title)
    if ( len(ticks) > 0 ):
        for tick in ticks:
            plot.axvline(tick, linestyle='--')
    # plt.gca().axes.get_xaxis().set_visible(False)
    plot.axes.get_xaxis().set_visible(False)


def plotChunkData(X, title, ticks = [], subtitles = None, num_components = 16, show=True):
    y_axis = X
    x_axis = range(0, y_axis.shape[1])
    total_size = (num_components*2,2)
    #maximun_val = np.max(y_axis) # much better
    if subtitles is None:
        subtitles = ["component %d" % i for i in range(0,num_components)]
    
    for i in range(0,num_components):
        plot(y_axis[i], x_axis, total_size, (2*i, 0), subtitles[i], 'r', ticks)
    plt.suptitle(title)
    if(show):
        plt.show()


def plot_chunk_data_one_tick_per_component(X, title, ticks = [], subtitles = None, num_components = 16):
    y_axis = X
    x_axis = range(0, y_axis.shape[1])
    total_size = (num_components*2,2)
    #maximun_val = np.max(y_axis) # much better
    if subtitles is None:
        subtitles = ["component %d" % i for i in range(0,num_components)]
    
    for i in range(0,num_components):
        plot(y_axis[i], x_axis, total_size, (2*i, 0), subtitles[i], 'r', [ticks[i]])
    plt.suptitle(title)
    #plt.show()


def plotDataFromBothChunks(X1, X2, ticks1=[], ticks2=[], title="smiles", num_components = 16, saveAt="", importance_order=None):
    total_graph_size = (num_components*2,4)

    ticks1 = fixTicks(ticks1)
    ticks2 = fixTicks(ticks2)
    
    if importance_order is None:
        importance_order = [np.arange(num_components), np.arange(num_components)]

    for index in range(num_components):
        indexA = importance_order[A][index]
        indexB = importance_order[B][index]

        y1 = X1[indexA]
        y2 = X2[indexB]
        plot(y1, range(0,y1.shape[0]), total_graph_size,(index*2, 0), "component %d" % indexA, 'r', ticks1 )
        plot(y2, range(0,y2.shape[0]), total_graph_size,(index*2, 2), "component %d" % indexB, 'b', ticks2 )
        #index = index+1
        
    plt.suptitle(title)
    if(saveAt != ""):
        plt.savefig(saveAt)
    else:
        plt.show()

def plotHeatMapsForComponents(components, title=""):
    # Create figure and axis objects
    N = components.shape[0]
    fig, axs = plt.subplots(nrows=N, ncols=1, figsize=(6, 10))

    # Loop over rows and create histogram for each row
    for i in range(N):
        axs[i].hist(components[i], color='blue', alpha=0.5)
        axs[i].set_title('Histogram of Row {}'.format(i+1))
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')

    # Adjust spacing between subplots
    fig.tight_layout()
    plt.title(title)
    # Display plot
    #plt.show()

def clearPlot():
    plt.cla()
    #plt.clf()

def savePlot(path):
    plt.savefig(path, bbox_inches='tight')
