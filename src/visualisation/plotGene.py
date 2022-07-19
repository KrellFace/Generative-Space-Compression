import matplotlib.pyplot as plt
import os
import src.config.enumsAndConfig as enumAndConfig
import src.config.helperMthds as helper

#Scatter plot of compressed level data 
def plot_compressed_data(toplot, var_exp, col_names, file_name, gen_names=[], label_extremes = False):

    #col1name = compTyp.name + ' 1'
    #col2name = compTyp.name + ' 2'
    col1name = col_names[0]
    col2name = col_names[1]

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    
    ax.set_xlabel(col1name + ': ', fontsize = 15)
    ax.set_ylabel(col2name + ': ', fontsize = 15)
        
    title = os.path.basename(file_name)
    #Set title without .png     
    ax.set_title(title[0:len(title)-4], fontsize = 20)

    #Color each generators points differently if we are running for multiple alternatives
    if len(gen_names)>0:
        plot_col = 0
        for generator in gen_names:
            #Generate a random color for the generator
            rgb = enumAndConfig.color_dict[plot_col]
            plot_col+=1 
            #Limit our targets to just current generator
            to_keep = toplot['generator_name'] == generator
            ax.scatter(toplot.loc[to_keep, col1name]
                        , toplot.loc[to_keep, col2name]
                        , c = [rgb]
                        , alpha = 0.5
                        , s = 50)
    #For single generator
    else:
        ax.scatter(toplot[0].loc[:, col1name]
                    , toplot[0].loc[:, col2name]
                    , s = 20)       
    
    if(label_extremes):
        coord_dict = helper.return_coord_dict_fromcoord_lists(toplot.index, toplot[col1name].tolist(), toplot[col2name].tolist())
        extreme_coords_for_labeling = helper.get_extreme_coords(coord_dict, 10)

        for key in extreme_coords_for_labeling:
            ax.annotate(extreme_coords_for_labeling[key][0], (extreme_coords_for_labeling[key][1],extreme_coords_for_labeling[key][2] ))

    ax.legend(gen_names)
    ax.grid()
    #plt.show()
    plt.savefig(file_name)


#Basic scatter plot
def simple_scatter(frame, col1, col2, title):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 

    ax.set_xlabel(col1, fontsize = 15)
    ax.set_ylabel(col2, fontsize = 15)        
    ax.set_title(title , fontsize = 20)

    ax.scatter(frame.loc[:, col1]
                , frame.loc[:, col2]
                , s = 5)       
    ax.grid()
    plt.show()