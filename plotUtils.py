# this script contain common plotting functions for the 6OHDA project
# code written by Dana Zemel
# last updated 10/9/2018 


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pptx
from pptx import Presentation 
from pptx.util import Inches
from io import BytesIO
import gc

def rosterPlot(ax, dff,dt,specing = 1, Color = None):
    #fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    t = np.linspace(0,dt*dff.shape[1],dff.shape[1])
    
    if Color==None:
       for d in range(0,dff.shape[0]):
           ax.plot(t,dff[d,:]+d*specing)
    elif isinstance(Color,str):
        for d in range(0,dff.shape[0]):
           ax.plot(t,dff[d,:]+d*specing,color=Color)
    elif isinstance(Color,list):
        if len(Color)<dff.shape[0]:
             Color[len(Color):dff.shape[0]]=[Color[-1] for x in range(len(Color),dff.shape[0])]
        for d in range(0,dff.shape[0]):
           ax.plot(t,dff[d,:]+d*specing,color=Color[d])
    else:
        if Color.shape[0]<dff.shape[0]:
             Color[-1:dff.shape[0],:]=Color[-1,:]
        for d in range(0,dff.shape[0]):
           ax.plot(t,dff[d,:]+d*specing,color=Color[d,:])

        
        
    



def PlotRelativeToOnset(ax,aligned,tPlot,Color='black',Label='',
                        linewidth=3,orizColor='darkred',orizLine=0.0,
                        orizStyle='dashed', Alpha = 0.1,mesErr = False):
    #this function takes the alligned data and plot it on plt axis 'ax'
    #TODO: sense checks on parameters... + hundle no oriz line
    #TODO: add paramas for legend and titles etc. 
    d = np.nanmean(aligned,axis=1)
    sd = np.nanstd(aligned,axis=1)
    if mesErr:
        sd = sd/np.sqrt(aligned.shape[1])
    
    ax.plot(tPlot,d,linewidth=3,color=Color,label=Label)
    ax.fill_between(tPlot, d-sd, d+sd,color=Color,alpha=Alpha)
    ax.axvline(x=orizLine,color=orizColor,linestyle=orizStyle)
    
def plt2pptx(slide,fig,left=Inches(0.93),top=Inches(1),width=None,height=None):
    # This function takes a matplotlib fig - save it 
    # to a StringIO and add to a slide in ppt usinng python-pptx
    # Inputs:
    #    slide - the pptx slide we want to add the image to 
    #    fig - matplotlib fig
    #    left - position from the left of slide
    #    top - position from the top of slide
    #    width - width of pic on the slide
    #    hight - hight of pic on slide
    # Output: 
    #    pic - the added pic to the ppt (just in case)

    
    image_stream = BytesIO()
    fig.savefig(image_stream,dpi=600, format='png')
    image_stream.seek(0)
    pic = slide.shapes.add_picture(image_stream, left, top,height=height,width=width)
 #   image_stream.truncate(0)
    image_stream.close()
    gc.collect()
    
    return pic
