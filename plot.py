import sys
from wbfluentpy.io.classes import ReportFilesOut,SolutionFiles
from filesystem import TableFileSystem
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
from itertools import zip_longest
import pandas as pd

RESIDUAL_NAMES = ['continuity' ,'x-velocity','y-velocity',  'z-velocity','energy','k','epsilon']
fig_size = (16,12)
FONTSIZE = 30
MULTI_FONTSIZE = 20
TICKSIZE = 20
MULTI_TICK_SIZE = 15
LEGEND_SIZE = 24
EXCLUDE_RESIDUAL = 'report-con'
ASPECT = 4.0/3.0

def line_plot(ax: axes,
              x: np.ndarray,
              y: np.ndarray,
              linewidth = 2,
              color = 'blue',
              linestyle = '-'):

    ax.plot(x,y,linewidth = linewidth,color = color,linestyle = linestyle)

def _residual_set_up(fs: TableFileSystem,
                     shape = None):

    if fs.solution_files is None:
        fs.map_solution_files()
    
    cols = [c for c in fs.solution_files.columns if EXCLUDE_RESIDUAL not in c]
    
    return _sizer(cols,shape = shape)

def _report_set_up(fs:TableFileSystem,
                   shape = None):
    
    if fs.report_files is None:
        fs.map_report_files()
    
    cols = [c for c in fs.report_files.columns if EXCLUDE_RESIDUAL not in c]
    
    return _sizer(cols,shape = shape)
        
def _sizer(cols: list,
            shape = None):

    if shape is not None:
        if shape[0]*shape[1] < len(cols):
            raise ValueError('Size of shape is to small to plot the number of residuals: {}'.format(len(cols)))
        
        nrows = shape[0]
        ncols = shape[1]
    
    else:
        ncols = int(np.floor((len(cols)*ASPECT)**0.5))
        nrows = int(np.ceil((1.0/ASPECT*ncols)))

    _,axes = plt.subplots(nrows = nrows,ncols = ncols,figsize = fig_size)
    
    return axes,cols



def plot_residuals(batch_folder: str,
                   filesys = TableFileSystem,
                   shared_x = False,
                   shared_y = False,
                   shape = None,
                   yscale = 'log',
                   *plot_args,
                   **plot_kwargs):

    fs = filesys(batch_folder)
    axes,cols = _residual_set_up(fs,shape = shape)

    for col,ax in zip_longest(cols,np.ravel(axes)):
        if col is None:
            ax.axis('off')
        else:
            df = fs.solution_files.get_variable(col)
            for c in df.columns:
                line_plot(ax,df.index,df[c],*plot_args,**plot_kwargs)

            if shared_x is False:
                ax.set_xlabel('Iteration Number',fontsize = MULTI_FONTSIZE)
            
            if shared_y is False:
                ax.set_ylabel(col,fontsize = MULTI_FONTSIZE)
            
            ax.set_yscale(yscale)
            ax.tick_params('both',labelsize = MULTI_TICK_SIZE)
    
    plt.tight_layout()
    return axes

def plot_residual_summary(batch_folder: str,
                                filesys = TableFileSystem,
                                shared_x = False,
                                shared_y = False,
                                shape = None,
                                ax = None,
                                yscale = 'log',
                                bound1 = np.min,
                                bound2 = np.max,
                                center = np.mean,
                                bound1_color = 'black',
                                bound2_color = 'black',
                                bound1_width = 4,
                                bound2_width = 4,
                                bound1_style = '--',
                                bound2_style = '--',
                                fill = True,
                                fill_color = 'blue',
                                fill_alpha = 0.5,
                                *plot_args,
                                **plot_kwargs):
    
    fs = filesys(batch_folder)
    fs.map_solution_files()
    axes,cols = _residual_set_up(fs,shape = shape)

    for col,ax in zip_longest(cols,np.ravel(axes)):
        if col is None:
            ax.axis('off')
        else:
            df = fs.solution_files.get_variable(col)
            make_residual_summary_plot('',col,df = df,ax = ax,
                                            shared_x = shared_x,
                                            shared_y = shared_y,
                                            yscale = yscale,
                                            bound1 = bound1,
                                            bound2 = bound2,
                                            center = center,
                                            bound1_color = bound1_color,
                                            bound2_color = bound2_color,
                                            bound1_width = bound1_width,
                                            bound2_width = bound2_width,
                                            bound1_style = bound1_style,
                                            bound2_style = bound2_style,
                                            fill = fill,
                                            fill_color = fill_color,
                                            fill_alpha = fill_alpha,
                                            fontsize= MULTI_FONTSIZE,
                                            ticksize= MULTI_TICK_SIZE,
                                            *plot_args,
                                            **plot_kwargs)
                
    plt.tight_layout()
    return axes

def plot_reports(batch_folder: str,
                diff = 0,
                absolute = True,
                normalized = False,
                filesys = TableFileSystem,
                shared_x = False,
                shared_y = False,
                shape = None,
                yscale = 'linear',
                *plot_args,
                **plot_kwargs):

    
    fs = filesys(batch_folder)
    fs.map_report_files()
    axes,cols = _report_set_up(fs,shape = shape)

    for col,ax in zip_longest(cols,np.ravel(axes)):
        if col is None:
            ax.axis('off')
        else:
            df = fs.report_files.get_variable(col)
            make_report_plot('',col,
                              diff = diff,
                              df = df,
                              ax =ax,
                              normalized= normalized,
                              absolute = absolute,
                              shared_x= shared_x,
                              shared_y = shared_y,
                              yscale = yscale,
                              ticksize= MULTI_TICK_SIZE,
                              fontsize= MULTI_FONTSIZE,
                              *plot_args,
                              **plot_kwargs)
    
    plt.tight_layout()
    return axes

def plot_report_summary(batch_folder: str,
                        diff = 0,
                        shape = None,
                        normalized = False,
                        absolute = True, 
                        filesys = TableFileSystem,
                        shared_x = False,
                        shared_y = False,
                        ax = None,
                        yscale = 'log',
                        bound1 = np.min,
                        bound2 = np.max,
                        center = np.mean,
                        bound1_color = 'black',
                        bound2_color = 'black',
                        bound1_width = 4,
                        bound2_width = 4,
                        bound1_style = '--',
                        bound2_style = '--',
                        fill = True,
                        fill_color = 'blue',
                        fill_alpha = 0.5,
                        *plot_args,
                        **plot_kwargs):

    fs = filesys(batch_folder)
    fs.map_report_files()
    axes,cols = _report_set_up(fs,shape = shape)

    for col,ax in zip_longest(cols,np.ravel(axes)):
        if col is None:
            ax.axis('off')
        else:
            df = fs.report_files.get_variable(col)
            make_report_summary_plot('',col,
                                            normalized = normalized,
                                            absolute = absolute,
                                            diff = diff,
                                            df = df,ax = ax,
                                            shared_x = shared_x,
                                            shared_y = shared_y,
                                            yscale = yscale,
                                            bound1 = bound1,
                                            bound2 = bound2,
                                            center = center,
                                            bound1_color = bound1_color,
                                            bound2_color = bound2_color,
                                            bound1_width = bound1_width,
                                            bound2_width = bound2_width,
                                            bound1_style = bound1_style,
                                            bound2_style = bound2_style,
                                            fill = fill,
                                            fill_color = fill_color,
                                            fill_alpha = fill_alpha,
                                            fontsize= MULTI_FONTSIZE,
                                            ticksize= MULTI_TICK_SIZE,
                                            *plot_args,
                                            **plot_kwargs)
    
    plt.tight_layout()
    return axes

def make_report_summary_plot(batch_folder: str,
                            name: str,
                            df = None,
                            diff = 0,
                            normalized = False,
                            absolute = True, 
                            filesys = TableFileSystem,
                            shared_x = False,
                            shared_y = False,
                            ax = None,
                            yscale = 'log',
                            bound1 = np.min,
                            bound2 = np.max,
                            center = np.mean,
                            bound1_color = 'black',
                            bound2_color = 'black',
                            bound1_width = 4,
                            bound2_width = 4,
                            bound1_style = '--',
                            bound2_style = '--',
                            fill = True,
                            fill_color = 'blue',
                            fill_alpha = 0.5,
                            fontsize = None,
                            ticksize = None,
                            *plot_args,
                            **plot_kwargs):


    if ticksize is None:
        ticksize = TICKSIZE
    if fontsize is None:
        fontsize = FONTSIZE
    
    if 'linewidth' not in plot_kwargs:
        plot_kwargs['linewidth'] = 6
    
    if ax is None:
        fig,ax = plt.subplots(figsize  = fig_size)
    
    if df is None:
        fs = filesys(batch_folder)
        fs.map_report_files()
        df = fs.report_files.get_variable(name)
    
    df = _handle_differencing(df,diff,absolute,normalized)

    ax = _summary_plot(ax,
                       df.index.to_numpy(),
                       df.to_numpy(),
                       bound1 = bound1,
                       bound2 = bound2,
                       center = center,
                       bound1_color = bound1_color,
                       bound2_color = bound2_color,
                       bound1_width = bound1_width,
                       bound2_width = bound2_width,
                       bound1_style = bound1_style,
                       bound2_style = bound2_style ,
                       fill = fill,
                       fill_color = fill_color,
                       fill_alpha = fill_alpha,
                       *plot_args,**plot_kwargs)

    if shared_x is False:
        ax.set_xlabel('Iteration Number',fontsize = fontsize)
    
    if shared_y is False:
        ax.set_ylabel(name,fontsize = fontsize)
    
    ax.set_yscale(yscale)
    ax.tick_params('both',labelsize = ticksize)
    
    return ax

def make_report_plot(batch_folder: str,
                    variable_name: str,
                    diff = 0,
                    absolute = True,
                    df = None,
                    normalized = False,
                    filesys = TableFileSystem,
                    shared_x = False,
                    shared_y = False,
                    ax = None,
                    yscale = 'linear',
                    fontsize = None,
                    ticksize = None,
                    *plot_args,
                    **plot_kwargs):
    
    if fontsize is None:
        fontsize = FONTSIZE
    if ticksize is None:
        ticksize = TICKSIZE
    
    if ax is None:
        fig,ax = plt.subplots(figsize = fig_size)
    
    if df is None:
        fs = filesys(batch_folder)
        fs.map_report_files()
        df = fs.report_files.get_variable(variable_name) 
       
    df = _handle_differencing(df,diff,absolute,normalized)
    _make_iterative_plot(df,variable_name,ax,
                         shared_x = shared_x,
                         shared_y = shared_y,
                         yscale = yscale,
                         fontsize = fontsize,
                         ticksize = ticksize,
                         *plot_args,
                         **plot_kwargs)
    
    return ax

def _handle_differencing(df: pd.DataFrame,
                         diff:int,
                         absolute:bool,
                         normalized:bool)-> pd.DataFrame:
    _df = None
    if diff != 0:
        _df = df.diff(periods = diff)
        if absolute:
            _df = _df.abs()
        if normalized:
            _df = _df/df.iloc[0:-1,:]
        
    if _df is None:
        return df
    else:
        return _df

def make_residual_plot(batch_folder: str,
                        residual_name: str,
                        filesys = TableFileSystem,
                        shared_x = False,
                        shared_y = False,
                        ax = None,
                        yscale = 'log',
                        *plot_args,
                        **plot_kwargs):


    """
    make a plot summarizing the residual specified by residual_name
    finding the residuals from the solution files specifed by the filesys class
    """

    if ax is None:
        fig,ax = plt.subplots(figsize = fig_size)
    
    fs = filesys(batch_folder)
    fs.map_solution_files()
    df = fs.solution_files.get_variable(residual_name)
    _make_iterative_plot(df,residual_name,ax,
                         shared_x = shared_x,
                         shared_y = shared_y,
                         yscale = yscale,
                         *plot_args,
                         **plot_kwargs)
    
    return ax


def _make_iterative_plot(df:pd.DataFrame,
                         name: str,
                         ax: axes,
                         shared_x = False,
                         shared_y = False,
                         yscale = 'log',
                         ticksize = None,
                         fontsize = None,
                         *plot_args,
                         **plot_kwargs):
    
    if fontsize is None:
        fontsize = FONTSIZE
    if ticksize is None:
        ticksize = TICKSIZE
    
    for c in df.columns:
        line_plot(ax,df.index,df[c],*plot_args,**plot_kwargs)
    
    if shared_x is False:
        ax.set_xlabel('Iteration Number',fontsize = FONTSIZE)
    
    if shared_y is False:
        ax.set_ylabel(name,fontsize = fontsize)
    
    ax.set_yscale(yscale)
    ax.tick_params('both',labelsize = ticksize)
    return ax

def make_residual_summary_plot(batch_folder: str,
                                residual_name: str,
                                df = None,
                                filesys = TableFileSystem,
                                shared_x = False,
                                shared_y = False,
                                ax = None,
                                yscale = 'log',
                                bound1 = np.min,
                                bound2 = np.max,
                                center = np.mean,
                                bound1_color = 'black',
                                bound2_color = 'black',
                                bound1_width = 4,
                                bound2_width = 4,
                                bound1_style = '--',
                                bound2_style = '--',
                                fill = True,
                                fill_color = 'blue',
                                fill_alpha = 0.5,
                                fontsize = None,
                                ticksize = None,
                                *plot_args,
                                **plot_kwargs):

    if ticksize is None:
        ticksize = TICKSIZE
    if fontsize is None:
        fontsize = FONTSIZE
    
    if 'linewidth' not in plot_kwargs:
        plot_kwargs['linewidth'] = 6
    
    if ax is None:
        fig,ax = plt.subplots(figsize  = fig_size)
    
    if df is None:
        fs = filesys(batch_folder)
        fs.map_solution_files()
        df = fs.solution_files.get_variable(residual_name)

    ax = _summary_plot(ax,
                       df.index.to_numpy(),
                       df.to_numpy(),
                       bound1 = np.min,
                       bound2 = np.max,
                       center = np.mean,
                       bound1_color = bound1_color,
                       bound2_color = bound2_color,
                       bound1_width = bound1_width,
                       bound2_width = bound2_width,
                       bound1_style = bound1_style,
                       bound2_style = bound2_style ,
                       fill = fill,
                       fill_color = fill_color,
                       fill_alpha = fill_alpha,
                       *plot_args,**plot_kwargs)

    if shared_x is False:
        ax.set_xlabel('Iteration Number',fontsize = fontsize)
    
    if shared_y is False:
        ax.set_ylabel(residual_name,fontsize = fontsize)
    
    ax.set_yscale(yscale)
    ax.tick_params('both',labelsize = ticksize)
    
    return ax

def _summary_plot(ax: axes,
                  x: np.ndarray,
                  Y: np.ndarray,
                  bound1 = np.min,
                  bound2 = np.max,
                  center = np.mean,
                  bound1_color = 'blue',
                  bound2_color = 'blue',
                  bound1_width = 2,
                  bound2_width = 2,
                  bound1_style = ':',
                  bound2_style = ':',
                  fill = True,
                  fill_color = 'blue',
                  fill_alpha = 0.5,
                  *plot_args,**plot_kwargs):
    
    b1 = bound1(Y,axis =1)
    b2 = bound2(Y,axis =1)
    cen = center(Y,axis = 1)

    for b,c,w,s in zip([b1,b2],[bound1_color,bound2_color],
                       [bound1_width,bound2_width],[bound1_style,bound2_style]):

        line_plot(ax,x,b,linestyle= s,linewidth= w,color =c)
        if fill:
            ax.fill_between(x,b,cen,alpha = fill_alpha,color = fill_color)
        
    line_plot(ax,x,cen,*plot_args,**plot_kwargs)

    return ax
