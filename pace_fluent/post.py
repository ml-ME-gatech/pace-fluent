#native imports
import numpy as np
import os
import sys
import pandas as pd
from pathlib import WindowsPath,PosixPath
import shutil

from fluentpy.fluentio.classes import SurfaceIntegralFile
from fluentpy.tui.fluent import SurfaceIntegrals
from fluentpy.tui.util import _surface_construction_arg_validator

#package imports
from .submit import FluentBatchSubmission
from .filesystem import TableFileSystem
from fluentpy.tui import FluentEngine

"""
Author: Michael Lanahan
Date Created: 09.07.2021
Last Edit: 11.03.2021

Description:
It may be the case that after simulations are run, various post-processing is desired 
on the resulting cases or data. This suite allows for that in some limited ways
"""

#Some constants up here
WARNINGS = True
LINE_BREAK = '\n'
SURFACE_INTEGRAL_FILE_DELIM = '-'
SURFACE_INTEGRAL_EXT = '.srp'

FLUENT_CASE_EXT = '.cas'
FLUENT_DATA_EXT = '.dat'

#Set up the pathlib here based on the system we are on - since this code 
#will likely be used on both windows and posix systems need to make compatible with
#both
os_name = sys.platform
if os_name == 'win32' or os_name == 'win64':
    _Path = WindowsPath
elif os_name == 'linux' or os_name == 'posix':
    _Path = PosixPath
else:
    raise ValueError('Cannot determine Path structure from platform: {}'.format(os_name))

def combine_batches(batch_folder1: str,
                    batch_folder2: str,
                    third_folder = None,
                    verbose  = True) -> None:
    """
    utilitiy for combining batches after running them seperately
    """
    
    if third_folder is None:
        copy_flag = False
        third_folder = batch_folder1
    
    batch1 = FluentBatchSubmission.from_batch_cache(batch_folder1)
    batch2 = FluentBatchSubmission.from_batch_cache(batch_folder2)

    #make sure that there are no overlapping items here
    bfolders1 = {folder: os.path.join(batch_folder1,folder) for folder in batch1.submission_object.keys()}
    bfolders2 =  {folder: os.path.join(batch_folder2,folder) for folder in batch2.submission_object.keys()}
    
    for bf1 in bfolders1:
        if bf1 in bfolders2:
            raise ValueError('Cannot have overlap between folders for combining of batches')
    
    new_batch = batch1 + batch2
    new_batch.bash_submit(third_folder,purge = copy_flag)

    def _transfer_folders(batch_folders: list,
                          batch: FluentBatchSubmission) -> None:

        """
        copy all of the folders to their respective locations
        """
        submission_folders = {folder: os.path.join(third_folder,folder) for folder in batch.submission_object.keys()}
        for name,folder in batch_folders.items():
            if name in submission_folders:
                for file in os.listdir(folder):
                    shutil.copy2(os.path.join(folder,file),
                                 submission_folders[name])
            
            if verbose:
                _name, _ = os.path.split(submission_folders[name])
                _original,_ = os.path.split(folder)
                print('copied contents of folder: {} from batch: {} to batch folder: {}'.format(name,_original,_name))

        
    if copy_flag:
        _transfer_folders(bfolders1,new_batch)
   
    _transfer_folders(bfolders2,new_batch)

class SurfaceIntegralBatch:

    """
    Extends the SurfaceIntegrals to work with batch
    folder structures
    """

    def __init__(self,folder: str,
                      id: list,
                      variable: list,
                      surface_type: list,
                      engine = FluentEngine,
                      folders = [],
                      id_pad = 1,
                      **engine_kwargs):

        self.id,self.variable,self.surface_type  = \
             self._validate_constructor_args(id,variable,surface_type)

        self.fs = TableFileSystem(folder)
        self.engine = engine
        self.surface_integrals = {}
        self.df = None
        self.folders = folders
        self.engine_kwargs = engine_kwargs
        self.id_pad = id_pad

    @staticmethod
    def _validate_constructor_args(id: list,
                                   variable: list,
                                   surface_type: list) -> tuple:

        return _surface_construction_arg_validator(id,variable,surface_type)
    
    def collect_case_files(self):
        
        """
        collects all of the case files into a dictionary structure
        where the key is the folder where the case file is contained
        and the value is the file name
        """

        self.fs.map_submit_folders(check_contents = [FLUENT_DATA_EXT])
        case_files = {key: None for key in self.fs.submission_folder_list}
        for folder in self.fs.submission_folder_list:
            for f in folder.iterdir():
                if FLUENT_CASE_EXT in f.name:
                    case_files[folder] = f
        
        return case_files
    
    def _local_surface_integral_collection(self) -> dict:
        """
        function to collect the surface integrals locally i.e. 
        probably on a windows machine, and also invoking the 
        fluent engine to generate the file
        """
        case_files = self.collect_case_files()

        if self.folders:
            case_files = {folder:case_file for folder,case_file in case_files.items() 
                            if folder.name in self.folders}
        
        for folder,case_file in case_files.items():

            si = SurfaceIntegrals(case_file,self.id,
                                       self.variable,
                                       self.surface_type,
                                       engine = self.engine,
                                       id_pad = self.id_pad,
                                       **self.engine_kwargs)

            attr = [_si.read() for _si in si()]
            _,case = os.path.split(folder)
            self.surface_integrals[case] = attr
        
        return self.surface_integrals
    
    def _post_surface_integral_collection(self,name = None) -> dict:
        """
        function to collect the surface integrals locally on a windows machine
        and NOT invoking the fluent engine - this assumes of course that the
        file was generated at the end of a fluent case run
        """
        if name is None:
            name = self.fs.case_file.name
        
        self.fs.map_submit_folders(check_contents= [FLUENT_DATA_EXT])
        case_files = {key:os.path.join(key,name) 
                     for key in self.fs.submission_folder_list}
    
        #optional argument here to only look at folders that are provided
        #at class instiantation
        if self.folders:
            case_files = {folder:case_file for folder,case_file in case_files.items() 
                            if folder.name in self.folders}

        for folder,case_file in case_files.items():

            si = SurfaceIntegrals(case_file,self.id,
                                       self.variable,
                                       self.surface_type,
                                       engine = self.engine,
                                       id_pad = self.id_pad,
                                       **self.engine_kwargs)
           
            for key in si.delete:
                si.delete[key] = False
            
            attr = []
            for id,variable,surface_type in zip(self.id,self.variable,self.surface_type):
                sif = SurfaceIntegralFile(si.file_name(id,surface_type,variable))
                attr.append(sif.read())
            
            _,case = os.path.split(folder)
            self.surface_integrals[case] = attr
        
        return self.surface_integrals
    
    def collect_surface_integrals(self,
                                  run_fluent: bool,
                                  pace = False,
                                  name = None):
        
        if not pace:
            if run_fluent:
                return self._local_surface_integral_collection()
            else:
                return self._post_surface_integral_collection(name = name)
        
        else:
            raise NotImplementedError('havent implmemented methods for pace')

    
    def readdf(self, run_fluent = False,
                     name = None):
        """ 
        parse the collected surface integrals files into a dataframe

        if some files or entries are not found across files, these 
        values will be filled in with nan to avoid erroring out.

        good work, very elegant 11.3.2021 
        """
        
        #collect the data information i.e. the name of the data values
        #and the values themselves
        self.collect_surface_integrals(run_fluent,name = name)
        data_dict = {}
        
        for case,attrs in self.surface_integrals.items():
            unknown_num = -1
            data_dict[case] = {}
            for attr in attrs:
                #handle the occasional absence of the boundary
                if attr['boundary'] is not None:
                    for boundary,value in zip(attr['boundary'],attr['value']):
                        if boundary == 'Net':
                            boundary_name = 'Net: ' + ''.join([attr['boundary'][i] + '-' for i in range(len(attr['boundary'])-1)])
                        else:
                            boundary_name = boundary + '-'
                        name = attr['type'] + '-' + boundary_name  +  attr['name'] + ' ' + attr['unit']
                        data_dict[case][name] = value
                else:
                    name = attr['type'] + ':' + str(unknown_num) + ':'
                    name += attr['name'] + ' ' + attr['unit']
                    data_dict[case][name] = attr['value'][0] 
        
        #parse the columns -padding additonal values with nan into an array
        all_cols = list(set([dat for sublist in data_dict.values() for dat in sublist.keys()]))

        array = np.empty([len(self.surface_integrals),len(all_cols)])
        array[:] = np.nan
        index = []
        for i,(case,data) in enumerate(data_dict.items()):
            index.append(case)
            for j,col in enumerate(all_cols):
                try:
                    array[i,j] = data[col]
                except KeyError:
                    pass

        try:
            self.df = pd.DataFrame(array,index = pd.Series(index),
                                   columns = all_cols,dtype = float)
            return self.df
        except ValueError as v:
            raise ValueError("This can happen in the (rare) case if multiple values are in file when they are not supposed to be: {}".format(str(v)))
        
    
    


