#native imports
import numpy as np
from abc import ABC,abstractmethod,abstractproperty
import os
import subprocess
import sys
import pandas as pd
from pathlib import WindowsPath,PosixPath
import shutil

#package imports
from .submit import FluentBatchSubmission, BatchSubmissionSummary
from .fluent import CaseDataReader
from .filesystem import TableFileSystem
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
EXIT_CHAR = 'q' + LINE_BREAK
EXIT_STATEMENT = 'exit'
WINDOWS_FLUENT_INIT_STATEMENT = 'fluent {} -t{} -g -i {} -o {}'
NAME_SPLIT_CHAR = '-'
SURFACE_INTEGRAL_FILE_DELIM = '-'
SURFACE_INTEGRAL_EXT = '.srp'
FLUENT_INPUT_NAME = 'input.fluent'
FLUENT_OUTPUT_NAME = 'output.fluent'
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

class PostEngine:

    """
    main class for the post processing engine using fluent
    """
    def __init__(self,file: str,
                      specification = '3ddp',
                      num_processors = 1,
                      reader = CaseDataReader):
        
        self.path,file_name = os.path.split(file)
        self.spec = specification
        self.__num_processors = num_processors
        self.__input = reader(file_name)
        self._additional_txt = ''
        self.input_file = os.path.join(self.path,FLUENT_INPUT_NAME)
        self.output_file = os.path.join(self.path,FLUENT_OUTPUT_NAME)
    

    def insert_text(self,other):
        self._additional_txt += other
    
    @property
    def num_processors(self):
        return str(self.__num_processors)

    def _fluent_initializer(self,
                            system = 'windows'):
        
        if system == 'windows':
            return WINDOWS_FLUENT_INIT_STATEMENT.format(self.spec,
                                                        self.num_processors,
                                                        FLUENT_INPUT_NAME,
                                                        FLUENT_OUTPUT_NAME) + EXIT_CHAR
    

    @property
    def input(self):
        return str(self.__input)
    
    def format_call(self):
        """
        format the text for the call, and also write the input file for 
        fluent
        """
        call_text = self._fluent_initializer()
        self.format_input_file()
        return call_text
    
    def format_input_file(self) -> None:
        """
        format the input file to fluent to read and create the surface integral
        """
        txt = self.input + LINE_BREAK
        txt += self._additional_txt + LINE_BREAK
        txt += EXIT_STATEMENT + LINE_BREAK

        with open(self.input_file,'w') as file:
            file.write(txt)
    
    def clean(self):
        """ 
        clean directory from input and output files
        """
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
        
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        
    def __call__(self):
        """
        This call does the following:
        (1) cleans the directory of the fluent case file from input and output files
        (2) formats the call
        (3) opens fluent and submits commands to fluent
        (4) cleans up the directory again
        """

        self.clean()
        txt = self.format_call()
        cwd = os.getcwd()
        os.chdir(self.path)
        process = subprocess.call(txt)
        os.chdir(cwd)
        self.clean()
        return process

class SurfaceIntegrals:
    """ 
    base class for all surface integrals that can be generated using 
    fluent. Hooks into an engine and executes upon inserting commands

    Instatation Arguments
    ---------------------
    file: str - string of the file to apply the surface integrals to
    id: str - the string of the boundary condition identification
    variable: str - the name of the variable to compute the surface integral for
    surface_type: str - the type of surface integral - i.e. area-weighted-avg
    engine: default - PostEngine - the engine to open fluent with
    """
    _prefix = '/report/surface-integrals/{}'

    def __init__(self,file:str,
                      id: str,
                      variable: str,
                      surface_type: str,
                      engine = PostEngine,
                      id_pad = 1,
                      **engine_kwargs
                  ):
        
        #attempt to instantiate the engine - if the engine isn't callable
        #then assume that this is being used at the end of computation
        try:
            self.engine = engine(file,**engine_kwargs)
        except TypeError:
            self.engine = None
        
        self.file = file
        self.id,self.variable,self.surface_type  = \
             self._validate_constructor_args(id,variable,surface_type)
        
        file_names = [self._generate_file_name(self.file,st,id,var) for st,id,var 
                      in zip(self.surface_type,self.id,self.variable)]

        self.delete = {fname: True for fname in file_names}
        self.id_pad = id_pad

    @staticmethod
    def _validate_constructor_args(id: list,
                                   variable: list,
                                   surface_type: list) -> tuple:

        return _surface_construction_arg_validator(id,variable,surface_type)
    
    def prefix(self,surface_type: str):
        return self._prefix.format(surface_type)
    
    def file_name(self,id: str,
                       surface_type: str,
                       variable: str):
        """
        get the filename to write the surface integral to
        """
        fname = self._generate_file_name(self.file,
                                         surface_type,
                                         id,
                                         variable)
        
        if self.delete[fname]:
            self.delete[fname] = False
            if os.path.exists(fname):
                os.remove(fname)
            
        return fname
    
    def format_text(self):
        """
        format the text for the call to the surface integral here
        """

        txt = ''
        
        shortest_id = min([len(id) for id in self.id])
        for ids,variable,surface_type in zip(self.id,self.variable,self.surface_type):
            txt += self.prefix(surface_type) + LINE_BREAK
            for id in ids:
                txt += id + LINE_BREAK
            
            
            if len(ids) < shortest_id + self.id_pad:
                diff = len(ids) - shortest_id
                for _ in range(self.id_pad - diff):
                    txt += ' , ' + LINE_BREAK 
            else:
                txt += ' , ' + LINE_BREAK
            
            txt += variable + LINE_BREAK
            txt += 'yes' + LINE_BREAK
            _,_file = os.path.split(self.file_name(ids,surface_type,variable))
            txt += _file + LINE_BREAK
        
        return txt

    def __call__(self,
                 return_engine = False):

        #enables the post module to simply be passed as txt
        #if the engine is not callable
        if self.engine is not None:

            self.engine.insert_text(self.format_text())
            engine_ouput = self.engine()
            sif = []
            for ids,variable,surface_type in zip(self.id,self.variable,self.surface_type):
                sif.append(SurfaceIntegralFile(self.file_name(ids,surface_type,variable)))

            if return_engine:
                return sif,engine_ouput
            else:
                return sif
        else:
            return self.format_text()

    @staticmethod
    def _generate_file_name(file: str,
                            surface_type: str,
                            ids: list,
                            variable: str) -> str:
        
        _ids = ''.join([id + SURFACE_INTEGRAL_FILE_DELIM for id in ids])[0:-1]

        path,_file = os.path.split(file)
        _file = os.path.splitext(_file)[0]

        write_file = ''.join([item + SURFACE_INTEGRAL_FILE_DELIM for item in [_file,surface_type,_ids,variable]])[0:-1]
         
        return os.path.join(path,write_file)

class SurfaceIntegralFile:
    """
    python class representation of the file created by a surface report
    as of 10.01.2021 now supports multiple surfaces but not multiple variables per surface
    """
    def __init__(self,file: str):
        
        self.path = _Path(file)
        self.attributes = {'type':None,
                           'name':None,
                           'unit':None,
                           'value':[],
                           'boundary': [],
                          }
    
    def _parse_txt(self,lines: list) -> None:
        """
        parse the text from the surface integrals file (native format from fluent)
        into attributes descibed by the "attributes" dictionary property. 

        configured to read multiple surfaces but not multiple variables (is this possible?)
        """
        #this first line here reads the header information on the file
        try:
            self.attributes['type'] =  lines[2].strip()
            self.attributes['name'],self.attributes['unit'] = self._space_delimited_line_split(lines[3])
            for i in range(5,len(lines)):
                try:
                    boundary,value = self._space_delimited_line_split(lines[i])
                    if boundary == 'Net':
                        self.attributes['boundary'].append(boundary)
                        self.attributes['value'].append(float(value))
                        break
                    elif '----' in boundary and '----' in value:
                        pass
                    else:
                        self.attributes['boundary'].append(boundary)
                        self.attributes['value'].append(float(value))
                except ValueError:
                    pass
        except IndexError:
            pass

    @staticmethod
    def _space_delimited_line_split(line: str) -> list:
        #convinience function for parsing specific kinds of text
        _line = line.strip()
        items = [i for i in _line.split('  ') if i != '']
        return items


    def read(self) -> dict:
        """
        read a surface integral file - as of 09.08.2021 only configured to read
        surface integral files containing 1 summary statistic
        """
        with open(str(self.path.resolve()),'r') as file:
            txt = file.readlines()
            self._parse_txt(txt)
        
        return self.attributes

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
        len_folders = len(batch_folders)
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
                      engine = PostEngine,
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
        
        
def sort_list_of_lists_by_list_len(input_list: list) -> list:

    list_len = [len(inner) for inner in input_list]
    permutation = sorted(range(len(list_len)), key = lambda t: list_len[t])

    return permutation

def apply_permutation_to_list(input_list: list,
                              permutation: list) -> list:

    return [input_list[i] for i in permutation]

def _surface_construction_arg_validator(id: list,
                                        variable: list,
                                        surface_type: list) -> tuple:

    """
    static function meant to validate the construction arguments
    also converts all of the arguments
    id,variable,surface_type 

    into lists by default so that multiple evaluations may be made with a single
    fluent engine call. If the input is a str for each of these, the list
    will be a len = 1 list.
    """

    return_tuple = []
    variable_names = ['id','variable','surface_type']
    len_list = 0
    cc= 0
    for list_or_str,var_name in zip([id,variable,surface_type],variable_names):
        if isinstance(list_or_str,str):
            return_tuple.append([list_or_str])
        elif isinstance(list_or_str,list):
            if var_name == 'id':
                to_append = []
                for item in list_or_str:
                    if isinstance(item,str) or isinstance(item,int):
                        to_append.append([str(item)])
                    elif isinstance(item,list):
                        inner_append = []
                        for inner_item in item:
                            inner_append.append(str(inner_item))
                        
                        to_append.append(inner_append)
                    else:
                        raise ValueError('ids may only be specified as integer or strings')
                
                return_tuple.append(to_append)

            else:
                for item in list_or_str:
                    if not isinstance(item,str):
                        raise ValueError('{} may only be specified as strings'.format(var_name))
                
                return_tuple.append(list_or_str)

        elif isinstance(list_or_str,int) and var_name == 'id':
            return_tuple.append([str(list_or_str)])
        else:
            raise ValueError('argument: {} must be a string or a list'.format(var_name))
        
        if cc == 0:
            len_list = len(return_tuple[0])
        
        if len(return_tuple[-1]) != len_list:
            raise ValueError('All input variables must be lists of the same length')

        cc+=1

    #getting some really weird bugs if the number of id's is not
    #greater than or equal to the previous number of listed id's
    #on multiple surface integral evaluations
    _return_tuple = []
    len_perm = sort_list_of_lists_by_list_len(return_tuple[0])
    for rt in return_tuple:
        _return_tuple.append(apply_permutation_to_list(rt,len_perm))
        
    return tuple(_return_tuple) 

    
    


