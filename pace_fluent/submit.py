#native imports
from abc import ABC
from typing import Iterable, Union
import os
from pandas.core.frame import DataFrame
import numpy as np
import shutil
from pathlib import Path, PosixPath,WindowsPath
from collections.abc import MutableMapping
import pandas as pd
import json

#package imports
from fluentpy.fluentio.disk import SerializableClass
from .pbs import FluentPBS
from fluentpy.tui.fluent import BatchCaseReader, FluentRun, LINE_BREAK
from .util import partition_boundary_table
from .filesystem import TableFileSystem

""" 
Author: Michael Lanahan
Date created: 08.04.2021
Last Edit: 09.09.2021

functions and classes for submitting fluent pace jobs to pace
"""

FLUENT_DATA_EXT = '.dat'
FLUENT_CASE_EXT = '.cas'
FLUENT_SOLUTION_EXT = '.trn'
FLUENT_REPORT_FILE_EXT = '.out'
WARNINGS = True

class Transfer(ABC):

    def __init__(self,src:Union[str,list], 
                      dst:Union[str,list]):

        self.__src = src
        self.__dst = dst

    @property
    def src(self):
        return self.__src
    
    @property
    def dst(self):
        return self.__dst
    
    @staticmethod
    def _single_source_call(src):
        if ':' in src:
            _src = src.split(':')[1]
        else:
            _src = src
        
        if os.path.isdir(_src):
            return '--recursive ' + src
        elif os.path.exists(_src):
            return src
        else:
            raise FileExistsError('source is not a file or a folder')
        
    def _batch_call(self):

        if len(self.dst) != len(self.src):
            raise ValueError('Cannot establish one-to-one relationship between source and destination with differing lengths')
        
        txt = ''
        for src,dst in zip(self.src,self.dst):
            _src = self._single_source_call(src)
            txt += _src + '  ' + dst + '\n'

        return txt

    def format_transfer_call(self):

        txt = 'globus transfer '
        if isinstance(self.src,str):
            if not isinstance(self.dst,str):
                raise ValueError('Cannot specify multiple outputs for destination if source is a string')

            src_call = self._single_source_call(self.src)
            return txt + src_call + '  ' + self.dst

        else:

            return txt + ' --batch ' + self._batch_call()
    
    def __call__(self):
        return self.format_transfer_call()
    
class FluentSubmission(SerializableClass):

    """
    Class for representing the unit "submission" of a fluent job to pace
    This MUST be instantiated with a FluentRun as a first argument and a 
    FluentPBS script as a second argument

    Methods 
    -----------
    - write
    - format_submit
    - submit

    """
    
    def __init__(self,fluent_run: FluentRun,
                      fluent_pbs: FluentPBS):
        
        self.__fluent_run = fluent_run
        self.__fluent_pbs = fluent_pbs
    
    @property
    def fluent_run(self):
        return self.__fluent_run
    
    @property
    def fluent_pbs(self):
        return self.__fluent_pbs
    
    def write(self,folder,
                   pace_pbs = 'fluent.pbs'):

        """ 
        writes the fluent pbs script & the fluent file to a folder 
        provided in the variable "folder". If the folder does not exist,
        the folder will be created
        """
        
        if not os.path.isdir(folder):
            os.mkdir(folder)
        
        self.fluent_run.write(os.path.join(folder,self.fluent_pbs.input_file))
        self.fluent_pbs.write(os.path.join(folder,pace_pbs))
    
    def format_submit(self,f,
                          pace_pbs = 'fluent.pbs'):
        """
        format the submission at the command line in pace
        """
        
        self.write(f)
        return ['qsub ',pace_pbs]
    
    def bash_submission(self,f,
                       pace_pbs = 'fluent.pbs'):

        """
        submit the job. Format the submission, i.e. write the file to the folder
        change to the directory, run the script, and then change back to the original directory
        return the output from the command line call
        """
        
        cmd = self.format_submit(f,pace_pbs = pace_pbs)
        return cmd
    
    def _from_file_parser(dmdict):
        
        fluentpbs = dmdict['fluent_pbs']['class']
        pbs = fluentpbs.from_dict(dmdict['fluent_pbs'])
        fluentrun = dmdict['fluent_run']['class']
        run = fluentrun.from_dict(dmdict['fluent_run'])

        return ([run,pbs],)
        
class FluentBatchSubmission:

    """
    an interface for the batched submission of fluent jobs 
    """

    _df = None

    def __init__(self,fluent_submission_list: list,
                      index = None,
                      prefix = '',
                      seperator = '-'):

        index,prefix,seperator = self._initializer_kwarg_parser(len(fluent_submission_list),
                                                                index,prefix,seperator)

        self.prefix = prefix
        self.seperator = seperator
        self.__submission_object = dict(zip([prefix + seperator + str(i) for i in index],
                                             fluent_submission_list))


    @staticmethod
    def _initializer_kwarg_parser(n: int,
                                  index: Iterable,
                                  prefix: str,
                                  seperator: str) -> tuple:

        if index is None:
            index = range(n)
        
        if prefix is None or prefix == '':
            try:
                if index.name is not None:
                    prefix = index.name
                else:
                    prefix = ''
                    seperator = ''
            except AttributeError:
                prefix = ''
            
        return index,prefix,seperator

    @property
    def submission_object(self):
        return self.__submission_object

    @submission_object.setter
    def submission_object(self,so):
        self.__submission_object = so
    
    def __add__(self,other) -> None:

        if not isinstance(other,FluentBatchSubmission):
            raise TypeError('can only add one fluent batch submission with another')

        def _add_df(df1,df2):

            return pd.concat([df1,df2],axis = 0)

        def _add_submission_object(so1,so2):
            
            so = dict.fromkeys(list(so1.keys()) + list(so2.keys()))

            for key in so2:
                if key in so1:
                    raise ValueError('ALl keys in submission object (controlled by the index) must be unique between the added batch submission')

            for key, values in so1.items():
                so[key] = values
            
            for key, values in so2.items():
                so[key] = values

            return so
        
        self.submission_object = _add_submission_object(self.submission_object,
                                                        other.submission_object)
        
        self._df = _add_df(self._df,other._df)
        
        return self

    def _populate_batch_information_folder(self,parent: str):
        """
        make the cache to access later
        """
        batch_cache = BatchCache(parent)
        batch_cache.cache_batch(self.submission_object,
                                self._df,
                                prefix = self.prefix,
                                seperator = self.seperator
                                )
    
    def format_submit(self,parent: str,
                          pace_pbs: str,
                          verbose = True,
                          purge = False):
        
        """
        final function here for formatting the submission
        makes appropriate directories if they do not exist
        and optionally purges data using a safety delete that does not allow
        recursion past a level of 2 on file folders, and will not delete .cas
        or .dat files
        """
        
        if not os.path.isdir(parent):
            os.mkdir(parent)
        else:
            if purge:
                _safety_delete(parent,max_depth= 2, keep_exts= ['.cas','.dat'])
            
            try:
                os.mkdir(parent)
            except FileExistsError:
                pass
        
        self._populate_batch_information_folder(parent)
        
        txt = ''
        for folder,submission in self.submission_object.items():
            if verbose:
                txt += 'echo "executing job located in folder: {}"'.format(folder) + LINE_BREAK
            
            command,file = submission.format_submit(os.path.join(parent,str(folder)))
            txt += 'cd '  + str(folder) + LINE_BREAK 
            txt += ''.join([command,file,LINE_BREAK])
            txt += 'cd ..' + LINE_BREAK

        return txt
    
    def bash_submit(self,parent: str,
                         batch_file = 'batch.sh',
                         pace_pbs = 'fluent.pbs',
                         purge = False,
                         verbose = True):

        _bf = os.path.join(parent,batch_file)
        txt = self.format_submit(parent,pace_pbs,
                                 verbose = verbose, 
                                 purge = purge)

        with open(_bf,'w',newline = LINE_BREAK) as file:
            file.write(txt)
    
    @classmethod
    def from_batch_cache(cls,
                         batch_folder: str): 
        """
        read in the batch_cache
        """
        batch_cache = BatchCache(batch_folder)
        so = batch_cache.read_submission_object_cache()
        fmt_kwargs = batch_cache.read_formatting_cache()
        df = batch_cache.read_df_cache()

        _cls = cls(list(so.values()),index = list(so.keys()),
                    **fmt_kwargs)

        #there are some troublesome classes that I have to hack to make work here
        for pwd,fluent_submission in _cls.submission_object.items():
            fluent_submission.fluent_run.reader.pwd = pwd 
                
        _cls._df = df
        return _cls

    @classmethod
    def from_frame(cls,
                   case_name: str,
                   turbulence_model:str,
                   df:DataFrame,
                   pbs:FluentPBS,
                   prefix = '',
                   seperator = '-',
                    *frargs,
                    **frkwargs):

        """
        Class method for creating a fluent batch submission using a data frame
        refer to the util.py document for details on the specification of the input dataframe

        the SAME pbs script is used for each of the runs - if there needs to be a different pbs script for 
        each run you should consider manually building the list

        frargs and frkwargs are passed to the fluent run objects for each of the FluentSubmissions - so they will
        be the same for each run - again if you need these to be different you should consider building the
        batch submission differently or submitting seperate jobs
        """

        index,_,seperator = cls._initializer_kwarg_parser(df.shape[0],
                                                          df.index,prefix,seperator)

        boundary_df = partition_boundary_table(df,turbulence_model)
        sl,index = cls.submission_list_from_boundary_df(case_name, boundary_df, pbs,
                                                         seperator,*frargs,**frkwargs)
        cls._df = df 
        return cls(sl,index = index,prefix = index.name)

    @staticmethod
    def submission_list_from_boundary_df(case_name: str,
                                         bdf:DataFrame,
                                         pbs:FluentPBS,
                                         seperator: str,
                                         *frargs,
                                         **frkwargs) -> list:

        """
        make a submission object from a boundary DataFrame i.e. a DataFrame of
        FluentBoundaryConditions.
        """
        
        submit_list = []
        name = '' if bdf.index.name is None else bdf.index.name
        for index in bdf.index:
            fluent_run = FluentRun(case_name,
                                   reader = BatchCaseReader,
                                   *frargs,**frkwargs)
            
            fluent_run.reader.pwd = name + seperator + str(index)
            fluent_run.boundary_conditions = list(bdf.loc[index])
            _pbs = pbs.copy()
            _pbs.pbs.name = name + seperator + str(index)

            submit_list.append(
                               FluentSubmission(fluent_run,
                                                _pbs)
                              )
        
        return submit_list,bdf.index

class BatchCache:
    
    _BATCH_CACHE_FOLDER = '_batchcache_'
    _SUBMISSION_CACHE_FILE = '.submission'
    _FRAME_CACHE_FILE = '.frame'
    _FORMATTING_CACHE_FILE = '.fmt'
    
    def __init__(self,batch_folder: str):

        self.batch_folder = batch_folder

        #check to see if the cache folder exists
        if not os.path.isdir(self.batch_cache):
            os.mkdir(self.batch_cache)

    @property
    def batch_cache(self):
        return os.path.join(self.batch_folder,self._BATCH_CACHE_FOLDER)

    def cache_submission_object(self,submission_object:dict) -> None:
        """
        cache the submission object for later retrieval
        """

        sf = os.path.join(self.batch_cache,self._SUBMISSION_CACHE_FILE)
        with open(sf,'w') as sf_file:
            for folder,submission in submission_object.items():
                submission_cache_file = os.path.join(self.batch_cache,folder)
                submission.serialize(submission_cache_file)
                sf_file.write(folder + LINE_BREAK)
    
    def cache_df(self, df: pd.DataFrame) -> None:
        """
        cache the data frame if there is one associated with the batch
        """

        frame_file = os.path.join(self.batch_cache,self._FRAME_CACHE_FILE)
        df.to_csv(frame_file)
    
    def cache_batch_formatting(self,kwargs) -> None:
        """
        cache the formatting arguments
        """

        with open(os.path.join(self.batch_cache,self._FORMATTING_CACHE_FILE),'w') as file:
            json.dump(kwargs,file)
    
    def cache_batch(self,submission_object: dict,
                         df = None,
                         **fmtkwargs) -> None:
        """
        cache everything associated with the batch for rebuild
        """
        self.cache_submission_object(submission_object)
        if df is not None:
            self.cache_df(df)
        
        self.cache_batch_formatting(fmtkwargs)

    def read_submission_object_cache(self) -> dict:
        """
        read in the submission object cache
        """
        sf = os.path.join(self.batch_cache,self._SUBMISSION_CACHE_FILE)
        with open(sf,'r') as file:
            folders = []
            for line in file.readlines():
                folders.append(line.strip())
        
        submission_object = dict.fromkeys(folders)
        for folder in folders:
            submission_file_name = os.path.join(self.batch_cache,folder)
            submission_object[folder] = FluentSubmission.from_file(submission_file_name)
        
        return submission_object
    
    def read_df_cache(self) -> pd.DataFrame:
        """
        read in the cached data frame
        """
        try:
            df_file = os.path.join(self.batch_cache,self._FRAME_CACHE_FILE)
            return pd.read_csv(df_file,header = 0,index_col= 0)
        except FileNotFoundError:
            return None
        
    def read_formatting_cache(self) -> dict:
        """
        read in the cached formatting
        """
        with open(os.path.join(self.batch_cache,self._FORMATTING_CACHE_FILE),'r') as file:
            fmt_data = json.load(file)
        
        return fmt_data

class BatchSubmissionSummary:

    COLUMNS = ['data','case','solution','report','completed']

    def __init__(self, folder: str):
        
        self._folder= folder
        self.filesys = TableFileSystem(folder)
    
    def get_folders_with_data(self):

        return list(self.filesys._find_ext_in_submission_folders(FLUENT_DATA_EXT).keys())
    
    def get_folders_with_case(self):
        return list(self.filesys._find_ext_in_submission_folders(FLUENT_CASE_EXT).keys())

    def get_folders_with_solution(self):
        self.filesys.map_solution_files()
        return list(self.filesys.solution_file_dict.keys())
    
    def get_folders_with_report_file(self):
        self.filesys.map_report_files()
        return list(self.filesys.report_file_dict.keys())
    
    def get_folders_with_completed_solution(self):
        
        folders = []
        self.filesys.solution_files
        print('here')
        for solution_file in self.filesys.solution_files.keys:
            with self.filesys.solution_files[solution_file] as sf:
                if sf.STATUS:
                    folder = self.filesys.Path(os.path.join(self._folder,sf.fluent_folder))
                    folders.append(folder)
        
        return folders
    
    def make_summary(self):

        dfolders = self.get_folders_with_data()
        cfolders = self.get_folders_with_case()
        sfolders = self.get_folders_with_solution()
        rfolders = self.get_folders_with_report_file()
        csfolders = self.get_folders_with_completed_solution()

        index = np.array(list(set(dfolders + cfolders + sfolders + rfolders + csfolders)))
        data = np.zeros(shape = [index.shape[0],len(self.COLUMNS)],dtype = bool)
        for i,_list in enumerate([dfolders,cfolders,sfolders,rfolders,csfolders]):
            for j in range(len(index)):
                if index[j] in _list:
                    data[j,i] = True
        
        return pd.DataFrame(data,index = index,columns = self.COLUMNS)
    
    def remake_batch(self,new_batch_folder: str,
                          missing_criterion: list,
                          **batch_kwargs) -> None:

        """ 
        allows the user to remake a batch based on some missing criterion specificed 
        by COLUMNS
        """
        
        for mc in missing_criterion:
            if mc not in self.COLUMNS:
                raise ValueError('can only remake batch based on missing criterion in: {}'.format(self.COLUMNS))
        
        summary = self.make_summary()
        
        try:
            _summary = summary[missing_criterion]
            exclude_series = _summary.all(axis = 1)
            cached_batch = FluentBatchSubmission.from_batch_cache(self._folder)
            for index in exclude_series.index:
                if exclude_series.loc[index]:
                    cached_batch.submission_object.pop(index.name)
            
            for folder,submission in cached_batch.submission_object.items():
                submission.fluent_run.reader.pwd = folder
            
            if cached_batch._df is not None:
                new_index = []
                dtype = cached_batch._df.index.dtype
                try:
                    name_length = len(cached_batch._df.index.name)
                except TypeError:
                    name_length = 0
                
                for index in exclude_series.index:
                    new_index.append((index.name[name_length:]))
                
                new_index = np.array(new_index,dtype = dtype)

                _exclude_series = pd.Series(exclude_series.to_numpy(),index = new_index)
                cached_batch._df = cached_batch._df[~_exclude_series]
            
            cached_batch.bash_submit(new_batch_folder,**batch_kwargs)
        except FileNotFoundError:
                raise FileNotFoundError('cannot find batch cache in batch folder')


class FileSystemTree:

    def __init__(self,root: str,
                      os_system = 'windows'):

        self.os_system = os_system.lower()
        self.__path = self._get_path(root)
        if self.os_system == 'windows':
            self.delim = '\\'
        else:
            self.delim = '/'
        
        self.__tree = None
        self.__depth = None

    @property
    def path(self):
        return self.__path

    @property
    def tree(self):
        if self.__tree is None:
            self.tree_contents()
        return self.__tree
    
    @tree.setter
    def tree(self,t):
        self.__tree = t

    @property
    def depth(self):
        return self.__depth
    
    @depth.setter
    def depth(self,d):
        self.__depth = d

    def iterfile(self):
        flattend = _flatten_dict(self.tree)
        for key in flattend:
            yield key

    def tree_contents(self):
        self.depth = 0
        self.tree = self._make_tree(self.path)
        return self.tree
    
    def _make_tree(self,root:str):
        
        if not issubclass(type(root),Path): 
            _root = self._get_path(root)
        else:
            _root = root
        
        contents = dict.fromkeys(_root.iterdir())
        for f in contents:
            if f.is_dir():
                contents[f] = self._make_tree(f)
            else:
                self.depth = max(self.depth,len(str(f.parent).split(self.delim)))
                contents[f] = None
            
        return contents
    
    def _get_path(self,path: str):
        
        if self.os_system == 'windows':
            path = WindowsPath(path)
        elif self.os_system == 'linux' or self.os_system == 'posix':
            path = PosixPath(path)
        else:
            raise ValueError("os_system must be specified by strings: (1) windows (2) linux or (3) posix")

        return path

def _flatten_dict(d, 
                  parent_key='', 
                  sep='\\') -> dict:
    
    """
    flatten a dictionary, concatenating the keys as string representations
    using sep as the seperator. 

    meant to flatten dictionaries of path from the pathlib library so will error out 
    if you pass something else
    """ 

    items = []
    for k, v in d.items():
        new_key = str(parent_key.parent) + sep + str(parent_key.name) + sep + str(k.name) if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)

def _safety_delete(root: str,
                   max_depth = 3,
                   keep_files = [],
                   keep_exts = []) -> None:
    """
    removes a folder in its entirety only if the folder specified by 
    root contains a tree structure with a maximum number of levels specified by 
    max_depth below it. 

    Will not delete the whole root if keep_files or keep_exts is not empty,
    will delete every other file that is not
    not contained in keep_files and does not have an extension in keep_exts

    makes sure you don't accidently do something stupid like delete your whole system
    or rather makes it really hard to do this
    """

    tree = FileSystemTree(root)
    tree.tree

    if tree.depth > max_depth:
        pass
    else:
        if not keep_files and not keep_exts:
            shutil.rmtree(root)
        else:
            for file in tree.iterfile():
                ext = os.path.splitext(file)[1]
                if file not in keep_files and ext not in keep_exts:
                    os.remove(file)
        
def test_transfer():
    end = '6df312ab-ad7c-4bbc-9369-450c82f0cb92:'
    start = '00f6bef2-f38f-11eb-832a-f56dd2959cb8:'
    src = start + 'test'
    print(src)
    endpoint = end + 'scratch/test-globus'

    transfer = Transfer(src,endpoint)
    call = transfer()
    print(call)

def test_submission():

    pbs = FluentPBS('fluent_input',
                   'p-my14-0/michael/test-file-exchange/test-case',
                    WALLTIME = 1000,
                    MEMORY = 8,
                    N_NODES = 1,
                    N_PROCESSORS = 12,
                    email = 'mlanahan3@gatech.edu')

    run = FluentRun('test.cas')
    submission = FluentSubmission(run,pbs)
    submission.write('job_submission')
    submit_Txt = submission.submit('job_test')
    print(submit_Txt)

def test_batch_submission():

    pbs = FluentPBS('fluent_input',
                   'p-my14-0/michael/test-file-exchange/test-case',
                    WALLTIME = 1000,
                    MEMORY = 8,
                    N_NODES = 1,
                    N_PROCESSORS = 12,
                    email = 'mlanahan3@gatech.edu')

    data = np.array([[4e6,300,3.8,0.003,0.04,300,4e6,3.8,0.013],
                     [4.1e6,300,3.8,0.003,0.05,300,4e6,3.8,0.013]])

    table = DataFrame(data,
                     columns = ['outer_tube_outlet:pressure:pressure-outlet',
                                 'outer_tube_outlet:temperature:pressure-outlet',
                                 'outer_tube_outlet:turbulent intensity:pressure-outlet',
                                 'outer_tube_outlet:hydraulic diameter:pressure-outlet',
                                 'inner_tube_inlet:mass flow rate:mass-flow-inlet',
                                 'inner_tube_inlet:temperature:mass-flow-inlet',
                                 'inner_tube_inlet:initial pressure:mass-flow-inlet',
                                 'inner_tube_inlet:turbulent intensity:mass-flow-inlet',
                                 'inner_tube_inlet:hydraulic diameter:mass-flow-inlet'])
    
    batch_submit = FluentBatchSubmission.from_frame('ICM-11.cas.gz',table,pbs)
    
    submit_msg = batch_submit.format_submit('test-batch')
    #print(submit_msg)

def test_file_tree():

    
    fst = FileSystemTree('test-batch')
    fst.tree
    for file in fst.iterfile():
        print(file)
    
    #_safety_delete('test-case',max_depth= 3)

def main():

    #test_transfer()
    #test_submission()
    #test_batch_submission()
    test_file_tree()



if __name__ == '__main__':
    main()
