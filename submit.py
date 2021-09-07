from abc import ABC
from typing import Collection, Union
import os
from pandas.core.frame import DataFrame
from pbs import FluentPBS
from fluent import BatchCaseReader, FluentRun, LINE_BREAK
from util import partition_boundary_table
import numpy as np
import shutil
from pathlib import Path, PosixPath,WindowsPath
from collections.abc import MutableMapping
""" 
Author: Michael Lanahan
Date created: 08.04.2021
Last Edit: 08.10.2021

functions and classes for submitting fluent pace jobs to pace
"""

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
    
class FluentSubmission:

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
        
class FluentBatchSubmission:

    """
    an interface for the batched submission of fluent jobs 
    """
    
    def __init__(self,fluent_submission_list: list,
                      index = None,
                      prefix = ''):

        if index is None:
            index = range(len(fluent_submission_list))
        
        if prefix is None:
            prefix = ''
        
        self.prefix = prefix
        self.__submission_object = dict(zip([prefix +'-'+ str(i) for i in index],
                                             fluent_submission_list))

    @property
    def submission_object(self):
        return self.__submission_object
    
    def format_submit(self,parent: str,
                          pace_pbs: str,
                          verbose = True,
                          purge = False):
        
        
        if not os.path.isdir(parent):
            os.mkdir(parent)
        else:
            if purge:
                _safety_delete(parent,max_depth= 2, keep_exts= ['.cas'])
            
            try:
                os.mkdir(parent)
            except FileExistsError:
                pass
        
        
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

        with open(batch_file,'w',newline = LINE_BREAK) as file:
            file.write(self.format_submit(parent,pace_pbs,verbose = verbose, 
                                          purge = purge))
    
    @classmethod
    def from_frame(cls,
                   case_name: str,
                   turbulence_model:str,
                   df:DataFrame,
                   pbs:FluentPBS,
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
        
        boundary_df = partition_boundary_table(df,turbulence_model)
        sl,index = cls.submission_list_from_boundary_df(case_name, boundary_df, pbs,*frargs,**frkwargs)
        return cls(sl,index = index,prefix = index.name)

    @staticmethod
    def submission_list_from_boundary_df(case_name: str,
                                         bdf:DataFrame,
                                         pbs:FluentPBS,
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
                                   name = name + '-' + str(index),
                                   reader = BatchCaseReader,
                                   *frargs,**frkwargs)
            fluent_run.reader.pwd = name + '-' + str(index)
            fluent_run.boundary_conditions = list(bdf.loc[index])
            _pbs = pbs.copy()
            _pbs.pbs.name = name + '-' + str(index)
            submit_list.append(
                               FluentSubmission(fluent_run,
                                                _pbs)
                              )
        
        return submit_list,bdf.index

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
                   max_depth = 2,
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
