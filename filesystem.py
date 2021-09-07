from abc import ABC,abstractmethod
from pathlib import WindowsPath,PosixPath
import sys
from wbfluentpy.io.classes import ReportFilesOut,SolutionFiles

"""
Author: Michael Lanahan
Date Created: 08.11.2021
Last Edit: 08.11.2021

The purpose of this folder is to provide classes and functions for working with
any batch file system structures created on PACE using fluent
"""

SOLUTION_EXT = '.trn'
REPORT_EXT = '.out'
REPORT_FILE_TAG = 'report-file'

class BatchFileSystem(ABC):

    def __init__(self, root: str,
                       safe = True,
                       *args,**kwargs):

        #check the file system to determine how to behave here - could be 
        #deployed on PACE so have to account for linux/posix platforms 
        #pathlib should make this easier
        os_name = sys.platform
        if os_name == 'win32' or os_name == 'win64':
            self.__Path = WindowsPath
        elif os_name == 'linux' or os_name == 'posix':
            self.__Path = PosixPath
        else:
            if not safe:
                raise OSError('Cannot determine platform of system and cannot gaurentee expected performance. Overide by passing the safe flag as false')
    
        self.__root = self.Path(root)
        self.__submission_folder_list = []
        self.__report_file_dict = {}
        self.__solution_file_dict = {}
        self.__report_files = None
        self.__solution_files = None

    @property
    def root(self):
        return self.__root
    
    @property
    def Path(self):
        return self.__Path

    @property
    def submission_folder_list(self):
        if not self.__submission_folder_list:
            self.map_submit_folders()
        return self.__submission_folder_list
    
    @property
    def report_file_dict(self):
        return self.__report_file_dict
    
    @property
    def solution_file_dict(self):        
        return self.__solution_file_dict
    
    @property
    def solution_files(self):
        if self.__solution_files is None:
            self._make_solution_files()
        
        return self.__solution_files
    
    @property
    def report_files(self):
        if self.__report_files is None:
            self._make_report_files()
        
        return self.__report_files

    @submission_folder_list.setter
    def submission_folder_list(self,sfl):
        self.__submission_folder_list = sfl

    @solution_file_dict.setter
    def solution_file_dict(self,sfd):
       self.__solution_file_dict = sfd
    
    @report_file_dict.setter
    def report_file_dict(self,rfd):
        self.__report_file_dict = rfd

    @solution_files.setter
    def solution_files(self,sf):
        self.__solution_files = sf

    @report_files.setter
    def report_files(self,rf):
        self.__report_files = rf 
    
    def map_solution_files(self):
        self.solution_file_dict = self._find_ext_in_submission_folders(SOLUTION_EXT)
    
    def map_report_files(self):
        self.report_file_dict = self._find_ext_in_submission_folders(REPORT_EXT,tag = REPORT_FILE_TAG)

    def _make_report_files(self):
        """
        make the report file class
        """
        if self.report_file_dict is None:
            self.map_report_files()

        arg_dict = {str(f.resolve()):str(f.parent.name) for files in 
                    self.report_file_dict.values() for f in files}
        
        self.report_files = ReportFilesOut(list(arg_dict.keys()),
                                          folder_names = list(arg_dict.values()))
        
        self.report_files.load()

    def _make_solution_files(self):

        """
        make the solution file class
        """
        if self.solution_file_dict is None:
            self.map_solution_files()
        
        arg_dict = {str(f.resolve()):str(f.parent.name) for files in 
                    self.solution_file_dict.values() for f in files}


        self.solution_files = SolutionFiles(list(arg_dict.keys()),
                                           folder_names = list(arg_dict.values()))
        
        self.solution_files.load()
    
    def _find_ext_in_submission_folders(self,ext,
                                             tag = '') -> dict:
        """
        go through the submission folders and find all such folders that contain a file
        with the extension "ext". Return a dictionary of found files with the keys of the dictionary
        the submission folder list and the values the files that are found. 

        Additional keyword argument "tag" allows for clarification with a nuemonic if there are multiple
        files with the same extension
        """
        _d = {}
        for folder in self.submission_folder_list:
            _found_files = []
            for f in folder.iterdir():
                if f.suffix == ext and tag in str(f):
                    _found_files.append(f)
            
            if _found_files:
                _d[folder] = _found_files
        
        return _d
    
    @abstractmethod
    def map_submit_folders(self):
        pass
    
class TableFileSystem(BatchFileSystem):

    def __init__(self,root: str,
                      safe = True):

        super().__init__(root,safe = safe)
    
    @property
    def case_file(self):
        return self.__case_file
    
    @case_file.setter
    def case_file(self,cf):
        self.__case_file = cf
    
    def map_submit_folders(self,
                           check_contents = []):

        """
        maps the submission folders, keeps a list of them and 
        provides the oppurtunity to check if files with the suffix provided by the list
        of check_contents exist in the folder. If they do not, these directories will be
        excluded from the submission_folder_list
        """
        sl = []
        for f in self.root.iterdir():
            if f.is_dir():
                flag = True
                if check_contents:
                    exts_in_dir = [_f.suffix for _f in f.iterdir()]
                    for c in check_contents:
                        if c not in exts_in_dir:
                            flag = False
                            break
                if flag: 
                    sl.append(f)


        self.submission_folder_list = sl
        

        



