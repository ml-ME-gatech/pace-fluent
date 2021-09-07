from abc import ABC
from datetime import timedelta
from copy import deepcopy

"""
Author: Michael Lanahan
Date Created: 08.01.2021
Last Edit: 08.05.2021

scripts for working with fluent using the PACE computational cluster at GT

https://docs.pace.gatech.edu/software/PBS_script_guide/

#Example PBS script

#PBS -N <job-name>                  -> name that shows up on the queue
#PBS -A [Account]                   -> Account - the account that is required to run the jobs - who to charge
#PBS -l nodes=1:ppn=8: cores24      -> resource-list; number of nodes and processers per node, specify number of cores
#PBS -l pmem=8gb                    -> memory allocated PER CORE
#PBS -l mem = 2gb                   -> total memory allocated over all nodes
#PBS -l walltime=2:00:00            -> projected walltime - hard stop here
#PBS -q inferno                     -> which que to submit too
#PBS -j oe                          -> combines output and error into one file
#PBS -o fluent.out                  -> names output files
#PBS -m <a,b,e>                     -> will send job satus
                                    -> a - if job aborted; b- when job begins; e - when job ends

cd $PBS_O_WORKDIR                               -> change to working directroy - where script is submited from
module load ansys/<version.number>              -> load ansys, with version number in <int>.<int> i.e. 19.12
fluent -t8 -g <inputfile> outputfile            -> run fluent command with input file and output file
"""

LINE_BREAK = '\n'
class PBS(ABC):

    """
    Abstract base class for the pbs header. Provides a base initialization method that formats the various
    initializer arguments into the required form for the pbs script. The formatting is done 
    in the property methods and is formatted into a script by the method "format_pbs_header()"
    this formatted text is then called like-so: 

    pbs = PBS(*args,**kwargs)
    text = pbs()

    where the variable "text" now contains the text in a pbs script

    note that the key word argument "memory_request" has options 'p' and 't'
    where 'p' is per core and 't' is total memory
    """
    
    line_leader = '#PBS '
    def __init__(self, name: str,
                       account: str,
                       queue: str,
                       output_file: str,
                       walltime: float,
                       memory:float,
                       nodes: int,
                       processors: int,
                       email = None,
                       email_permissions = None,
                       memory_request = 'p',
                       *args,
                       **kwargs):

        self.__name = name
        self.__account = account
        self.__queue = queue
        self.__output_file = output_file
        self.__walltime = walltime
        self.__memory = memory
        self.__nodes = nodes
        self.__processors = processors
        self.__email = email
        self.__email_permissions = email_permissions
        self.memory_request = memory_request.lower()
        if self.memory_request != 'p' and self.memory_request != 't':
            raise ValueError('memory must be requested on a per node "p" or total "t" basis')


    
    @property
    def name(self):
        return '-N ' + self.__name 
    
    @property
    def account(self):
        return '-A ' + self.__account
    
    @property
    def queue(self):
        return '-q ' + self.__queue
    
    @property
    def output_file(self):
        return '-o ' + self.__output_file
    
    @property
    def walltime(self):
        return '-l walltime=' + str(timedelta(seconds = self.__walltime))
    
    @property
    def memory(self):
        msg = '-l {}mem=' + str(self.__memory) + 'gb'
        if self.memory_request == 'p':
            return msg.format('p')
        else:
            return msg.format('')
    
    
    @property
    def processesors_nodes(self):
        return '-l nodes=' + str(self.__nodes) + ':ppn=' + str(self.__processors)
    
    @property
    def email(self):
        if self.__email is None:
            return self.__email
        else:
            return '-M '+  self.__email
    
    @property
    def email_permissions(self):
        if self.__email_permissions is None:
            return self.__email_permissions
        else:
            return '-m ' + self.__email_permissions

    @name.setter
    def name(self,n):
        self.__name = n
    
    def format_pbs_header(self):

        txt = ''
        for item in [self.name,self.account,self.queue,self.output_file,self.walltime,
                     self.memory,self.processesors_nodes,self.email,self.email_permissions]:
        
            if item is not None:
                txt += self.line_leader + ' ' + item + LINE_BREAK
        
        return txt
    
    def copy(self):
        return deepcopy(self)

    def __call__(self):
        return self.format_pbs_header()



class DefaultPBS(PBS):

    """
    a default pbs script. The account and queue are now hardcoded into the initializer
    while variables such as walltime,memory, nodes and processors are still required
    """
    
    def __init__(self,name: str,
                       walltime: float,
                       memory:float,
                       nodes: int,
                       processors: int,
                       output_file = 'fluent.out',
                       email = None,
                       email_permissions = 'abe',
                       memory_request = 'p'):


        super().__init__(name,'GT-my14','inferno',output_file,walltime,
                        memory,nodes,processors,email = email,
                        email_permissions= email_permissions,memory_request = memory_request)


class FluentPBS:

    """
    A default PBS scrit for a fluent function call. Provides additional formatting 
    for calling ANSYS fluent after the pbs script. Note that the type of pbs 
    can be changed by supplying the keyword argument
    
    pbs = MyPBSClass
    
    in the initialization of the FluentPBS class

    The initial header from the pbs class, along with the additional text required for the fluent
    call are formatted in the function "format_call" and the text is returned by the following syntax

    fluentpbs = FluentPBS(*args,**kwargs)
    txt = fluentpbs()

    where the variable "txt" contains all of the information required for a pbs script with fluent
    """
    
    PBS_PDIR = '$PBS_O_WORKDIR'
    PBS_NODE_FILE = '$PBS_NODEFILE'

    def __init__(self,
                 input_file: str,
                 name = None,
                 WALLTIME = None,
                 MEMORY = None,
                 N_NODES = 1,
                 N_PROCESSORS = 1,
                 output_file = 'pace.out',
                 version = '2019R3',
                 email = None,
                 email_permissions = 'abe',
                 mpi_option = 'intel',
                 pbs = DefaultPBS,
                 memory_request = 'p'):
        
        if WALLTIME is None:
            raise ValueError('wall time must be specified in order to run script')
            
        if MEMORY is None:
            raise ValueError('The amount of memory must be specified')
        
        if name is None:
            name = input_file
        
        self.pbs = pbs(name,WALLTIME,MEMORY,N_NODES,N_PROCESSORS,
                              output_file,email = email,email_permissions = email_permissions,
                              memory_request = memory_request)
        
        self.mpi_opt = mpi_option
        self.version = version
        self.N_PROCESSORS = N_PROCESSORS
        self.N_NODES = N_NODES
        self.input_file = input_file

    def format_machine_file(self):
        """
        sets the MPI option correctly for pace
        """
        if self.mpi_opt == 'pcmpi' or self.mpi_opt == 'intel' or self.mpi_opt == 'ibmmpi':
            return ' -mpi=' + self.mpi_opt
        elif self.mpi_opt == 'pib':
            return ' -' + self.mpi_opt
        elif self.mpi_opt is None:
            return ''
        
        else:
            raise ValueError('{} is not a valid mpi option'.format(self.mpi_opt))
    
    def format_cnf(self):
        """
        this is required for the process affinity to work on pace
        """
        return ' -cnf=' + self.PBS_NODE_FILE + ' '

    def format_call(self):
        """
        format the whole script here
        """
        txt = self.pbs() + LINE_BREAK
        txt += self.format_change_dir(self.PBS_PDIR) +LINE_BREAK
        txt += self.format_load_ansys(self.version) +LINE_BREAK
        mpi = self.format_machine_file()
        cnf = self.format_cnf()
        txt += self.format_fluent_footer(self.N_PROCESSORS,self.N_NODES,self.input_file,mpi,cnf) + LINE_BREAK

        return  txt
    
    @staticmethod
    def format_change_dir(chdir):
        """
        this is important to change to the PBS dir which is an environment variable 
        native to the bash shell on PACE
        """
        return 'cd ' + chdir
    
    @staticmethod
    def format_load_ansys(version):

        """
        format the command to load the version of ansys
        """
        return 'module load ansys/' + version
    
    @staticmethod
    def format_fluent_footer(processors:str,
                             nodes: str,
                             input_file:str,
                             mpi: str,
                             cnf: str) -> str:
        """
        format the fluent call in the pbs script
        """
        
        return 'fluent 3ddp -t' + str(int(processors*nodes)) + mpi + cnf + ' -g < ' + input_file + ' > outputfile'
    
    def __call__(self):
        """
        callable interface here
        """
        return self.format_call()
    
    def copy(self):
        """
        useful for making a bunch of copies in a batch script
        """
        return deepcopy(self)
    
    def write(self,f):
        """
        write the script to a file or IOStream (whatever its called in python)
        """
        try:
            with open(f,'w',newline = LINE_BREAK) as file:
                file.write(self.format_call())

        except TypeError:
            f.write(self.format_call())


        
    
def main():
    
    pbs = FluentPBS('fluent_input',
                    'p-my14-0/michael/test-file-exchange/test-case',
                    WALLTIME = 1000,
                    MEMORY = 8,
                    N_NODES = 1,
                    N_PROCESSORS = 12,
                    email = 'mlanahan3@gatech.edu')

    print(pbs())
    with open('fluent_input.pbs','w',newline = LINE_BREAK) as file:
        pbs.write(file)

if __name__ == '__main__':
    main()





