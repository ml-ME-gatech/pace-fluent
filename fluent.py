from abc import ABC,abstractmethod,abstractstaticmethod
from functools import wraps
import os
from typing import OrderedDict
from numpy.core.einsumfunc import einsum

"""
Author: Michael Lanahan
Date Created: 08.05.2021
Last Edit: 08.16.2021

The purpose of this file is provide python class interfaces to TUI commands in 
the command line or batch mode fluent software. The hope is that this will make 
the batch submission of jobs more intelligble & easier to create for use on the 
PACE computing cluster @ GT

"""

WARNINGS = True
LINE_BREAK = '\n'
EXIT_CHAR = 'q' + LINE_BREAK
FLUENT_INIT_STATEMENT = 'fluent 3ddp -t{} < {} > {}'

ALLOWABLE_SOLVERS = ['pressure-based']
ALLOWABLE_VISCOUS_MODELS = ['ke-standard','kw-standard']
ALLOWABLE_MODELS = ['energy'] 
ALLOWABLE_MODELS += ['viscous/' + vm for vm in ALLOWABLE_VISCOUS_MODELS]
ALLOWABLE_BOUNDARY_TYPES = ['mass-flow-inlet','pressure-outlet','wall']

ALLOWABLE_DISCRITIZATION_SCHEMES = ['denisty','epsilon','k','mom','pressure','temperature']
ALLOWABLE_DISCRITIZATION_ORDERS = {'First Order Upwind':0,
                                   'Second Order Upwind':1,
                                   'Power Law': 2,
                                   'Central Differencing':3,
                                   'QUICK':4,
                                   'Standard':10,
                                   'Linear':11,
                                   'Second Order':12,
                                   'Body Force Weighted':13,
                                   'PRESTO!': 14,
                                   'Continuity Based':15,}

ALLOWABLE_VARIABLE_RELAXATION = ['body-force','epsilon','temperature','density', 'k', 'turb-viscosity']
ALLOWABLE_RELAXATION = {'courant number':200,
                        'momentum':0.75,
                        'pressure':0.75}

class Initializer:

    """
    representation of the initializer object in Fluent
    """
    
    _prefix = 'solve/initialize/'
    ALLOWABLE_INITIALIZER_TYPES = ['hyb-initialization','initialize-flow']
    def __init__(self,init_type = 'hyb-initialization'):

        if init_type not in self.ALLOWABLE_INITIALIZER_TYPES:
            raise ValueError('initializer must be one of: {}'.format(self.ALLOWABLE_INITIALIZER_TYPES))
        
        self.__init_type = init_type

    @property
    def init_type(self):
        return self.__init_type
    
    def __str__(self):
        return self._prefix + self.init_type

class Relaxation(ABC):

    def __init__(self,variable: str,
                      value: float,
                      *args,**kwargs) -> None:
        
        if isinstance(variable,str) and (isinstance(value,int) or isinstance(value,float)):
            self._check_allowable_variables(variable,value)
            variable = [variable]
            value = [value]

        elif isinstance(variable,list) and isinstance(value,list):
            if len(variable) != len(value):
                raise AttributeError('lists of variables and values must be the same length')
            else:
                for var,val in zip(variable,value):
                    self._check_allowable_variables(var,val)

        else:
            raise ValueError('cannot make relaxation from variable: {} and value: {}'.format(variable,value))
                
        self.__var_value_dict = dict(zip(variable,value))

    @abstractstaticmethod
    def _check_allowable_variables(variable: str,
                                    value: float):
        pass

    @classmethod
    def from_dict(cls,
                  input_dict: dict):

        vars = []
        vals = []
        for var,val in input_dict.items():
            vars.append(var)
            vals.append(val)
        
        return cls(vars,vals)

    @property
    def var_value_dict(self):
        return self.__var_value_dict
    
    @var_value_dict.setter
    def var_value_dict(self,vvd):
        self.__var_value_dict = vvd

    def format_relaxation(self):
        txt = self._prefix + LINE_BREAK
        for variable,value in self.var_value_dict.items():
            txt += str(variable) + LINE_BREAK
            txt += str(value) + LINE_BREAK
        
        txt += EXIT_CHAR + EXIT_CHAR + EXIT_CHAR
        return txt
    
    def __str__(self):
        return self.format_relaxation()

class ScalarRelaxation(Relaxation):

    _prefix = 'solve/set/under-relaxation'

    def __init__(self,variable: str,
                      value: float):

        super().__init__(variable,value)

    @staticmethod
    def _check_allowable_variables(variable: str,
                                    value: float) -> None:
                    
        if variable in ALLOWABLE_VARIABLE_RELAXATION:        
            if value > 1 or value < 0:
                raise ValueError('Value must be between 0 and 1, not {}'.format(value))
        
        else:
            raise ValueError('{} not an allowable variable for relaxation'.format(variable))

    
class EquationRelaxation(Relaxation):

    """
    class for allowing the adjustment of relaxation factors
    to aid in the convergence of difficult solutions
    This treats explicitly the equation relaxations so 
    momentum
    pressure
    and allows adjustment of the Courant number
    """
    _prefix = 'solve/set/p-v-controls'

    def __init__(self,variable: str,
                      value: float):

        super().__init__(variable,value)

        #unlike the other relaxation, the prompt requires all three
        #relaxation inputs regardless - we will provide the defaults if none
        #are provided
        for key,value in ALLOWABLE_RELAXATION.items():
            if key not in self.var_value_dict:
                self.var_value_dict[key] = value

        #this needs to be in the correct order
        self.var_value_dict = OrderedDict((k,self.var_value_dict[k]) for 
                                            k in ALLOWABLE_RELAXATION.keys())
        
    @staticmethod
    def _check_allowable_variables(variable: str,
                                   value: float) -> None:
        if variable in ALLOWABLE_RELAXATION:
            if value == 'default':
                value = ALLOWABLE_RELAXATION[variable]
            
            if variable == 'momentum' or variable == 'pressure':
                if value > 1 or value < 0:
                    raise ValueError('Value must be between 0 and 1, not {}'.format(value))
        else:
            raise ValueError('{} not an allowable variable for relaxation'.format(variable))
    
    def format_relaxation(self):
        txt = self._prefix + LINE_BREAK
        for _,value in self.var_value_dict.items():
            txt += str(value) + LINE_BREAK
        
        txt += EXIT_CHAR + EXIT_CHAR + EXIT_CHAR 
        return txt
        
class NISTRealGas:

    """
    class for setting up a real gas model for fluent. This is required if 
    we want to shift from interpolated/constant properties in fluent to 
    NIST real gas models after a few iterations - you may want to do this 
    because NIST real gas models can make convergence very difficult/
    can error fluent out if the solutions are very outside of the reccomended
    range of the correlation used. 

    this will check to make sure that the fluid supplied is allowable within the
    fluent database so that this doesn't cause errors at runtime
    """

    _prefix = '/define/user-defined/real-gas-models/nist-real-gas-model'
    
    def __init__(self,gas: str):
        lib = self.read_lib()
        if '.fld' not in gas:
            gas += '.fld'
        
        if gas not in lib:
            raise FileExistsError('Gas property: {} not available in the NIST Real Gas Library'.format(gas))
        
        self.__gas = gas

    @property
    def gas(self):
        return self.__gas

    def format_real_gas(self):

        txt = self._prefix + LINE_BREAK
        txt += 'yes' + LINE_BREAK
        txt += self.gas + LINE_BREAK
        txt += 'yes' + LINE_BREAK
        #Answer yes to all of the default lookup values
        txt += ' , , , , , , , ' + LINE_BREAK
        #use nist lookup table for thermal property calculation
        txt += 'yes' + LINE_BREAK
        return txt

    def __str__(self):
        return self.format_real_gas()

    @staticmethod
    def read_lib():
        """
        reads in the list of fluids taken from fluent - or rather the real 
        gas models taken from fluent. This is to check
        the supplied fluid again. 

        This is very computationally inefficient practice to do this parsing everytime
        but I am lazy and because this function shoud not be called a bunch of times
        it should be fine.
        """

        path = os.path.split(__file__)[0]
        lib_name = os.path.join(path,'nist_real_gas_lib')
        with open(lib_name,'r') as file:
            string = file.read()
        
        lines = string.split(LINE_BREAK)
        flds = []
        for line in lines:
            fld = [f.strip() for f in line.split(' ')]
            flds += fld
        
        return flds

class Discritization:

    """
    class for changing discritization schemes. This can be useful if you are
    having issues with convergence of the solution i.e. starting out at first order
    and working to second order/higher orders
    """

    _prefix = '/solve/set/discretization-scheme'
    pmap = {'Second Order Upwind':'Second Order',
            'First Order Upwind':'Linear'}
        
    def __init__(self,schemes = 'all',orders = None):

        if schemes == 'all':
            self.__schemes = ALLOWABLE_DISCRITIZATION_SCHEMES
        else:
            if not isinstance(schemes,list):
                schemes = [schemes]

            for scheme in schemes:
                if scheme not in ALLOWABLE_DISCRITIZATION_SCHEMES:
                    raise ValueError('No discritization scheme for field variable: {}'.format(scheme))

            self.__schemes = schemes
        
        if orders is None:
            self.__orders = ['Second Order Upwind' for _ in range(len(self.schemes))]
        else:
            if not isinstance(orders,list):
                orders = [orders for _ in range(len(self.schemes))]
            if len(orders) != len(self.schemes):
                raise AttributeError('Orders and schemes must be of the same length')
            
            for order in orders:
                if order not in ALLOWABLE_DISCRITIZATION_ORDERS: 
                    raise ValueError('order of {} not allowed'.format(order))
            
            self.__orders = orders

    @property
    def schemes(self):
        return self.__schemes
    
    @property
    def orders(self):
        return self.__orders
    
    def format_default_scheme(self,scheme,order):
        """
        schemes for most variables here
        """
        txt = ''
        txt += scheme + LINE_BREAK
        txt += str(ALLOWABLE_DISCRITIZATION_ORDERS[order]) + LINE_BREAK
        return txt
    
    def format_pressure_scheme(self,scheme,order):
        """
        the scheme for presure is different for some reason
        """
        txt = ''
        try:
            order = self.pmap[order]
        except KeyError:
            pass

        txt += scheme + LINE_BREAK
        txt += str(ALLOWABLE_DISCRITIZATION_ORDERS[order]) + LINE_BREAK
        return txt
    
    def format_discritization(self):
        """
        format the discrization scheme for TUI
        """
        txt = self._prefix + LINE_BREAK
        for s,o in zip(self.schemes,self.orders):
            if s == 'pressure':
                txt += self.format_pressure_scheme(s,o)
            else:
                txt += self.format_default_scheme(s,o)
        
        txt += EXIT_CHAR + EXIT_CHAR + EXIT_CHAR
        return txt
    
    def __str__(self):
        """
        string representation
        """
        return self.format_discritization()
    
class FileIO(ABC):

    """
    Base class for file-io in fluent - inherited by various reader's and writers
    """
    _prefix = ''
    _suffix = ''

    def __init__(self,file,*args,**kwargs):

        self.__file = file
    
    @property
    def file(self):
        return self.__file
    
    def __str__(self):
        return self._prefix + ' ' + self.file + self._suffix
    
class CaseReader(FileIO):

    """
    representation of the read-case command
    """
    
    _prefix = 'file/read-case'

    def __init__(self,file):

        if '.cas' not in file and WARNINGS:
            print('Warning:: file: {} is not of .cas type, fluent may be unable to read'.format(file))
        
        super().__init__(file)
    
class BatchCaseReader(CaseReader):

    """
    case reader for batched inputs
    """
    _prefix = 'sync-chdir ..' + LINE_BREAK + 'file/read-case'
    def __init__(self,file):

        super().__init__(file)

    @property
    def _suffix(self):
        return LINE_BREAK + 'sync-chdir {}'.format(self.pwd)
        
    @property
    def pwd(self):
        return self.__pwd

    @pwd.setter
    def pwd(self,pwd):
        self.__pwd = pwd
    
class DataWriter(FileIO):

    """
    representation of the write-data command
    """
    
    _prefix = 'file/write-data'

    def __init__(self,file):

        super().__init__(file)

class CaseWriter(FileIO):

    """
    representation of the write-case command
    """
    
    _prefix = 'file/write-case'

    def __init__(self,file):

        super().__init__(file)

class Solver_Iterator:

    """
    base representation of a solver iterator - this could be replace by a significnatly
    more complex procedure, but for now just iterates a case for a given amount of time
    """
    _prefix = 'solve/iterate'
    def __init__(self,niter = 200):
        self.__niter = niter
    
    @property
    def niter(self):
        return self.__niter
    
    def __str__(self):
        return self._prefix + ' ' + str(self.niter)

class Solver:

    """
    the solver class, must be initialzed with an initializer (for the solver)
    and a Solver_Iterator
    """
    
    def __init__(self,
                 initializer = Initializer(),
                 solver_iterator = Solver_Iterator()):

        self.__initializer = initializer
        self.__solver_iterator = solver_iterator

    @property
    def initializer(self):
        return self.__initializer
    
    @property
    def solver_iterator(self):
        return self.__solver_iterator

    @property
    def usage(self):
        return 'parallel timer usage'

class ConvergenceConditions:

    _prefix = '/solve/convergence-conditions'

    def __init__(self,variables: list,
                      condition = 'all',
                      initial_values_to_ignore = 0,
                      previous_values_to_consider = 1,
                      stop_criterion = 1e-3,
                      print_value = True,
                      ):

        self.__variables = variables
        self.__condition = condition.lower()
        self.__initial_values_to_ignore = initial_values_to_ignore
        self.__previous_values_to_consider = previous_values_to_consider
        self.__print = print_value
        self.__stop_criterion = stop_criterion

    @property
    def variables(self):
        return self.__variables
    
    @property
    def condition(self):
        if self.__condition == 'all':
            return '1'
        elif self.__condition == 'any':
            return '2'
        else:
            raise ValueError('condition must be "all" or "any"')

    @property
    def print_value(self):
        if self.__print:
            return 'yes' 
        else:
            return 'no'
        
    @property
    def initial_values_to_ignore(self):
        return self.__initial_values_to_ignore
    
    @property
    def previous_values_to_consider(self):
        return self.__previous_values_to_consider
    
    @property
    def stop_criterion(self):
        return self.__stop_criterion
    
    def format_condition(self):

        txt = 'condition' + LINE_BREAK
        txt += self.condition + LINE_BREAK
        return txt
    
    def add_condition(self,name):

        txt = 'add' + LINE_BREAK
        txt += name + '-convergence' +  LINE_BREAK
        txt += 'initial-values-to-ignore' + LINE_BREAK
        txt += str(self.initial_values_to_ignore) + LINE_BREAK
        txt += 'previous-values-to-consider' + LINE_BREAK
        txt += str(self.previous_values_to_consider) + LINE_BREAK
        txt += 'print' + LINE_BREAK
        txt += self.print_value + LINE_BREAK
        txt += 'stop-criterion' + LINE_BREAK
        txt += str(self.stop_criterion) + LINE_BREAK
        txt += 'report-defs' + LINE_BREAK
        txt += name + LINE_BREAK
        txt += EXIT_CHAR

        return txt
    
    def format_convergence_conditions(self):

        txt = self._prefix + LINE_BREAK
        txt += self.format_condition()
        txt += 'conv-reports' + LINE_BREAK
        for var in self.variables:
            txt += self.add_condition(var)
        
        txt += EXIT_CHAR + EXIT_CHAR
        return txt

    def __str__(self):
        return self.format_convergence_conditions()

class FluentCase:
    
    """ 
    class for representing a fluent case
    """
    
    def __init__(self,case_name: str):

        self.__case_file = case_name

    @property
    def case_file(self):
        return self.__case_file

class TurbulentBoundarySpecification(ABC):

    _line_break = LINE_BREAK

    def __init__(self):
        pass

    @abstractmethod
    def turbulence_spec(self):
        pass

    def skip_choices(self,num):

        return ''.join(['no' + self._line_break for _ in range(num)])

class TwoEquationTurbulentBoundarySpecification(TurbulentBoundarySpecification):

    def __init__(self):
        self.__tke = 1
        self.__length_scale = None
        self.__intensity = None
        self.__viscosity_ratio = None
        self.__hydraulic_diameter = None

    @property
    def intensity(self):
        return self.__intensity
    
    @property
    def tke(self):
        return self.__tke

    @property
    def length_scale(self):
        return self.__length_scale

    @property
    def viscosity_ratio(self):
        return self.__viscosity_ratio

    @property
    def hydraulic_diameter(self):
        return self.__hydraulic_diameter
    
    @intensity.setter
    def intensity(self,intensity):
        self.__intensity = intensity
    
    @tke.setter
    def tke(self,tke):
        self.__tke = tke

    @length_scale.setter
    def length_scale(self,ls):
        self.__length_scale = ls

    @viscosity_ratio.setter
    def viscosity_ratio(self,vr):
        self.__viscosity_ratio = vr
    
    @hydraulic_diameter.setter
    def hydraulic_diameter(self,hd):
        self.__hydraulic_diameter = hd

    def _intensity_and_hydraulic_diameter(self) -> str:

        txt = self.skip_choices(3)
        txt += 'yes' + self._line_break
        #no to profile
        #txt += 'no' + self._line_break
        txt += str(self.intensity) + self._line_break
        #no to profile
        #txt += 'no' + self._line_break
        txt += str(self.hydraulic_diameter) + self._line_break

        return txt
    
    def turbulence_spec(self,specification):
        if specification == 'intensity and hydraulic diameter':
            return self._intensity_and_hydraulic_diameter()
        elif specification == 'k and omega':
            return self._k_and_omega_specification()
        elif specification == 'k and epsilon':
            return self._k_and_epsilon_specification()
        else:
            raise NotImplementedError('Havent implemented boundary specification mechanisms beyond Intensity and Hydraulic Diameter and K and Omega')
    
class StandardKOmegaSpecification(TwoEquationTurbulentBoundarySpecification):

    def __init__(self):

        super().__init__()
        self.__omega = 1
    
    @property
    def omega(self):
        return self.__omega
    
    @omega.setter
    def omega(self,o):
        self.__omega = o
    
    def _k_and_omega_specification(self) -> str:
        
        #no to profile
        txt = 'yes' + self._line_break + 'no' + self._line_break
        txt += str(self.tke) + self._line_break
        #no to profile
        txt += 'no' + self._line_break
        txt += str(self.omega) + self._line_break
        return txt

class StandardKEpsilonSpecification(TwoEquationTurbulentBoundarySpecification):

    def __init__(self):

        super().__init__()
        self.__tdr = 1

    @property
    def tdr(self):
        return self.__tdr

    @tdr.setter
    def tdr(self,tdr):
        self.__tdr = tdr

    def _k_and_epsilon_specification(self) -> str:
        
        #no to profile
        txt = 'yes' + LINE_BREAK + 'no' + LINE_BREAK
        txt += str(self.tke) + LINE_BREAK
        #no to profile
        txt += 'no' + LINE_BREAK
        txt += str(self.tdr) + LINE_BREAK
        return txt

def _assign_turbulence_model(model:str) -> TurbulentBoundarySpecification:

    assignment = {'ke-standard':StandardKEpsilonSpecification,
                  'kw-standard':StandardKOmegaSpecification}

    try:
        return assignment[model]()
    except KeyError:
        raise ValueError('cannot identify the requested model: {}'.format(model))

class FluentBoundaryCondition(ABC):

    _prefix = '/define/boundary-conditions/'
    _line_break = LINE_BREAK
    _line_start = '/'

    def __init__(self,name: str,
                      boundary_type: str,
                      models: list,
                      solver: str,
                      *args,**kwargs):
        
        if solver not in ALLOWABLE_SOLVERS:
            raise ValueError('solver: {} is not allowed'.format(solver))
        
        for model in models:
            if model not in ALLOWABLE_VISCOUS_MODELS and model not in ALLOWABLE_MODELS:
                raise ValueError('model: {} is not allowed'.format(model))
        
        if boundary_type not in ALLOWABLE_BOUNDARY_TYPES:
            raise ValueError('Cannot parse boundary of type: {}'.format(boundary_type))
        
        self.__btype = boundary_type
        self.__name = name
        self.__models = models
        self.__solver = solver

    @property
    def name(self):
        return self.__name

    @property
    def models(self):
        return self.__models
    
    @property
    def solver(self):
        return self.__solver

    @property
    def btype(self):
        return self.__btype

    def enter_statement(self):
        return self._prefix + self.btype + self._line_break + self.name + self._line_break        

    @abstractmethod
    def format_boundary_condition(self):

        pass

class FluentSolidBoundaryCondition(FluentBoundaryCondition):
    
    def __init__(self,name: str,
                      boundary_type: str,
                      models: list,
                      solver: str,
                      *args,**kwargs):

        super().__init__(name,boundary_type,models,solver,*args,**kwargs)

    def __call__(self):

        return self.format_boundary_condition()
    
    def format_boundary_condition(self):

        return super().format_boundary_condition()
    
class WallBoundaryCondition(FluentSolidBoundaryCondition):


    """
    caf - convective augmentation factor
    """
    
    def __init__(self,name: str,
                      models: list,
                      solver: str,
                      shell_conduction = False):

        super().__init__(name,'wall',models,solver)
        self.__wall_thickness = 0
        self.__generation = 0
        self.__heat_flux = 0
        self.__caf = 1
        
        if shell_conduction:
            self.shell_conduction = 'yes'
        else:
            self.shell_conduction = 'no'

    @property
    def wall_thickness(self):
        return self.__wall_thickness
    
    @property
    def generation(self):
        return self.__generation

    @property
    def heat_flux(self):
        return self.__heat_flux
    
    @property
    def caf(self):
        return self.__caf
    
    @wall_thickness.setter
    def wall_thickness(self,wt):
        self.__wall_thickness = wt
    
    @generation.setter
    def generation(self,g):
        self.__generation = g
    
    @heat_flux.setter
    def heat_flux(self,hf):
        self.__heat_flux = hf
    
    @caf.setter
    def caf(self,caf):
        self.__caf = caf
    
    def __str__(self):

        txt = 'wall thickness: ' + str(self.wall_thickness) + self._line_break
        txt += 'heat generation: ' + str(self.generation) + self._line_break
        txt += 'heat flux: ' + str(self.heat_flux) + self._line_break
        
        return txt
    
    def format_heat_generation(self):

        #no to profile
        txt = 'no' + self._line_break
        txt += str(self.generation) + self._line_break
        return txt
    
    def format_heat_flux(self):

        #no to profile
        txt = 'no' + self._line_break
        txt += str(self.heat_flux) + self._line_break
        return txt

    def format_convective_augmentation(self):

        #no to profile
        txt = 'no' + self._line_break
        txt += str(self.caf) + self._line_break
        return txt
    
    def format_shell_conduction(self):

        #no to shell conduction
        txt = self.shell_conduction + self._line_break
        return txt
    
    def format_boundary_condition(self):

        txt = self.enter_statement()
        txt += str(self.wall_thickness) + self._line_break
        txt += self.format_heat_generation()
        #no to change material
        txt += 'no' + self._line_break
        #no to change Thermal BC Type
        txt += 'no' + self._line_break
        txt += self.format_heat_flux()
        txt += self.format_shell_conduction()
        txt += self.format_convective_augmentation()

        return txt
        
class FluentFluidBoundaryCondition(FluentBoundaryCondition):

    
    """
    Note on variable names: 
    tke - total kinetic energy
    tdr - total dissipation rate
    """ 

    def __init__(self,name:str,
                      boundary_type: str,
                      models: list,
                      solver: str,
                      turbulence_model: str,
                      *args,**kwargs):

        super().__init__(name,boundary_type,models,solver,*args,**kwargs)
        self.__direction_vector = None
        self.__temperature = None
        self.__turbulence_model = _assign_turbulence_model(turbulence_model)
        self.__turbulence_specification = None

    def add_statement(self,text: str,
                           newtext: str) -> str:
        
        return text + self._line_start + newtext + self._line_start
    
    @property
    def turbulence_model(self):
        return self.__turbulence_model
    
    @property
    def direction_vector(self):
        return self.__direction_vector

    @property
    def temperature(self):
        return self.__temperature
      
    @direction_vector.setter
    def direction_vector(self,dv):
        self.__direction_vector = dv

    @temperature.setter
    def temperature(self,t):
        self.__temperature = t
    
    @property
    def turbulence_specification(self):
        return self.__turbulence_specification
    
    @turbulence_specification.setter
    def turbulence_specification(self,ts):
        self.__turbulence_specification = ts
    
    def direction_spec(self) -> str:

        """
        deals with setting the direction vector
        """
    
        txt = ''
        if self.direction_vector is None:
            txt = 'no' + self._line_break
            #normal to boundary
            txt += 'yes' + self._line_break
        else:
            txt = 'yes' + self._line_break + 'yes' + self._line_break
            for i in range(3):
                txt += 'no' + self._line_break + str(self.direction_vector[i]) + self._line_break
            
        return txt
    
    def temperature_spec(self) -> str:

        """
        deals with the tempearture specifications
        """

        #no to profile
        txt = 'no' + self._line_break
        txt += str(self.temperature) + self._line_break

        return txt

    def __call__(self):
        return self.format_boundary_condition(turbulent_specification= self.turbulence_specification)

class MassFlowInlet(FluentFluidBoundaryCondition):

    """
    Mass flow rate boundary condition class
    """
    
    def __init__(self,name: str,
                      models: list,
                      solver: str,
                      turbulence_model: str):

        super().__init__(name,'mass-flow-inlet',models,solver,turbulence_model)
        self.__mass_flow_rate = None
        self.__mass_flux = None
        self.__init_pressure = 0
    
    @property
    def mass_flow_rate(self):
        return self.__mass_flow_rate
    
    @property
    def mass_flux(self):
        return self.__mass_flux

    @property
    def init_pressure(self):
        return self.__init_pressure
    
    @mass_flow_rate.setter
    def mass_flow_rate(self,mfr):
        self.__mass_flow_rate = mfr
    
    @mass_flux.setter
    def mass_flux(self,mf):
        self.__mass_flux = mf

    @init_pressure.setter
    def init_pressure(self,ip):
        self.__init_pressure = ip

    def __str__(self):

        if self.mass_flow_rate is not None:
            txt = 'mass flow: {}'.format(self.mass_flow_rate)
        elif self.mass_flux is not None:
            txt = 'mass flux: {}'.format(self.mass_flux)
        else:
            txt = 'mass: None'
        
        txt += self._line_break
        txt += 'temperature: {}'.format(self.temperature) + self._line_break
        txt += 'pressure: {}'.format(self.init_pressure) + self._line_break
        
        return txt
        
    def mass_spec(self) -> str:

        """ 
        deal with the mass flow rate specifications

        Assumes (for now) that the case has formatted boundary condition accepting either 
        the mass flow rate or the mass flux argument - because of the stupid syntax of the tui 
        this has to be pre-user set. i.e. if the case was configured by the user to accept a 
        MassFlux and you try to input a MassFlowRate it will best case crash the run, worst 
        case it could run with unpredictable results
        """ 

        if self.mass_flow_rate is None:
            txt = 'no' + self._line_break + 'yes' + self._line_break
            #no to profile
            txt += 'no' + self._line_break
            if self.mass_flux is None:
                txt = 'yes' + self._line_break + 'no' + self._line_break

                #no to profile
                txt += 'no' + self._line_break 
                txt += '1' + self._line_break

            else:
                txt += str(self.mass_flux) + self._line_break
        
        else:
            if self.mass_flux is not None:
                raise ValueError('You cannot specify both the mass flux and the mass flow rate')
            
            txt = 'yes' + self._line_break + 'no' + self._line_break
            #no to profile
            txt += 'no' + self._line_break
            txt += str(self.mass_flow_rate) + self._line_break
        

        return txt

    def pressure_spec(self) -> str:

        """
        deals with initial pressure specifications
        """

        #no to profile
        txt = 'no' + self._line_break
        txt += str(self.init_pressure) + self._line_break
        
        return txt

    def format_boundary_condition(self,turbulent_specification = 'K and Epsilon') -> str:

        txt = self.enter_statement()
        txt += 'yes' + self._line_break
        txt += self.mass_spec()
        txt += self.temperature_spec()
        txt += self.pressure_spec()
        txt += self.direction_spec()
        txt += self.turbulence_model.turbulence_spec(turbulent_specification)
        
        return txt

class PressureOutlet(FluentFluidBoundaryCondition):

    def __init__(self,name:str,
                      models: list,
                      solver: str,
                      turbulence_model: str):

        super().__init__(name,'pressure-outlet',models,solver,turbulence_model)
        self.__pressure = None

    @property
    def pressure(self):
        return self.__pressure
    
    @pressure.setter
    def pressure(self,p):
        self.__pressure = p
    
    def pressure_spec(self):
        """
        specifications for the pressure
        """ 
        #no to profile
        txt = 'no' + self._line_break
        txt += str(self.pressure) + self._line_break
        return txt

    def format_boundary_condition(self,turbulent_specification = 'K and Epsilon') -> str:

        txt = self.enter_statement()
        txt += 'yes' + self._line_break
        txt += self.pressure_spec()
        txt += self.temperature_spec()
        txt += self.direction_spec()
        txt += self.turbulence_model.turbulence_spec(turbulent_specification)
        #no to a couple of end options that are pretty obscure:
        #Radial Equiliibrium Pressure Distribution
        #Average Pressure Specification
        #Specify Targeted Mass Flow Rate
        txt += 'yes' + self._line_break + ''.join(['no' + self._line_break for _ in range(3)])

        return txt
    
    def __str__(self):

        txt = 'pressure: {}'.format(self.pressure) + self._line_break
        txt += 'temperature: {}'.format(self.temperature) + self._line_break
        
        return txt

class FluentRun:
    
    """
    class for producing a fluent batch job file using provided information.
    The intent is to make this as easy to use as possible, so the only required argument
    is the case_file as a string argument, and everything else should be handled
    automatically
    """
    
    _line_break = LINE_BREAK
    def __init__(self,case_file: str,
                      output_name = 'result',
                      transcript_file = 'solution.trn',
                      reader = CaseReader,
                      data_writer = DataWriter,
                      case_writer = CaseWriter,
                      solver = Solver(),
                      convergence_condition = None,
                      boundary_conditions = [],
                      *args,**kwargs):

        self.__case = FluentCase(case_file)
        self.__reader = reader(case_file)
        self.__case_writer = case_writer(output_name + '.cas')
        self.__data_writer = data_writer(output_name + '.dat')
        self.__solver = solver
        self.__transcript_file = transcript_file
        self.__file_name = case_file
        self.__boundary_conditions = boundary_conditions
        self.__convergence_condition = convergence_condition

    @property
    def case(self):
        return self.__case

    @property
    def file_name(self):
        return self.__file_name
    
    @file_name.setter
    def file_name(self,fn):
        self.__file_name = fn
    
    @property
    def reader(self):
        return self.__reader
    
    @property
    def case_writer(self):
        return self.__case_writer
    
    @property
    def convergence_condition(self):
        return self.__convergence_condition

    @property
    def data_writer(self):
        return self.__data_writer
    
    @property 
    def solver(self):
        return self.__solver
    
    @property
    def transcript_file(self):
        return self.__transcript_file
    
    @property
    def boundary_conditions(self):
        return self.__boundary_conditions
    
    @boundary_conditions.setter
    def boundary_conditions(self,bc):
        self.__boundary_conditions = bc
    
    def boundary_conditions_spec(self):

        txt = LINE_BREAK + ';Boundary Conditions' + LINE_BREAK + LINE_BREAK
        for bc in self.boundary_conditions:
            txt += bc()
        

        return txt

    def format_convergence_condition(self):
        if self.convergence_condition is None:
            return ''
        else:
            txt = LINE_BREAK + ';Convergence Conditions' + LINE_BREAK + LINE_BREAK
            return txt + str(self.converence_condition)
    
    def format_fluent_file(self) -> str:

        """
        format the fluent input file
        """
        
        txt = 'file/start-transcript ' + self.transcript_file + self._line_break     
        
        txt +=  str(self.reader) + self._line_break
        txt += self.format_convergence_condition()
        txt += self.boundary_conditions_spec()
        txt += str(self.solver.initializer) + self._line_break
        txt += str(self.solver.solver_iterator) + self._line_break
        txt += str(self.solver.usage) + self._line_break
        txt += str(self.case_writer) + self._line_break
        txt += str(self.data_writer) + self._line_break
        txt += 'exit' + self._line_break

        return txt

    def __call__(self):
        return self.format_fluent_file()

    def write(self,f) -> None:

        try:
            with open(f,'w',newline = LINE_BREAK) as file:
                file.write(self.format_fluent_file())

        except TypeError:
            f.write(self.format_fluent_file())

    @classmethod
    def fromfile(cls,file_name):
        
        _cls = cls()
        _cls.file_name = file_name
        return _cls
    
def get_test_mfr():

    mass_flow = MassFlowInlet('inner_tube_inlet',['energy','ke-standard'],'pressure-based')
    mass_flow.mass_flow_rate = 0.04
    mass_flow.init_pressure = 4.2e6
    mass_flow.temperature = 360
    mass_flow.hydraulic_diameter = 0.013
    mass_flow.intensity = 3.8

    return mass_flow

def get_test_pressure():
    pressure_outlet = PressureOutlet('outer_tube_outlet',['energy','ke-standard'],'pressure-based')
    pressure_outlet.pressure = 4.3e6
    pressure_outlet.temperature = 320
    pressure_outlet.hydraulic_diameter = 0.003
    pressure_outlet.intensity = 3.8

    return pressure_outlet

def test_fluent_run():
    
    mfr = get_test_mfr()
    po = get_test_pressure()

    solver = Solver(solver_iterator= Solver_Iterator(niter = 500))
    ff = FluentRun('test.cas',solver = solver, boundary_conditions= [mfr,po])
    ff.write('fleunt_input.fluent')

def test_mass_flow_rate_bc():

    mass_flow = get_test_mfr()
    txt = mass_flow(turbulent_specification = 'Intensity and Hydraulic Diameter')
    print(txt)

def test_pressure_outlet_bc():

    po = get_test_pressure()
    print(po())

def test_format_convergence_conditions():

    cc = ConvergenceConditions(['p-out','max-temp'])
    print(str(cc))

def test_discritization():

    disc = Discritization(orders = 'First Order Upwind')
    print(disc)

def test_real_gas():

    nrg = NISTRealGas('co2')
    print(nrg)

def test_scalar_relaxation():

    relaxation = ScalarRelaxation(['density','temperature'],[0.8,0.7])
    print(relaxation)

def test_equation_relaxation():

    #relax = EquationRelaxation('pressure',0.8)
    #print(relax)
    relax = EquationRelaxation.from_dict({'pressure':0.7})
    print(relax)

def main():

    #test_fluent_run()
    #test_mass_flow_rate_bc()
    #test_pressure_outlet_bc()
    #test_format_convergence_conditions()
    #test_discritization()
    #test_real_gas()
    #test_scalar_relaxation()
    test_equation_relaxation()

if __name__ == '__main__':
    main()