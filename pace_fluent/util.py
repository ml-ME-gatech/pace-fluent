#native imports
from typing import Type
from pandas import DataFrame,Series
import numpy as np
from pandas.core.indexes.base import Index

#package imports
from fluentpy.tui.fluent import UDF, PressureOutlet,FluentFluidBoundaryCondition, MassFlowInlet, TurbulentBoundarySpecification, WallBoundaryCondition
from fluentpy.tui.fluent import FluentBoundaryCondition

""""
Author: Michael Lanahan
Date Created: 08.06.2021
Last Edit: 11.23.2021

utility functions and classes for working with Fluent Batch Files
and pace utility formatting
"""

TURBULENT_SPECIFICATIONS =  {'k and epsilon':['turbulent kinetic energy','turbulent dissipation rate'],
                             'k and omega':['turbulent kinetic energy','specific dissipation rate'],
                             'intensity and length scale':['turbulent intensity','turbulent length scale'],
                             'intensity and viscosity ratio':['turbulent intensity','turbulent viscosity ratio'],
                             'intensity and hydraulic diameter':['turbulent intensity','hydraulic diameter']}

SOLVER = 'pressure-based'
MODELS = ['energy','ke-standard','kw-standard']
ALLOWABLE_BOUNDARY_TYPES = ['mass-flow-inlet','pressure-outlet','wall']

def split_boundary_name(string: str,
                        delim = ':') -> tuple:
    """
    expects a boundary condition input in the format
    name:input:type 
    """ 
    return (s.strip() for s in string.split(delim))

def sort_boundary_list(blist: list,
                       delim = ':') -> dict:

    """
    essentially develops a "log" of boundary conditions that is a dictionary
    the keys of the dictionary are the names:type and the values of the dictionary
    are lists of the variables for that particular boundary condition
    """

    log = {}
    for item in blist:
        name,variable,btype = split_boundary_name(item)
        
        if btype not in ALLOWABLE_BOUNDARY_TYPES:
            raise TypeError('boundary condiiton type specified by: {} is not allowed'.format(btype))
        
        name = name + delim + btype
        try:
            log[name].append(variable)
        except KeyError:
            log[name] = [variable]
    
    return log

def partition_boundary_table(df: DataFrame,
                             turbulence_model: str,
                             delim = ':') -> DataFrame:

    """
    partition the input boundary table into a dataframe where the 
    columns are in the format

    name:type

    and the rows contain the boundary conditions of the appropriate type. The number
    of rows will be identical to the number of inputs rows of the table
    """
    
    models = _infer_models_from_headers(df.columns)
    sorted_df = sort_boundary_list(df.columns)
    boundary_list = dict.fromkeys(sorted_df.keys())
    for name,svars in sorted_df.items():
        _name = name.split(delim)
        cols = [_name[0]  + delim + c + delim + _name[1] for c in svars]
        bc_map = dict(zip(sorted_df[name],cols))
        _df = df[cols]
        boundary_list[name] = []
        for i in range(_df.shape[0]):
            if isinstance(_df.iloc[i],Series):
                __df = df.iloc[i]
            else:
                __df = df.iloc[i].squeeze()
            boundary_list[name].append(
                                       make_boundary_condition_from_series(name.split(delim)[1],
                                                                           name.split(delim)[0],
                                                                           __df,
                                                                           bc_map,
                                                                           turbulence_model,
                                                                           models)
                                     )
    
    bdf = DataFrame.from_dict(boundary_list)
    bdf.index = df.index
    return bdf

def _infer_models_from_headers(columns: list) -> list:
    """
    simple function for infering models from the supplied headers. 
    This is not very sophisiticated at the moment and should be carefully
    tested later - right now there are not really any consequences though
    """

    models = []
    for column in columns:
        if 'temperature' in column or 'heat flux' in column or 'htc' in column:
            models.append('energy')
        
        if 'turbulent dissipation rate' or 'intensity' in column:
            models.append('ke-standard')
        
        if 'specific dissipation rate' in column:
            models.append('kw-standard')
    
    return list(set(models))


def _parse_udf_from_str(string : str) -> UDF:
    """
    expects the specification of a UDF as a string
    from a table to have the following format: 

    <file_name#condition_name#udf_name#data_name>

    OR

    <file_name#condition_name#udf_name#data_name#compile>

    the second option allows for compilation of the UDF during runtime

    example
    <htc.c#convection_coefficient#udf#HTC::UDF>

    The file name can be an absolute or relative path to the file
    """

    string = string.strip()
    
    if string[0] == '<' and string[-1] == '>':
        string = string[1:-1]
        try:
            file_name,condition_name,udf_name,data_name =\
                tuple([sstr.strip() for sstr in string.split('#')])
            
            udf = UDF(file_name,udf_name,data_name,condition_name)
            
            return udf
        except ValueError:
            file_name,condition_name,udf_name,data_name,compile =\
                tuple([sstr.strip() for sstr in string.split('#')])
            
            if compile.lower() == 'compile':
                compile = True
            else:
                raise TypeError('{} not a valid option for compile field'.format(compile))
            
            udf = UDF(file_name,udf_name,data_name,condition_name)
            udf.compile = True
            
            return udf

        except ValueError:
            raise ValueError('UDF Specified by: {} not valid'.format(string))

    else:
        raise ValueError('string does not specify a UDF')
        
def enable_udf(boundary_class: FluentBoundaryCondition) -> callable:

    """
    decorator function for specifying the boundary conditions from the table. 
    Handles all of the assignment, and parsing of the UDF's but requires that
    the decorated function make the variable "mapping" available so that
    it is understood how to map the fields in the table to the attributes
    of the particular boundary condition class
    """
    
    def udf_enabled_boundary_condition(bc_assignment: callable) -> callable:
        
        def _make_boundary_condition(self,series:Series,
                                            name: str,
                                            bc_map: dict,
                                            models: list, 
                                            turbulence_model = None) -> FluentBoundaryCondition:
            
            if turbulence_model is None:
                boundary_condition = boundary_class(name,models,SOLVER)
            else:
                boundary_condition = boundary_class(name,models,SOLVER,turbulence_model)
            
            for table_name,attr_name in self.mapping.items():
                #handle the case of a UDF
                try:
                    if bc_map[table_name] in series:
                        if isinstance(series.loc[bc_map[table_name]],str):
                            try:
                                udf = _parse_udf_from_str(series.loc[bc_map[table_name]])
                                boundary_condition.add_udf(udf)
                            except ValueError:
                                try:
                                    series.loc[bc_map[table_name]] = float(series.loc[bc_map[table_name]])
                                except TypeError:
                                    raise TypeError('cannot convert value: {} from column {} to float'.format(series.loc[bc_map[table_name]],table_name))
                                
                        else:
                            boundary_condition.__setattr__(attr_name,series.loc[bc_map[table_name]])
                except KeyError:
                    pass

            if turbulence_model is None:
                return boundary_condition
            else:
                return _handle_turbulent_boundary_specifications(boundary_condition,
                                                                 series,
                                                                 bc_map)

        return _make_boundary_condition
    
    return udf_enabled_boundary_condition


def make_boundary_condition_from_series(btype: str,
                                        name: str,
                                        series: Series,
                                        bc_map: dict,
                                        turbulence_model: str,
                                        models: list):

    """"
    The treatment of a row representation of a boundary condition
    This is an entry function to parse based upon the type of boundary
    condition we are working with
    """

    mapping = {'pressure-outlet':MakePressureOutletBoundaryCondition(),
               'mass-flow-inlet':MakeMassFlowInletBoundaryCondition(),
               'wall':MakeWallBoundaryCondition()}
    
    return mapping[btype](series,name,bc_map,models,turbulence_model = turbulence_model)

class MakeWallBoundaryCondition:
    
    """
    handles the wall boundary condition formation from a table
    """
    def __init__(self):
        self.mapping = {'heat flux':'heat_flux',
                        'heat_generation':'generation',
                        'wall thickness':'wall_thickness',
                        'htc':'convection_coefficient',
                        'free stream temperature':'free_stream_temperature',
                        'convective augmentation factor':'caf'}
    
    @enable_udf(WallBoundaryCondition)
    def __call__(self,*args) -> WallBoundaryCondition:
        pass
                                    
class MakePressureOutletBoundaryCondition:

    """
    handles the pressure outlet boundary condition formation from a table
    """

    def __init__(self):
        self.mapping = {'temperature':'temperature',
                        'pressure':'pressure'}
    
    @enable_udf(PressureOutlet)
    def __call__(self,*args,**kwargs) -> PressureOutlet:
        pass


class MakeMassFlowInletBoundaryCondition:
    """
    handles the mass flow inlet boundary condition formation from a table
    """
    
    def __init__(self):
        self.mapping = {'temperature':'temperature',
                        'initial pressure':'init_pressure',
                        'mass flow rate':'mass_flow_rate',
                        'mass flux':'mass_flux'}
    
    @enable_udf(MassFlowInlet)
    def __call__(self,*args,**kwargs) -> MassFlowInlet:
        pass

def _handle_turbulent_boundary_specifications(boundary_condition: FluentFluidBoundaryCondition,
                                              series: Series,
                                              bc_map: dict,
                                              ) -> FluentFluidBoundaryCondition:

    """
    handles turbulent input specifications from a table

    !!IMPORTANT!!
    if you provide multiple mechanisms for specifiying turbulence at the boundary,
    this will pick the first of which that can be succesfully identified from the table 
    in the same order that the Fluent TUI requests - it will NOT error out.  
    """
    
    if TURBULENT_SPECIFICATIONS['k and epsilon'][0] in bc_map and TURBULENT_SPECIFICATIONS['k and epsilon'][1] in bc_map:
        boundary_condition.turbulence_model.tke = series.loc[bc_map['turbulent kinetic energy']]
        boundary_condition.turbulence_model.tdr = series.loc[bc_map['turbulent dissipation rate']]
        boundary_condition.turbulence_specification = 'k and epsilon'
    elif TURBULENT_SPECIFICATIONS['k and omega'][0] in bc_map and TURBULENT_SPECIFICATIONS['k and omega'][1] in bc_map:
        boundary_condition.turbulence_model.tke = series.loc[bc_map['turbulent kinetic energy']]
        boundary_condition.turbulence_model.omega = series.loc[bc_map['specific dissipation rate']]
        boundary_condition.turbulence_specification = 'k and omega'
    elif TURBULENT_SPECIFICATIONS['intensity and length scale'][0] in bc_map and TURBULENT_SPECIFICATIONS['intensity and length scale'][1] in bc_map:
        boundary_condition.turbulence_model.intensity = series.loc[bc_map['turbulent intensity']]
        boundary_condition.turbulence_model.length_scale = series.loc[bc_map['turbulent length scale']]
        boundary_condition.turbulence_specification = 'intensity and length scale'
    elif TURBULENT_SPECIFICATIONS['intensity and viscosity ratio'][0] in bc_map and TURBULENT_SPECIFICATIONS['intensity and viscosity ratio'][1] in bc_map:
        boundary_condition.turbulence_model.intensity = series.loc[bc_map['turbulent intensity']]
        boundary_condition.turbulence_model.viscosity_ratio = series.loc[bc_map['turbulent viscosity ratio']]
        boundary_condition.turbulence_specification = 'intensity and viscosity ratio'
    elif TURBULENT_SPECIFICATIONS['intensity and hydraulic diameter'][0] in bc_map and TURBULENT_SPECIFICATIONS['intensity and hydraulic diameter'][1] in bc_map:
        boundary_condition.turbulence_model.intensity = series.loc[bc_map['turbulent intensity']]
        boundary_condition.turbulence_model.hydraulic_diameter = series.loc[bc_map['hydraulic diameter']]
        boundary_condition.turbulence_specification = 'intensity and hydraulic diameter'
    
    else:
        raise ValueError('Cannot specify turbulent parameters based upon input')

    return boundary_condition