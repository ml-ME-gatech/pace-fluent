#native imports
from pandas import DataFrame,Series
import numpy as np

#package imports
from fluentpy.tui.fluent import PressureOutlet,FluentFluidBoundaryCondition, MassFlowInlet, WallBoundaryCondition

""""
Author: Michael Lanahan
Date Created: 08.06.2021
Last Edit: 09.09.2021

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
                                                                           turbulence_model)
                                     )
    
    bdf = DataFrame.from_dict(boundary_list)
    bdf.index = df.index
    return bdf


def make_boundary_condition_from_series(btype: str,
                                        name: str,
                                        series: Series,
                                        bc_map: dict,
                                        turbulence_model: str):

    """"
    The treatment of a row representation of a boundary condition
    This is an entry function to parse based upon the type of boundary
    condition we are working with
    """

    if btype == 'pressure-outlet':
        return _make_pressure_outlet_boundary_condition(name,series,bc_map,turbulence_model)
    elif btype == 'mass-flow-inlet':
        return _make_mass_flow_inlet_boundary_condition(name,series,bc_map,turbulence_model)
    elif btype == 'wall':
        return _make_wall_boundary_condition(name,series,bc_map)
    
    else:
        raise TypeError('boundary condition of type: {} not supported for table parsing'.format(btype))

def _make_wall_boundary_condition(name: str,
                                  series: Series,
                                  bc_map: dict) -> WallBoundaryCondition:
    
    """
    handles the wall boundary condition formation from a table
    """

    wall = WallBoundaryCondition(name,MODELS,SOLVER)
    try:
        wall.heat_flux = series.loc[bc_map['heat flux']]
    except KeyError:
        pass

    try:
        wall.generation = series.loc[bc_map['heat generation']]
    except KeyError:
        pass

    try:
        wall.wall_thickness = series.loc[bc_map['wall thickness']]
    except KeyError:
        pass

    return wall

def _make_pressure_outlet_boundary_condition(name: str,
                                             series: Series,
                                             bc_map: dict,
                                             turbulence_model: str) -> PressureOutlet:

    """
    handles the pressure outlet boundary condition formation from a table
    """
    
    pressure_outlet = PressureOutlet(name,MODELS,SOLVER,turbulence_model)
    try:
        """
        it may be the case that energy equation is not solved for and temperature
        is not required to be specified
        """
        pressure_outlet.temperature = series.loc[bc_map['temperature']]
    except KeyError:
        pass

    #pressure is always a required specification
    pressure_outlet.pressure = series.loc[bc_map['pressure']]

    return _handle_turbulent_boundary_specifications(pressure_outlet,
                                                     series,
                                                     bc_map)

def _make_mass_flow_inlet_boundary_condition(name: str,
                                             series: Series,
                                             bc_map: dict,
                                             turbulence_model: str) -> MassFlowInlet:
    
    """
    handles the mass flow inlet boundary condition formation from a table
    """
    
    mass_flow_inlet = MassFlowInlet(name,MODELS,SOLVER,turbulence_model)

    try:
        """
        it may be the case that energy equation is not solved for and temperature
        is not required to be specified
        """
        mass_flow_inlet.temperature = series.loc[bc_map['temperature']]
    except KeyError:
        pass
    
    #initial pressure is always required
    mass_flow_inlet.init_pressure = series.loc[bc_map['initial pressure']]
    
    #we may be either given a mass flow rate or a mass flux
    try:
        mass_flow_inlet.mass_flow_rate = series.loc[bc_map['mass flow rate']]
    except KeyError:
        mass_flow_inlet.mass_flux= series.loc[bc_map['mass flux']]

    return _handle_turbulent_boundary_specifications(mass_flow_inlet,
                                                     series,
                                                     bc_map)
    
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


def test_single_pressure_outlet_format():

    data = np.array([[4e6,300,3.8,0.003],
                     [4.1e6,300,3.8,0.003]])

    table = DataFrame(data,
                      columns = ['outer_tube_outlet:pressure:pressure-outlet',
                                 'outer_tube_outlet:temperature:pressure-outlet',
                                 'outer_tube_outlet:turbulent intensity:pressure-outlet',
                                 'outer_tube_outlet:hydraulic diameter:pressure-outlet']
                        )
    
    df = partition_boundary_table(table,'ke-standard')
    print(df)


def test_single_mass_flow_inlet_format():

    data = np.array([[0.04,300,4e6,3.8,0.013],
                     [0.05,300,4e6,3.8,0.013]])

    table = DataFrame(data,
                     columns = ['inner_tube_inlet:mass flow rate:mass-flow-inlet',
                                'inner_tube_inlet:temperature:mass-flow-inlet',
                                'inner_tube_inlet:initial pressure:mass-flow-inlet',
                                'inner_tube_inlet:turbulent intensity:mass-flow-inlet',
                                'inner_tube_inlet:hydraulic diameter:mass-flow-inlet'])
    
    df = partition_boundary_table(table,'ke-standard')
    print(df)

def test_wall_format():

    data = np.expand_dims(np.array([1.0,2.0]),axis = 1)

    table = DataFrame(data,
                      columns = ['heated-surf:heat flux:wall'])
    
    df = partition_boundary_table(table,None)

    print(df)

def test_multiple_table():

    data = np.array([[4e6,300,3.8,0.003,0.04,300,4e6,3.8,0.013],
                     [4.1e6,300,3.8,0.003,0.05,300,4e6,3.8,0.013],
                     [4.2e6,300,3.8,0.003,0.06,300,4e6,3.8,0.013]])

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
    
    table.to_csv('example_inputs.csv')
    df = partition_boundary_table(table,'ke-standard')
    print(df)
    

def main():

    test_single_pressure_outlet_format()
    test_single_mass_flow_inlet_format()
    test_wall_format()
    test_multiple_table()

if __name__ == '__main__':
    main()