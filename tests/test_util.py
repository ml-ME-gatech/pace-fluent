from pandas import DataFrame,Series
import numpy as np
from pace_fluent.pace_fluent.util import * 

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
    
def test_udf_specification():
    data = np.expand_dims(np.array([1.0,2.0]),axis = 1)

    table = DataFrame(data,
                      columns = ['heated-surf:free stream temperature:wall'])
    
    table['heated-surf:htc:wall'] = ['<test.c#convection_coefficient#udf#HTC::LIBUDF>' for _ in range(2)]


    df = partition_boundary_table(table,None)

    print(df.columns)
    bc1 = df['heated-surf:wall'].iloc[0]
    text = bc1.format_boundary_condition()
    print(text)


def main():

    #test_single_pressure_outlet_format()
    #test_single_mass_flow_inlet_format()
    #test_wall_format()
    #test_multiple_table()
    test_udf_specification()

if __name__ == '__main__':
    main()