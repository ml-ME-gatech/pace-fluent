#native imports
import unittest

#native-ish imports
import pandas as pd

#package imports
from pace_fluent.pace_fluent.submit import FluentSubmission,FluentBatchSubmission
from pace_fluent.pace_fluent.pbs import FluentPBS
from pace_fluent.pace_fluent.fluent import FluentRun,Solver_Iterator,Solver
from pace_fluent.pace_fluent.post import SurfaceIntegrals

class TestFluentSubmit(unittest.TestCase):
    
    test_file = 'test_submit/test.submit'
    test_batch_folder = 'test_submit/test_batch'
    test_batch_add_folder = 'test_submit/test_batch_add'
    test_csv_file = 'test_submit/He_tin_600C_1.csv'

    def test_fluent_submit_read_write(self):

        fluent_run = FluentRun('test.cas')
        pbs = FluentPBS('fluent_input',
                        WALLTIME = 20*60*60,
                        MEMORY = 24,
                        memory_request = 't',
                        N_NODES = 1,
                        N_PROCESSORS = 2,
                        email = 'mlanahan3@gatech.edu')

        submit = FluentSubmission(fluent_run,pbs)
        submit.serialize(self.test_file)

        submit_read = FluentSubmission.from_file(self.test_file)

    def test_fluent_batch_read_write(self):

        fluent_run = FluentRun('test.cas')
        pbs = FluentPBS('fluent_input',
                        WALLTIME = 20*60*60,
                        MEMORY = 24,
                        memory_request = 't',
                        N_NODES = 1,
                        N_PROCESSORS = 2,
                        email = 'mlanahan3@gatech.edu')

        submit1 = FluentSubmission(fluent_run,pbs)
        
        fluent_run2 = FluentRun('test1.cas')
        submit2 = FluentSubmission(fluent_run2,pbs.copy())

        batchsubmit = FluentBatchSubmission([submit1,submit2],seperator= '')
        batchsubmit.bash_submit(self.test_batch_folder,purge= True)

        batchsubmit_read = FluentBatchSubmission.from_batch_cache(self.test_batch_folder)

    def test_fluent_batch_add(self):
        parameter_table = pd.read_csv(self.test_csv_file,index_col= 0)
        parameter_table.index = parameter_table.index + 6

        pbs = FluentPBS('fluent_input',
                        WALLTIME = 6*60*60,
                        MEMORY = 8,
                        N_NODES = 1,
                        N_PROCESSORS = 8,
                        email = 'mlanahan3@gatech.edu',
                        mpi_option = 'pcmpi')

        solver_iterator = Solver_Iterator(niter = 650)
        solver = Solver(solver_iterator = solver_iterator)

        si_outlet = SurfaceIntegrals('modified_HEMJ60deg-3.cas','13','temperature','area-weighted-avg')
        post = [si_outlet]

        submission = FluentBatchSubmission.from_frame('modified_HEMJ60deg-3.cas',
                                                    'ke-standard',
                                                    parameter_table,
                                                    pbs,
                                                    solver = solver,
                                                    post = post)

        parameter_table.index += parameter_table.shape[0]
        submission2 = FluentBatchSubmission.from_frame('modified_HEMJ60deg-3.cas',
                                                    'ke-standard',
                                                    parameter_table,
                                                    pbs,
                                                    solver = solver,
                                                    post = post)

        submission3 = submission + submission2
        submission3.bash_submit(self.test_batch_add_folder,purge = True)

        #test addition with no df in second batch
        fluent_run = FluentRun('test.cas')
        pbs = FluentPBS('fluent_input',
                        WALLTIME = 20*60*60,
                        MEMORY = 24,
                        memory_request = 't',
                        N_NODES = 1,
                        N_PROCESSORS = 2,
                        email = 'mlanahan3@gatech.edu')

        submit1 = FluentSubmission(fluent_run,pbs)
        
        fluent_run2 = FluentRun('test1.cas')
        submit2 = FluentSubmission(fluent_run2,pbs.copy())

        batchsubmit = FluentBatchSubmission([submit1,submit2],seperator= '')
        submission3 = submission + batchsubmit
        submission3.bash_submit(self.test_batch_add_folder,purge = True)

def main():
    unittest.main()

if __name__ == '__main__':
    main()