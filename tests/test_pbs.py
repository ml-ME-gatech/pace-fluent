import unittest
from pace_fluent.pace_fluent.pbs import FluentPBS

class TestFluentPBS(unittest.TestCase):
    
    test_file = 'test_pbs/test.pbs'
    def test_pbs_read_write(self):

        pbs = FluentPBS('fluent_input',
                WALLTIME = 20*60*60,
                MEMORY = 24,
                memory_request = 't',
                N_NODES = 1,
                N_PROCESSORS = 2,
                email = 'mlanahan3@gatech.edu')
        
        pbs.serialize(self.test_file)
        pbsread = FluentPBS.from_file(self.test_file)
               
def main():
    unittest.main()

if __name__ == '__main__':
    main()