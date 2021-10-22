import unittest
from pace_fluent.pace_fluent.fluent import FluentRun

class TestFluentRun(unittest.TestCase):
    
    test_file = 'test_fluent/test.run'
    def test_fluent_run_read_write(self):

        fluent_run = FluentRun('test.cas')
        fluent_run.serialize(self.test_file)

        fluent_run_read = FluentRun.from_file(self.test_file)


def main():
    unittest.main()

if __name__ == '__main__':
    main()