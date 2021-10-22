from unittest import TestCase
import unittest
from post import SurfaceIntegrals, SurfaceIntegralFile, SurfaceIntegralBatch
import os

path = 'test-post/case-423'
name = 'result'

class TestGetSurfaceIntegrals(TestCase):

    fname = os.path.join(path,name) + '.cas'

    def test_one(self):
        pass
        """
        si = SurfaceIntegrals(self.fname,'13','temperature','area-weighted-avg')
        si()
        """
    
    def test_multiple(self):
        pass
        """
        sib = SurfaceIntegralBatch('test-post','13','temperature','area-weighted-avg')
        sib.collect_surface_integrals()
        """
    
    def test_df(self):
        sib = SurfaceIntegralBatch('test-post','13','temperature','area-weighted-avg')
        df = sib.readdf()
        print(df)


"""
class TestSurfaceIntegralFile(TestCase):

    fname = os.path.join(path,'test2')
    ifname = os.path.join(path,'test')

    def test_file_parser(self):

        sif = SurfaceIntegralFile(self.fname)
        attrs = sif.read()
    
    def test_incorrect_file_parser(self):
        sif = SurfaceIntegralFile(self.ifname)
        attrs = sif.read()
        self.assertEqual(attrs['value'],None)
        self.assertEqual(attrs['boundary'],None)
"""

def main():
    unittest.main()

if __name__ == '__main__':
    main()

