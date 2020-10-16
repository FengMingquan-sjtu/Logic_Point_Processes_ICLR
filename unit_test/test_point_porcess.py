import sys
sys.path.extend(["../","./"])
from model.point_process import Point_Process 

class Test_Point_Process:
    """ Testing gourp for ./model/point_process.py
    Notice when grouping tests inside classes is that each test has a unique instance of the class.
    Having each test share the same class instance would be very detrimental to test isolation
    """
    def test_intensity(self):
        pp = Point_Process()
        ret = pp.intensity() 
        assert (ret == 1)

if __name__ == '__main__':
    unittest.main()
