import sys
sys.path.extend(["../","./"])
from model.point_process import Point_Process 

"""
testing codes follows pytest format, thus:
1) file names in the form "test_*.py"
2) class names start with "Test_"
3) function names start with "test_"

"""
class Test_Point_Process:
    """ Testing gourp for ./model/point_process.py
    Notice when grouping tests inside classes is that each test has a unique instance of the class.
    Having each test share the same class instance would be very detrimental to test isolation
    """
    def test_intensity(self):
        pass


