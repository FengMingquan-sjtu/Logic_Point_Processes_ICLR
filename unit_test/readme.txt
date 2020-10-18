1. testing codes follows pytest format, thus:
    1) file names in the form "test_*.py"
    2) class names start with "Test_"
    3) function names start with "test_"

2. Notice when grouping tests inside classes, each test has a unique instance of the class.
    Having each test share the same class instance would be very detrimental to test isolation