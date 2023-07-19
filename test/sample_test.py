import sys, unittest, json
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))


class SampleTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_example(self):
        self.assertEqual(1,1)



if __name__ == '__main__':
    unittest.main()
