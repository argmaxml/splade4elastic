import sys, unittest, json
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from splade4elastic import *


class BasicTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        splade = MLMBaseRewriter("bert-base-uncased")
        text = "The quick brown fox jumps over the lazy dog"
        print(splade.query_expand(text))
        self.assertTrue(True)

    def test_splade(self):
        splade = SpladeRewriter("bert-base-uncased")
        text = "The quick brown fox jumps over the lazy dog"
        print(splade.query_expand(text))
        self.assertTrue(True)

    def test_linear(self):
        splade = LinearMLMRewriter("bert-base-uncased")
        text = "The quick brown fox jumps over the lazy dog"
        print(splade.query_expand(text))
        self.assertTrue(True)



if __name__ == '__main__':
    unittest.main()
