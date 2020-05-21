import unittest
import utils

class TestStringMethods(unittest.TestCase):
    
    def test_emb(self):
        # wv = utils.load_biosentvec_emb()
        wv = utils.load_glove_emb()
        print(wv['AIDS'])

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()