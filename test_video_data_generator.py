import unittest
import pandas as pd
from video_data_generator import VideoDataGenerator

class TestVideoDataGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv("test.csv")
        cls.vdg = VideoDataGenerator(df, "data", file_col="video_name", y_col="tag")

    def test_len(self):
        self.assertEqual(len(self.vdg), 10)

    def test_getitem(self):
        self.assertEqual(self.vdg[0][0][0].shape, (4, 120, 224, 224, 3))
        self.assertEqual(self.vdg[0][0][1].shape, (4, 120))
        self.assertEqual(self.vdg[0][1].shape, (4,))

    def test_on_epoch_end(self):
        first_batch = self.vdg[0]
        self.vdg.on_epoch_end()
        self.assertNotEqual(self.vdg[0], first_batch)

if __name__ == "__main__":
    unittest.main()
