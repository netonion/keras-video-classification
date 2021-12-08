import unittest
from load_video import load_video

short_video = "./data/DE014_os/de014_os_3s_4-7s.avi"
long_video = "./data/DE130_os/de130_os_17-18s.avi"

class TestLoadingVideo(unittest.TestCase):

    def test_load_short_video(self):
        vid, mask = load_video(short_video)
        self.assertEqual(vid.shape, (120, 224, 224, 3))
        self.assertTrue(vid[0].any())
        self.assertFalse(vid[52:].any())
        self.assertTrue(mask[:52].all())
        self.assertFalse(mask[52:].any())

    def test_load_long_video(self):
        vid, mask = load_video(long_video)
        self.assertEqual(vid.shape, (120, 224, 224, 3))
        self.assertTrue(vid[0].any())
        self.assertTrue(vid[-1].any())
        self.assertTrue(mask.all())

if __name__ == "__main__":
    unittest.main()
