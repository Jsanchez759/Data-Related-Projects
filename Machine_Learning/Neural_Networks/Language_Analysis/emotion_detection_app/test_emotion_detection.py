from EmotionDetection.emotion_detection import emotion_detector
import unittest

class TestEmotionDetection(unittest.TestCase):
    def test_emotion_detector(self):
        result_1 = emotion_detector('I am so happy')
        self.assertEqual(result_1['emotion'], 'positive')

unittest.main()