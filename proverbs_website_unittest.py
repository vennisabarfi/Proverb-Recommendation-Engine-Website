import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import random
import pandas as pd

# Import functions to be tested
from your_script_name import compute_proverbs_similarity

class TestProverbsRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up necessary variables and data before running tests
        cls.tokenizer = MagicMock()
        cls.model = MagicMock()
        cls.loaded_embeddings = MagicMock()
        cls.proverbs = ["proverb1", "proverb2", "proverb3"]
        cls.meanings = ["meaning1", "meaning2", "meaning3"]

    def test_compute_proverbs_similarity(self):
        # Test case when there are similar proverbs
        user_input = "test_input"
        with patch("random.choice", return_value="proverb1") as mock_choice:
            similar_proverbs = compute_proverbs_similarity(user_input, self.tokenizer, self.model, self.loaded_embeddings, self.proverbs, self.meanings)
            self.assertEqual(similar_proverbs, ["proverb1"])

        # Test case when there are no similar proverbs
        user_input = "test_input"
        with patch("random.choice", return_value=None) as mock_choice:
            similar_proverbs = compute_proverbs_similarity(user_input, self.tokenizer, self.model, self.loaded_embeddings, [], [])
            self.assertEqual(similar_proverbs, [])

    @patch("streamlit.text_input", return_value="test_input")
    @patch("streamlit.button", return_value=True)
    @patch("streamlit.write")
    @patch("random.choice", return_value="proverb1")
    def test_streamlit_proverb_recommender(self, mock_choice, mock_write, mock_button, mock_text_input):
        # Simulate the streamlit front-end flow and check if it works correctly
        import your_script_name

        your_script_name.main()

        mock_text_input.assert_called_once_with("Enter a theme/mood here", placeholder="family, love, happiness, wisdom, stress")
        mock_button.assert_called_with("Submit")
        mock_choice.assert_called_with(['proverb1'])
        mock_write.assert_called_with("proverb1")

if __name__ == '__main__':
    unittest.main()
