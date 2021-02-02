"""Tests for FlashCard"""
import unittest

from flashcard_model.flashcard import FlashCard


class TestFlashCard(unittest.TestCase):
    def test_default_values(self):
        flash_card = FlashCard(question='test_ques', answer='test_ans')
        self.assertEqual(flash_card.question, 'test_ques')
        self.assertEqual(flash_card.answer, 'test_ans')
        self.assertEqual(flash_card.card, {'card': 'test_ques', 'answer': 'test_ans'})
