"""Tests for FlashCardGenerator"""
import json
import unittest

from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from flashcard_model.flashcard import FlashCard
from flashcard_model.flashcard_generator import FlashCardGenerator

test_tokenizer_text = ["Duck is a bird"]
extracted_answer = [['<pad> Duck']]

test_generate_question = ['generate question:  <hl> Duck <hl>  is a bird </s>']
expected_generated_question = ['What is a bird?']

expected_flashcards = FlashCard(question='What is a bird?', answer='Duck')
results_dict = expected_flashcards.card
expected_flash_card_indent_3 = json.dumps([results_dict], indent=3)
expected_flash_card_indent_4 = json.dumps([results_dict], indent=4)


class TestFlashCardGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = FlashCardGenerator(model_name="valhalla/t5-small-qa-qg-hl")
        self.generator._initialize_model()

    def test_initialize_model(self):
        self.assertIsInstance(self.generator.model, T5ForConditionalGeneration)
        self.assertIsInstance(self.generator.text_processor.tokenizer, T5TokenizerFast)

    def test_flashcard_generator_class(self):
        generator_2 = FlashCardGenerator(model_name="test")
        generator_3 = FlashCardGenerator(model_name="test")
        self.assertEqual(repr(generator_2), "FlashCardGenerator('test', 'test', 'cpu', '100', '20')")
        self.assertNotEqual(self.generator, generator_2)
        self.assertEqual(generator_2, generator_3)

    def test_load_tokenizer(self):
        model = self.generator.load_model(self.generator.model_name)
        self.assertIsInstance(model, T5ForConditionalGeneration)

    def test_extract_answers(self):
        sentence, answer = self.generator._extract_answers(test_tokenizer_text[0])
        self.assertListEqual(sentence, test_tokenizer_text)
        self.assertListEqual(answer, extracted_answer)

    def test_generate_questions(self):
        question = self.generator._generate_questions(test_generate_question)
        self.assertListEqual(question, expected_generated_question)

    def test_generate(self):
        results = self.generator.generate_cards(test_tokenizer_text[0])
        self.assertEqual(results, [expected_flashcards])

    def test_post_process_cards(self):
        results = self.generator.postprocess_flashcards([expected_flashcards], indent=3)
        self.assertEqual(results, expected_flash_card_indent_3)
        results = self.generator.postprocess_flashcards([expected_flashcards])
        self.assertEqual(results, expected_flash_card_indent_4)
