"""Tests for TextPreprocessor"""
import unittest
from requests.exceptions import HTTPError

from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from preprocessing.text_preprocessor import TextPreprocessor

test_tokenizer_text = ["I saw a duck"]

expected_tokens = {'input_ids': [[27, 1509, 3, 9, 14938, 1, 0, 0, 0, 0]],
                   'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]}

expected_tokens_nopadding = {'input_ids': [[27, 1509, 3, 9, 14938, 1]],
                             'attention_mask': [[1, 1, 1, 1, 1, 1]]}

expected_tokens_no_special_tokens = {'input_ids': [[27, 1509, 3, 9, 14938]],
                                     'attention_mask': [[1, 1, 1, 1, 1]]}

expected_answer_extraction_input = (test_tokenizer_text, ['extract answers: <hl> I saw a duck <hl> </s>'])

expected_answer_extraction_input_hl = [{'answer': test_tokenizer_text[0],
                                        'source_text': 'generate question:  <hl> I saw a duck <hl>  </s>'}]


class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextPreprocessor("valhalla/t5-small-qa-qg-hl")

    def test_initialize_model(self):
        self.assertIsNone(self.processor.tokenizer)
        self.processor._initialize_tokenizer()
        self.assertIsInstance(self.processor.tokenizer, T5TokenizerFast)

    def test_preprocessor_class(self):
        processor_2 = TextPreprocessor("test")
        self.assertEqual(repr(processor_2), "TextPreprocessor('test')")
        self.assertNotEqual(self.processor, processor_2)

    def test_load_tokenizer(self):
        loaded_tokenizer = self.processor.load_tokenizer(self.processor.tokenizer_name)
        self.assertIsNotNone(loaded_tokenizer)
        self.assertIsInstance(loaded_tokenizer, T5TokenizerFast)
        self.assertRaises((HTTPError, OSError, EnvironmentError), self.processor.load_tokenizer, "test")

    def test_tokenize(self):
        actual_tokens = self.processor.tokenize(test_tokenizer_text, max_length=10)
        self.assertListEqual(expected_tokens['input_ids'], actual_tokens['input_ids'].tolist())
        self.assertListEqual(expected_tokens['attention_mask'], actual_tokens['attention_mask'].tolist())

        #no padding
        actual_tokens = self.processor.tokenize(test_tokenizer_text, padding=False, max_length=10)
        self.assertListEqual(expected_tokens_nopadding['input_ids'], actual_tokens['input_ids'].tolist())
        self.assertListEqual(expected_tokens_nopadding['attention_mask'], actual_tokens['attention_mask'].tolist())

        #max_length_variation
        actual_tokens = self.processor.tokenize(test_tokenizer_text,  max_length=5, add_special_tokens=False)
        self.assertListEqual(expected_tokens_no_special_tokens['input_ids'], actual_tokens['input_ids'].tolist())
        self.assertListEqual(expected_tokens_no_special_tokens['attention_mask'],
                             actual_tokens['attention_mask'].tolist())

    def test_prepare_answer_extraction(self):
        result = self.processor.prepare_answer_extraction_format(test_tokenizer_text[0])
        self.assertEqual(result, expected_answer_extraction_input)

    def test_prepare_question_generation_from_answers(self):
        result = self.processor.prepare_question_generation_format(test_tokenizer_text,
                                                                   [expected_answer_extraction_input[0]])
        self.assertListEqual(result, expected_answer_extraction_input_hl)
