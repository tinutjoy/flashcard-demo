"""Implement class FlashCardGenerator using T5 model. For more details on the model,
please refer to https://huggingface.co/valhalla/t5-small-qa-qg-hl and https://arxiv.org/abs/1910.10683"""
import json
from itertools import chain

from typing import List
from typing import Optional
from typing import Tuple

from transformers import AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from flashcard_model.flashcard import FlashCard
from preprocessing.text_preprocessor import TextPreprocessor


class FlashCardGenerator:
    """
    Generates Flashcards using a pretrained language model.
    """

    def __init__(self, model_name: str,
                 tokenizer_name: Optional[str] = None,
                 device: Optional[str] = None,
                 max_question_length: int = 100,
                 max_answer_length: int = 20) -> None:
        """
        :param model_name: name of the pretrained model.
        :param tokenizer_name: A text processor class object that processes the inout text.
        :param device: whether to sue cpu or gpu for model inference. Uses cpu by default.
        :param max_question_length: Maximum length of generated question sequence.
        :param max_answer_length: Maximum length of generated answer sequence.
        """
        self.model_name = model_name
        self.model: Optional[T5ForConditionalGeneration] = None

        if tokenizer_name is None:
            self.tokenizer_name = model_name
        else:
            self.tokenizer_name = tokenizer_name
        self.text_processor = None

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

    @staticmethod
    def load_model(model_name: str) -> T5ForConditionalGeneration:
        """
        Downloads pretrained tokenizer.
        :param model_name: name of the pretrained model.
        :return: Pretrained model.
        """
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _initialize_model(self) -> None:
        """
        Initializes the pretrained model and text processor class
        with the tokenizer.
        """
        if self.model is None:
            self.model = self.load_model(self.model_name)
            assert self.model is not None, f"Issues with loading pretrained {self.model_name} model"
            self.model.to(self.device)
        if self.text_processor is None:
            self.text_processor = TextPreprocessor(self.tokenizer_name)
            self.text_processor._initialize_tokenizer()

    def generate_cards(self, text: str) -> Optional[List[FlashCard]]:
        """
        Generate questions and answers for the flash card.
        :param text: input text.
        :return: A list of flashcards for the input text.
        """
        text = " ".join(text.split())
        # Initialize the model
        self._initialize_model()
        # Extract possible answers.
        sentences, answers = self._extract_answers(text)
        # Do not proceed if the extracted answer is empty
        if not list(chain(*answers)):
            return None
        # Generate possible questions given the text and answers.
        question_gen_format = self.text_processor.prepare_question_generation_format(sentences, answers)
        question_generation_input = [example['source_text'] for example in question_gen_format]
        questions = self._generate_questions(question_generation_input)

        # Create flash cards using the questions and answers
        flash_cards = [FlashCard(question=question, answer=example['answer'])
                       for example, question in zip(question_gen_format, questions)]
        return flash_cards

    @staticmethod
    def postprocess_flashcards(flash_cards: List[FlashCard], indent: int = 4) -> str:
        """
        Postprocess and jsonify the flash_cards.
        :param flash_cards: List of flash cards.
        :param indent: indent for JSON format.
        :return: JSON formatted string.
        """
        processed_flash_cards = [cards.card for cards in flash_cards]
        return json.dumps(processed_flash_cards, indent=indent)

    def _generate_questions(self, inputs: List[str],
                            padding: bool = True,
                            truncation: bool = True,
                            skip_special_tokens: bool = True) -> List[str]:
        """
        Generates questions from a given text.
        :param inputs: input text in the format needed by the pretrained model.
        :type padding: Pad the output while tokenizing.
        :type skip_special_tokens: Truncate the output while tokenizing.
        :type skip_special_tokens: Skip special tokens while tokenizing.
        :return: Generated questions.
        """
        # Tokenize input
        inputs = self.text_processor.tokenize(inputs, padding=padding, truncation=truncation)
        # Generate questions (encoded).
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            max_length=self.max_question_length,
            num_beams=4,
        )

        # Decode the model output.
        questions = [self.text_processor.tokenizer.decode(ids,
                                                          skip_special_tokens=skip_special_tokens) for ids in outs]
        return questions

    def _extract_answers(self, text: str,
                         padding: bool = True,
                         truncation: bool = True,
                         skip_special_tokens: bool = False) -> Tuple[List[str], List[List[str]]]:
        """
        Generates answers from a given text.
        :param text: input text.
        :type padding: Pad the output while tokenizing.
        :type skip_special_tokens: Truncate the output while tokenizing.
        :type skip_special_tokens: Skip special tokens while tokenizing.
        :return: extracted sentences and corresponding answers.
        """
        # Prepare the format required by the model for generating answers.
        sentences, model_inputs = self.text_processor.prepare_answer_extraction_format(text)
        tokenized_model_inputs = self.text_processor.tokenize(model_inputs,
                                                              padding=padding,
                                                              truncation=truncation)

        # Generate answers (encoded) using pretrained model.
        outs = self.model.generate(
            input_ids=tokenized_model_inputs['input_ids'].to(self.device),
            attention_mask=tokenized_model_inputs['attention_mask'].to(self.device),
            max_length=self.max_answer_length,
        )

        # Decode the generated answers.
        decoded_results = [self.text_processor.tokenizer.decode(ids,
                                                                skip_special_tokens=skip_special_tokens)
                           for ids in outs]
        answers = [item.split('<sep>')[:-1] for item in decoded_results]
        return sentences, answers

    def __repr__(self):
        return f"FlashCardGenerator('{self.model_name}', '{self.tokenizer_name}', '{self.device}', " \
               f"'{self.max_question_length}', '{self.max_answer_length}')"

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.model_name == other.model_name \
               and self.tokenizer_name == other.tokenizer_name \
               and self.device == other.device \
               and self.max_question_length == other.max_question_length \
               and self.max_answer_length == other.max_answer_length
        return False
