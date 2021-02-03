"""Implement class TextPreprocessor"""
from typing import List
from typing import Dict
from typing import Tuple
from typing import Any
from typing import Union
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

nltk.download('punkt')


class TextPreprocessor:
    """
    Implements various text processing including tokenizing.
    """
    def __init__(self, tokenizer_name: str) -> None:
        """
        :param tokenizer_name: Pretrained model used for tokenizing the text.
        """
        self.tokenizer_name: str = tokenizer_name
        self.tokenizer: Optional[T5TokenizerFast] = None

    @staticmethod
    def load_tokenizer(tokenizer_name: str) -> T5TokenizerFast:
        """
        Downloads pretrained tokenizer.
        :param tokenizer_name: name of the pretrained tokenizer.
        :return: Pretrained tokenizer model.
        """
        return AutoTokenizer.from_pretrained(tokenizer_name)

    def _initialize_tokenizer(self) -> None:
        """
        Initializes the pretrained tokenizer.
        """
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer(self.tokenizer_name)
        assert self.tokenizer is not None, \
            f"Issues with loading pretrained {self.tokenizer_name} model"

    def tokenize(self,
                 sentences: List[str],
                 padding: bool = True,
                 truncation: bool = True,
                 add_special_tokens: bool = True,
                 max_length: int = 512
                 ) -> BatchEncoding:
        """
        Use pretrained tokenizer to tokenize the input text.
        :param sentences: input text.
        :param padding: Whether to pad the sequence after tokenizing.
        :param truncation: Whether to truncate the sequence.
        :param add_special_tokens: Whether to ad special tokens.
        :param max_length: Maximum length of the token sequence.
        :return: tokenized text.
        """
        self._initialize_tokenizer()
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            sentences,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return tokenized_inputs

    @staticmethod
    def prepare_answer_extraction_format(text: str) -> Tuple[List[str], List[str]]:
        """
        Prepares text required by the model to extract answers.
        :param text: input text.
        :return: sentences and the required format.
        """
        sentences = sent_tokenize(text)
        inputs = []
        for sent_idx_outer in range(len(sentences)):
            source_text = "extract answers:"
            for sentence_idx_inner, sentence in enumerate(sentences):
                if sent_idx_outer == sentence_idx_inner:
                    sentence = "<hl> %s <hl>" % sentence
                source_text = "%s %s" % (source_text, sentence)
                source_text = source_text.strip()
            source_text = source_text + " </s>"
            inputs.append(source_text)
        return sentences, inputs

    @staticmethod
    def prepare_question_generation_format(sentences: List[str],
                                           answers: List[List[str]]) -> List[Dict[str, Union[str, Any]]]:
        """
        Prepare text required by the model to generate questions.
        :param sentences: List of sentences.
        :param answers: List of text in the required format.
        :return: Returns the text in the required format.
        """
        inputs = []
        for answer_idx, answer in enumerate(answers):
            if not answer:
                continue
            for answer_text in answer:
                sentence = sentences[answer_idx]
                sentences_copy = sentences[:]
                answer_text = answer_text.strip().replace('<pad>', '').strip()
                ans_start_idx = sentence.index(answer_text)
                sentence = f"{sentence[:ans_start_idx]} <hl> {answer_text} <hl> " \
                           f"{sentence[ans_start_idx + len(answer_text):]}"
                sentences_copy[answer_idx] = sentence
                source_text = " ".join(sentences_copy)
                source_text = f"generate question: {source_text} </s>"
                inputs.append({"answer": answer_text, "source_text": source_text})
        return inputs

    def __repr__(self):
        return f"TextPreprocessor('{self.tokenizer_name}')"

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.tokenizer_name == other.tokenizer_name
        return False
