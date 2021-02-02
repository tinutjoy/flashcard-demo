"""Implements FlashCard dataclass"""
from dataclasses import dataclass


@dataclass(init=True, repr=True, eq=True)
class FlashCard:
    question: str
    answer: str

    def __post_init__(self) -> None:
        self.card = {'card': self.question, 'answer': self.answer}
