"""API for flashcard demo"""
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.security import HTTPBasic

from flashcard_model.flashcard_generator import FlashCardGenerator


class InputRequest(BaseModel):
    input_text: str


class FlashCardResponse(BaseModel):
    flash_cards: str


app = FastAPI()
security = HTTPBasic()
flashcard_generator = FlashCardGenerator(model_name="valhalla/t5-small-qa-qg-hl")


@app.post("/generate", response_model=FlashCardResponse)
def generate(request: InputRequest) -> BaseModel:
    flashcards = flashcard_generator.generate_cards(text=request.input_text)
    flashcards_processed = flashcard_generator.postprocess_flashcards(flashcards, indent=4)
    return FlashCardResponse(flash_cards=flashcards_processed)
