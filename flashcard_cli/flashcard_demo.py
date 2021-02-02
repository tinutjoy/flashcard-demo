"""Demo for generating flash cards"""
import os
import json
import argparse


from flashcard_model.flashcard_generator import FlashCardGenerator


def flash_card_demo(experiment_directory: str, file_name: str, model_name: str, max_content_length: int,
                    save_flashcards: bool, indent: int = 4) -> None:
    """
    Generates flash card for the text read from the input file.
    :param experiment_directory: local path of input file.
    :param file_name: input file name.
    :param model_name: name of the pretrained model.
    :param max_content_length: Max length of input text.
    :param save_flashcards: whether to save the flashcards or not, defaults to True.
    :param indent: indent for formatting JSOn.
    """
    with open(os.path.join(experiment_directory, file_name), 'r') as input_file:
        input_text = input_file.read().replace('\n', ' ')

    if len(input_text.split(" ")) > max_content_length:
        raise UserWarning(f"\n Maximum number of words exceeded the limit {max_content_length}. "
                          f"\nPlease provide a shorter text")
    flashcard_generator = FlashCardGenerator(model_name=model_name)
    flashcards = flashcard_generator.generate_cards(input_text)
    processed_flashcards = flashcard_generator.postprocess_flashcards(flashcards, indent)
    print(processed_flashcards)
    if save_flashcards:
        with open(os.path.join(experiment_directory, 'flash_cards.json'), 'w', encoding='utf-8') as outfile:
            json.dump(processed_flashcards, outfile, ensure_ascii=False, indent=indent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate flashcards from the given text')
    parser.add_argument('-d',
                        '--experiment_directory',
                        type=str,
                        required=True,
                        help="Path where we store the data (if any) related to this experiment")
    parser.add_argument('-f',
                        '--file_name',
                        type=str,
                        required=True,
                        help="Name of the input text file, and must be kept in the experiment directory")
    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=False,
                        default="valhalla/t5-small-qa-qg-hl",
                        choices=["valhalla/t5-small-qa-qg-hl", "valhalla/t5-base-qa-qg-hl"],
                        help="Name of the pretrained model used for generating flash cards")
    parser.add_argument('--max_len',
                        type=int,
                        required=False,
                        default=500,
                        help="Maximum content length")
    parser.add_argument('--save_cards',
                        type=bool,
                        required=False,
                        default=True,
                        help="save flash cards in the directory")
    args = parser.parse_args()
    flash_card_demo(args.experiment_directory, args.file_name, args.model_name, args.max_len, args.save_cards)
