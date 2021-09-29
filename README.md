# Flash Card Demo

Generates a set of question/answer pairs a.k.a flashcards for a given text.

#### Please install the packages from requirements.txt. The code is tested on python 3.7.

To Run the CLI  demo, please use the following command. 
```sh
usage: python -m flashcard_cli.flashcard_demo  [-h] -d EXPERIMENT_DIRECTORY -f FILE_NAME
                                               [-m {valhalla/t5-small-qa-qg-hl,valhalla/t5-base-qa-qg-hl}]
                                               [--max_len MAX_LEN] [--save_cards SAVE_CARDS]
```
For additional details, please refer to the python script `flashcard_demo.py`.

Start the analytics module server:

```sh
./flashcard_analytics/start_service
```

and see the demo here,  ```http://127.0.0.1:8000/docs```. Here is an example script. 

```sh
./flashcard_analytics/send_request "I live in Melbourne" 
```
