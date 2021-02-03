# Flash Card Demo

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

Use the following script as an example to send a request to the server. It accepts the text and returns the 
response as a json. Additionally, please try out the demo here ```http://127.0.0.1:8000/docs```.

```sh
./flashcard_analytics/send_request "I live in Melbourne" 
```
Note: I have borrowed some code from the repo mentioned here (https://huggingface.co/valhalla/t5-base-qa-qg-hl) which has posted the pretrained model. However, I have rewritten the code to make it modular, and readable. I have also added unittests for all the modules. Regarding API, I have used FastAPI (https://github.com/tiangolo/fastapi) to create an endpoint for the model.
Unit tests are included in `tests` folder under each module.
