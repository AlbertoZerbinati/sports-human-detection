# Sports Human Detection
Compiuter Vision project for the detection of humans in sports images.

A.Y. 2022-2023
Authors: Alberto Zerbinati, Marco Sedusi, Marco Calì

## Usage
1. Download the raw dataset from https://drive.google.com/file/d/1OEn1nHN4T0PdysuzRUNyqhGtnbWjI-UB/view
2. Extract all images in the `data/images` folder
3. `pip install -r requirements.txt` (create a virtual environment if you desire)
4. Create the final dataset by running `./data/create_crops.sh` (use `chmod` if needed)
5. (optional) Train the model by running `python cnn/train.py --train_model --save_trained_model --evaluate_model --model_path models/people_detection_model.pt`. This will train the model, save it to the `models` folder in a format that can be loaded by `torch.load` and also in C++, and finally evaluate it against the test set. A pretrained model is provided anyway.
6. ... ⚙️ Build & Launch ⚙️ ...
