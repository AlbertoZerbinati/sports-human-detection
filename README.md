# Sports Human Detection
Compiuter Vision project for the detection of humans in sports images.

A.Y. 2022-2023
Authors: Alberto Zerbinati, Marco Sedusi, Marco Calì

## Usage
1. (optional) If you want to test the code in a reproducible environment, we provide a **Dockerfile**. Build the image with `docker build -t sports-human-detection .` and run it with `docker run -it sports-human-detection`. This was used during development and we think it's a great way to test the code. The following steps assume you are using a 
correctly configured environment in terms of C++, OpenCV, cmake, etc installations.
1. Download the raw **dataset** from https://drive.google.com/file/d/1OEn1nHN4T0PdysuzRUNyqhGtnbWjI-UB/view
4. **Extract** all images in the `data/images` folder
5. `pip install -r requirements.txt` (create a virtual **environment** if you desire)
5. Create the final **dataset** by running `./data/create_crops.sh` (use `chmod` if needed)
6. (optional) **Train** the model by running `python cnn/train.py --train_model --save_trained_model --evaluate_model --model_path models/people_detection_model.pt`. This will train the model, save it to the `models` folder in a format that can be loaded by `torch.load` and also in C++, and finally evaluate it against the test set. A pretrained model is provided anyway.
7. ... ⚙️ **Build** & **Launch** ⚙️ ...
