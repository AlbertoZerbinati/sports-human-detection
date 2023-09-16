# Sports Human Detection
A.Y. 2022-2023 - unipd - Computer Vision project

Detection of players and field in sport images.

Authors: Alberto Zerbinati, Marco Sedusi, Marco Calì

## Usage
- (*optional*) If you want to test the code in a reproducible environment, we provide a **Dockerfile**:
  - Build the image with `docker build -t sports-human-detection .`
  - Run it with `docker run -it sports-human-detection`
  - Docker was used during development and we think it's a great way to distribute the code. The following steps assume you are using a 
correctly configured environment in terms of C++, OpenCV, cmake, etc.
- (*optional*) If you want to **train you own model** with our dataset:
  - Download the **raw dataset** from https://drive.google.com/file/d/1OEn1nHN4T0PdysuzRUNyqhGtnbWjI-UB/view
  - **Extract** all images in the `data/images` folder
  - `pip install -r requirements.txt` (create a virtual environment if you desire)
  - Create the **final dataset** by running `./data/create_crops.sh` (use `chmod` if needed)
  - **Train** the model by running `python cnn/train.py --train_model --save_trained_model --evaluate_model --model_path models/people_detection_model.pt`. A pretrained model is provided anyway
- ... ⚙️ **Build** & **Launch** ⚙️ ...
