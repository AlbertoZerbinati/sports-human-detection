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
- (*optional*) If you want to **train you own model** on our dataset:
  - Download the **raw dataset** from <https://drive.google.com/file/d/1OEn1nHN4T0PdysuzRUNyqhGtnbWjI-UB/view>
  - **Extract** all images in the `data/images` folder
  - `pip install -r requirements.txt` (create a virtual environment if you desire)
  - Create the **final dataset** by running `./data/create_crops.sh` (use `chmod` if needed)
  - **Train** the model by running `python cnn/train.py --train_model --save_trained_model --evaluate_model --model_path models/people_detection_model.pt`. A pretrained model is provided anyway
- Download project dependencies:
  - [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
  - [LibTorch](https://pytorch.org/get-started/locally/) — 
  We used this version: [link to download](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip). Extract the folder in the root of the project and adjust the `CMakeLists.txt` file accordingly.
- **Build** and make the project 🧰
  - `mkdir build && cd build`
  - `cmake ..`
  - `make`
- **Run** the project 🚀
  - `./sports_human_detection <image_path> <model_path>`

## Report
Ask for permission [here](https://docs.google.com/document/d/1_8SdJ6yfRL37Bn0gcs749Rhd29lRVZCEUC0VRQvNA3Y/edit#heading=h.5wxqaqinvaq4).
