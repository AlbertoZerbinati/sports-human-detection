# Sports Human Detection
Players and field segmentation in sport images.

## Usage

You have two options for setting up and running the project:

1. **Using Docker** (Reproducible Environment):
  - Build the provided Docker image with `docker build -t sports-human-detection .`
  - Run the container with `docker run --name sports-container -it sports-human-detection`
  - Docker was used during development and is a recommended way to distribute the code. After these steps, you're set to download the dataset and proceed with the build/train/run steps below.
  - 💡 Tip: use the command `docker cp <local_path> sports-container:<container_path>` to copy local files (e.g. dataset images) from your local pc to the docker container.
  
2. **Local Installation** (with OpenCV and LibTorch):
  - Download project dependencies:
    - [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
    - [LibTorch](https://pytorch.org/get-started/locally/) —
    We used this version, for the x86 CPU architecture: [link to download](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip). Extract the folder in the root of the project and adjust the `CMakeLists.txt` file accordingly.

**Common Steps for Both Options:**
- (*optional*) If you want to **train your own model** on our dataset (a pretrained model is provided anyway):
  - Download the **raw dataset** from <https://drive.google.com/file/d/1OEn1nHN4T0PdysuzRUNyqhGtnbWjI-UB/view>
  - **Extract** all images in the `data/images` folder
  - `pip3 install -r requirements.txt` (create a virtual environment if you desire)
  - Create the **final dataset** by running `./data/create_crops.sh` (use `chmod` if needed)
  - **Train** the model by running `python3 cnn/train.py --train_model --save_trained_model --evaluate_model --model_path models/people_detection_model.pt`
- **Build** the project 🧰
  - `mkdir build && cd build`
  - `cmake ..`
  - `make`
- **Run** the project 🚀
  - from within the `build` folder, run `./sports_human_detection <image_path> <model_path> <ground_truth_bboxes_file_path> <ground_truth_segmentation_mask_path>`

Choose the option that best fits your needs to get started.

## Report

Ask for permission [here](https://docs.google.com/document/d/1_8SdJ6yfRL37Bn0gcs749Rhd29lRVZCEUC0VRQvNA3Y/edit#heading=h.5wxqaqinvaq4)

---

Authors: Alberto Zerbinati, Marco Sedusi, Marco Calì

A.Y. 2022-2023 - unipd - Computer Vision project
