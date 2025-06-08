[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/BOO70ufO)

# Setup

1. Clone the repo
2. `cd` into the **root** directory
3. Setup and activate a virtual env
4. `pip install -r requirements.txt`
5. Download the `gesture_dataset_sample` from [rhsslk3.ur.de](https://rhsslk3.ur.de/~sca04209/gesture_dataset_sample.zip)
6. Unpack the downloaded archive into the **root** directory

> 💡 `requirements.txt` includes `jupyter` and `ipykernel` so the virtual environment can also be used as a kernel for the notebook in the first task

> 💡 The folder structure with the downloaded dataset should look like this
```
assignment-05-cnn-realdegrees-1/
├── 01-exploring_hyperparameters/
├── 02-dataset/
├── 03-media_control/
└── gesture_dataset_sample/
    ├── _annotations/
    ├── dislike/
    ├── fist/
    └── .../
```

# Exploring Hyperparameters (number of convolution layers)

1. Complete the setup steps above
2. Select the virtual environment as the kernel for the notebook
3. Run all cells

> 💡 Documentation on this task can be found in the notebook

# Gathering a Dataset 

```sh
cd 02-dataset
python -------------
```

# Gesture-based Media Controls

```sh
cd 03-media_control
python media_control.py
```
