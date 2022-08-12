# A Bayesian Based Deep Unrolling Algorithm for Single-Photon Lidar Systems

Official code for the paper: [1] https://ieeexplore.ieee.org/document/9763322

The proposed method (BU3D) unrolls an existing Bayesian method [2] into a new neural network to reconstruct depths from single-photon Lidar data. In this document, $ROOT means the root of the repository. The repository conatins three main folders: `run` for testing, `train` for training and `unroll` for the PyTorch model.

TODO: Detect available GPU size and automatically change to CPU mode.

## A simple example

You can open the CoLab notebook https://colab.research.google.com/github/jakeoung/BayesianUnrolling/blob/master/BU3D_simple_example.ipynb

## Required python packages

- pytorch (>= 1.9.0) (including torchvision)
- matplotlib, scipy, numpy
- mat73 (install by typing `pip install mat73`)

## Testing

We provide the pretrained model in `run/model_baseline.pth`. Our method works on both CPU and GPU and the provided code automatically detects if there is an available GPU. (For large histogram data, it might require the 14GB of GPU memory. If you don't have such GPU, you can still run on CPU, by setting the environment `export CUDA_VISIBLE_DEVICES=""`.)

### Synthetic data (Art and Reindeer scene in the middlebury dataset [3])

We provide a synthetic data of Art scene in `run/Art_4.0_1.0` (with PPP=4 and SBR=1). The additional testing data for Art and Reindeer scene can be downloaded [here](https://drive.google.com/file/d/1HtJxjWHd-53-Z6qDqmXaHbycP9QlFG6z/view?usp=sharing) with different levels of PPP and SBR.

To test on such data, open the command and type:
```
cd run
python test_middlebury1024.py
```
In this python file, we shift the system impulse response function (`run/irf/irf_middlebury1024.mat`) and use FFT to generate initial multiscale depths. These multiscale depths are used as an input of the proposed network.


### Real data

To test the real data provided by [4] Lindell et al., download the data [here](https://www.computationalimaging.org/publications/single-photon-3d-imaging-with-deep-sensor-fusion/). You need to put mat data files on `run` folder such as `checkerboard.mat`. (We already included an approximated impulse response function in `run/irf/irf_real.mat`)

Go to `run` folder and execute `python test_real.py`. You need to change the scene name in the main function. The outputs will be saved in `$ROOT/result/` folder.

### Your own data

I recommend you to check the ipython notebook `run/simple_example.ipynb`. To test your own data, the basic procedure is:

1. Prepare a histogram data of shape [height x width x timebins]
2. Prepare an impulse response function and shift it to prepare for using FFT, using the function `shift_h`
3. Generate multiscale depths, using the function `gen_initial_multiscale`
4. Load the pretrained model and run the network

To get started, you can copy the file `run_middlebury1024.py` and modify it, to input your ToF and IRF.

## Training

- Install the [Julia language](https://julialang.org) >= v1.60 and install all the required packages by running [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) and, typing `]`
```
add Random FileIO Plots Images ImageFiltering DrWatson Distributions HDF5 StatsBase Statistics FFTW MAT Flux FourierTools
```

- Download [middlebury data](https://vision.middlebury.edu/stereo/data/) for 2005 and 2006 datasets (HalfSize) and save them, so that the file structure should be like: (You can also refer to `train/download_data.sh`)
```
$ROOT/data/middlebury/raw/Aloe/disp1.png
$ROOT/data/middlebury/raw/Aloe/view1.png
$ROOT/data/middlebury/raw/Books/disp1.png
$ROOT/data/middlebury/raw/Books/view1.png
...
```

- Download [MPI Sintel stereo data](http://sintel.is.tue.mpg.de/stereo) and save them, so that the file structure should be like:
```
$ROOT/data/mpi/training/clean_left/alley_1/frame_0001.png
$ROOT/data/mpi/training/disparities/alley_1/frame_0001.png
$ROOT/data/mpi/training/clean_left/alley_2/frame_0001.png
$ROOT/data/mpi/training/disparities/alley_2/frame_0001.png
...
```

- From the root folder, run
```
cd train
julia create_data_middlebury.jl 1.0 1.0   # it will make training data from middlebury data with SBR=1.0, PPP=1.0
julia create_data_middlebury.jl 1.0 64.0  # with SBR=1.0, PPP=64.0
julia create_data_middlebury.jl 64.0 1.0
julia create_data_middlebury.jl 64.0 64.0
julia create_data_mpi.jl 1.0 1.0
julia create_data_mpi.jl 1.0 64.0
julia create_data_mpi.jl 64.0 1.0
julia create_data_mpi.jl 64.0 64.0
```
If everything is alright, you can see the trainnig data in `$ROOT/data` folder.

- Install the some required python package by typing
```
pip install h5py, tensorboard
```
- Go to `$ROOT/run` folder and run `python train.py`. The training would take around 9 hours on a linux server with the NVIDIA RTX 3090 GPU and the resulting model file (.pth) can be found in `result/mpi` folder.
- You can load such pth file to test your data. For example, replace `run/model_baseline.pth` with your pth file and run `test_middlebury1024.py` there. Note that if you change some settings when training, you need to specify the changes when you load the model: e.g. `model = Model(nscale=12, nblock=3)`.

## Reference

- [1] J. Koo, A. Halimi, and S. McLaughlin, A Bayesian Based Deep Unrolling Algorithm for Single-Photon Lidar Systems IEEE J. Sel. Top. Signal Process.
, 2022.
- [2] A. Halimi, A. Maccarone, R. A. Lamb, G. S. Buller, and S. McLaughlin, Robust and Guided Bayesian Reconstruction of Single-Photon 3D Lidar Data: Application to Multispectral and Underwater Imaging’, IEEE Trans. Comput. Imaging, vol. 7, pp. 961–974, 2021.
- [3] H. Hirschmuller and D. Scharstein, Evaluation of cost functions for stereo matching, in IEEE CVPR, 2007.
- [4] D. B. Lindell, M. O’Toole, and G. Wetzstein, Single-photon 3D imaging with deep sensor fusion’, ACM Trans. Graph., vol. 37, no. 4, 2018

## Acknowledgement

This work was supported by the UK Royal Academy of Engineering under the Research Fellowship Scheme (RF/201718/17128) and EPSRC Grants EP/T00097X/1,EP/S000631/1,EP/S026428/1.
