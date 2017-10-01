# Train an object classifier on [ImageNet](http://image-net.org/download-images) using multiple gpus in Torch7

This repo shows how to train a object classifier over ImageNet/Cifar10/Cifar100/MNIST using a multi-threaded, multi-gpu approach.

## Features

- Several types of networks like AlexNet, Overfeat, VGG, Googlenet, etc. are available for training;
- Multi-GPU support;
- Data loading/processing using multiple threads;
- Easily apply data augmentation;
- Integration with the `dbcollection` package.

## Requirements

- NVIDIA GPU with compute capability 3.5+ (2GB+ ram)
- [torch7](http://torch.ch/docs/getting-started.html#_)
- [torchnet](https://github.com/torchnet/torchnet)
- [dbcollection](https://github.com/dbcollection/dbcollection)


## Running the code

The main script comes with several options which can be listed by running the script with the flag --help
```bash
th main.lua --help
```

To train a network using the default settings, simply do:
```bash
th main.lua
```

> Note: You must have the ImageNet ILSVRC2012 dataset (or any other dataset) setup before running this script. For more information about how to setup your datasets using `dbcollection` see [here](https://github.com/dbcollection/dbcollection).

By default, the script trains theAlexNet model on 1 GPU with the CUDNN backend and loads data from disk using 4 CPU threads.

To run an alexnet model using two or more GPUs, set `nGPU` to the number of GPUs you want to use (in this example only two are used):
```bash
th main.lua -nGPU 2 -netType alexnet
```

In case you want to specify which gpus do use, do the following:
```bash
CUDA_VISIBLE_DEVICES=0,1 th main.lua -nGPU 2 -netType alexnet
```

> Note: this will select the first two GPUs detected in your system.

To use more threads for data loading/processing, use the `nThreads` flag to specify the number of threads you want to use.

```bash
th main.lua -nThreads 2
```

For a complete list of available options, please see the `opts.lua` file or run `th main.lua --help` in the command line.


## Data loading benchmark comparison

For most datasets, loading the necessary metadata (filenames, labels, etc.) from disk should carry a very small, almost  insignificant overhead compared to loading metadata from memory.

To showcase this, some scripts under `benchmark/` for the ImageNet ILSVRC2012 and Cifar10 datasets are available for benchmarking this. Here it is used the average time for 1000 data fetches with `batchsize=128` and `nThreads=4`.

The `train` scores use more data augmentation preprocessing compared to the `test` scores which uses less data augmentation techniques.


Dataset | train | test
--- | --- | ---
Cifar10 *(disk)* | 0.01509s | 0.00953s
Cifar10 *(ram)* | **0.00772s**  | **0.00557s**
ILSVRC2012 *(disk)* | 0.34635 | **0.35729**
ILSVRC2012 *(ram)* | **0.34553** | 0.36107


> Note: This tests were done using a 6-core Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz, 32GB ram, 2TB SSHD and Ubuntu 14.04. Note that the overhead is very small when using datasets with bigger images like the Imagenet, meaning that the overhead can be hidden by using enough cores or a faster disk.


## Code Description

- `main.lua` (~250 lines) - Script using torchnet's api for training and testing a network over ImageNet.

- `utils.lua` (~125 lines) - Multi-gpu functions for loading/storing/setting a model.

- `transforms.lua` (~500 lines) - Data augmentation functions, mostly derived from [here](https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua) and [here](https://github.com/NVIDIA/DIGITS/pull/777).

- `configs.lua` (~200 lines) - Setup configurations (options, model, logger, etc.)

- `statistics.lua` (~100 lines) - Computes the dataset's mean/std statistics for 10000 samples and stores it to `./cache` dir.

- `model.lua` (~40 lines) - Creates/Loads a model from training/testing.

- `data.lua` (~110 lines) - Contains the methods to featch/load data of the available datasets.

## License

MIT license (see the LICENSE file)

## Disclamer

This code has been inpired on torchnet's [mnist training example](https://github.com/torchnet/torchnet/blob/master/example/mnist.lua), soumith's [multi-gpu ImageNet training code](https://github.com/soumith/imagenet-multiGPU.torch) and @karandwivedi42 [multigpu-torchnet](https://github.com/karandwivedi42/imagenet-multiGPU.torchnet).