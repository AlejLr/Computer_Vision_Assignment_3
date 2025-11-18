# Computer Vision Assignment 3

## Video Gesture Recognition

This is the repo for the third and last assignment of the course computer vision, where we need to design from 0 and train a model to recognise hand gestures from a video.

## Environment

To create the conda environment on the LIACS lab Ubuntu machines:

~~~bash
conda env create -f environment.yml
conda activate jester-env
~~~

Then verify the GPU:

~~~bash
python -c "import torch; print(torch.cuda.is_available())"
~~~
