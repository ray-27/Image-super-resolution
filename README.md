# Image-super-resolution
My current ongoing research on image resolution enhancement using deep learning techniques.

## Context

- [[#Data Preparation]]
	- [[#Edge image data]]
- 
---

# Data Preparation

- We have used [Div2k dataset](https://www.kaggle.com/datasets/joe1995/div2k-dataset) it contains 2k images 
- We have resized the image by a scale of 4 and applied Bi-cuberic interpolation

## Edge image data
- Before building the 3 channels super resolution architecture, we planned to build a model that can increase the resolution of the edges of in the image.
- This dataset model can be found in the `edge_dataset.py` file.
- 
