# HW9

- Mean field inference
- oltzmann machine model

## Problem 1
Mean field inference for binary images The MNIST dataset consists of 60, 000 images of handwritten digits, curated by Yann LeCun, Corinna Cortes, and Chris Burges. You can find it here , together with a collection of statistics on recognition, etc. We will use the first 500 of the training set images.
Obtain the MNIST training set, and binarize the first 500 images by mapping any value below .5 to -1 and any value above to 1. For each image, create a noisy version by randomly flipping 2% of the bits.
Now denoise each image using a Boltzmann machine model and mean field inference. Use theta_{ij}=0.2 for the H_i, H_j terms and theta_{ij}=2 for the H_i, X_j terms. To hand in: Report the fraction of all pixels that are correct in the 500 images.

Assume that theta_{ij} for the H_i, H_j terms takes a constant value c. We will investigate the effect of different values of c on the performance of the denoising algorithm. Think of your algorithm as a device that accepts an image, and for each pixel, predicts 1 or -1. You can evaluate this in the same way we evaluate a binary classifier, because you know the right value of each pixel. A receiver operating curve is a curve plotting the true positive rate against the false positive rate for a predictor, for different values of some useful parameter. We will use c as our parameter. To hand in: Using c=(-1, 0, 0.2, 1, 2) plot a receiver operating curve for your denoising algorithm.

## Submission Instruction
1. Average accuracy on the first 500 images  
2. One set of sample images for each digit -- For each digit you should put up a row of a (sample image, noised version, denoised version via MFI). This means should have a total of 30 images (10 rows, 3 columns).*    
3. Best reconstruction (original, noisy, denoised)*  
4. Worst reconstruction (original, noisy, denoised)*   
5. ROC curve   
6. Code snippet   
7. Any other relevant code.    
