# HW8
- Image Segmentaion using EM

## Pre-problem
Do exercise 10.1 in the 23 October version of the book. Also, read the mixture of normal EM notes carefully, and also look at the movies

## Problem 1
Image segmentation using EM You can segment an image using a clustering method - each segment is the cluster center to which a pixel belongs. In this exercise, you will represent an image pixel by its r, g, and b values (so use color images!). Use the EM algorithm applied to the mixture of normal distribution model lectured in class to cluster image pixels, then segment the image by mapping each pixel to the cluster center with the highest value of the posterior probability for that pixel. You must implement the EM algorithm yourself (rather than using a package). Test images are here, and you should display results for all three of them. Till then, use any color image you care to.
Segment each of the test images to 10, 20, and 50 segments. You should display these segmented images as images, where each pixel's color is replaced with the mean color of the closest segment
We will identify one special test image. You should segment this to 20 segments using five different start points, and display the result for each case. Is there much variation in the result? The test image is the painting of trees
For boasting rights (no credit, sorry, and don't submit), identify the artist and explain why the trees are that shape
