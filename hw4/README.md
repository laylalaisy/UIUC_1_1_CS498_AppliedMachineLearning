<meta HTTP-EQUIV="REFRESH" content="0; url=http://www.cs.toronto.edu/~kriz/cifar.html">



# HW4
- PCA +PCoA 

## Pre-problem
This component is not for grading, and you should not submit. But before you start the problem, you should check that you can do 6.3 and 7.3 in the 18-Sep version of the book. Go through each problem, and if you have trouble, ask in office hours. This is to help you get ready conceptually for the programming problem.

## Problem 1
You may use any programming language that amuses you for this homework.

Do problem 7.7 (a) and (b) in the 18-Sep version of the textbook.

CIFAR-10 is a dataset of 32x32 images in 10 categories, collected by Alex
Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is often used to evaluate
machine learning algorithms. You can download this dataset from https://
www.cs.toronto.edu/∼kriz/cifar.html.  
- (a) For each category, compute the mean image and the first 20 principal
components. Plot the error resulting from representing the images of each
category using the first 20 principal components against the category.
- (b) Compute the distances between mean images for each pair of classes. Use
principal coordinate analysis to make a 2D map of the means of each
categories. For
