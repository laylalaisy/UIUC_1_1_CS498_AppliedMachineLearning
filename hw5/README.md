# HW5 
- k-means 
- KNN

You may use packages for k-means, for nearest neighbors, and for classification

Do problem 9.4 in the 18-Sep version of the textbook.

## Problem 1
Obtain the activities of daily life dataset from the UC Irvine machine learning
website (https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer
data provided by Barbara Bruno, Fulvio Mastrogiovanni and Antonio Sgorbissa).  
- (a) Build a classifier that classifies sequences into one of the 14 activities provided.
To make features, you should vector quantize, then use a histogram
of cluster centers (as described in the subsection; this gives a pretty explicit
set of steps to follow). You will find it helpful to use hierarchical
k-means to vector quantize. You may use whatever multi-class classifier
you wish, though I’d start with R’s decision forest, because it’s easy to
use and effective. You should report (a) the total error rate and (b) the
class confusion matrix of your classifier.
- (b) Now see if you can improve your classifier by (a) modifying the number
of cluster centers in your hierarchical k-means and (b) modifying the size
of the fixed length samples that you use.

 

## Submission Instructions
This homework is very much like an experiment. Here's what we're looking for in the submission report: 
- Page 1: Table listing the experiments carried out with the following columns. Size of the fixed length sample Overlap (0-X%) K- value Classifier Accuracy We expect you to have tried at least 2 values of K and at least 2 different lengths of the windows for quantization. Note: For K-means please also list if you used standard K-means or hierarchical
- Page 2:
Histograms of the mean quantized vector (Histogram of cluster centres like in the book) for each activity with the K value that gives you the highest accuracy. (Please state the K value)
Class confusion matrix from the classifier that you used. Please make sure to label the row/columns of the matrix so that we know which row corresponds to what.
- Page 3: Code snippets (few lines) highlighting the following:
  - 1. segmentation of the vector
  - 2. k-means
  - 3. generating the histogram
  - 4. classification
- Page 4 and beyond: Any relevant code screenshot

We will also be awarding extra points for in-depth experimentation and analysis.

What we're mostly looking for is if how likely are we to be able to replicate the results following the steps you took. So, things you should look for are patterns in the confusion matrix between activities which are similar. Explaining your results in a few lines would thus be very useful. These page restrictions are not _STRICT_ in the sense that you can have an experiments table that flows into two pages if you did a lot of experiments, in such cases please make sure to mark both the pages for that question while uploading.

Really sorry again for the long delay in posting the instructions. If there are any issues that this causes please let us know and we will try to alleviate them.
