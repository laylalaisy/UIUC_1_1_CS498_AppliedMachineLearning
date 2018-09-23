Reference：

PCA：

- http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
- https://yoyoyohamapi.gitbooks.io/mit-ml/content/%E7%89%B9%E5%BE%81%E9%99%8D%E7%BB%B4/articles/PCA.html
- https://www.jiqizhixin.com/articles/2017-07-05-2

- https://blog.csdn.net/u012162613/article/details/42177327



http://luthuli.cs.uiuc.edu/~daf/courses/AML-18-Fall/AML-F18-HW-3.html

Problem 1
You may use any programming language that amuses you for this homework. You may use a PCA package if you so choose but remember you need to understand what comes out of the package to get the homework right!

The goal of this homework is to use PCA to smooth the noise in the provided data. At https://www.kaggle.com/t/e9337b95218e48a1be69a69e3826688a , you will find a five noisy versions of the Iris dataset, and a noiseless version.

For each of the 5 noisy data sets, you should compute the principle components in two ways. In the first, you will use the mean and covariance matrix of the noiseless dataset. In the second, you will use the mean and covariance of the respective noisy datasets. Based on these components, you should compute the mean squared error between the noiseless version of the dataset and each of a PCA representation using 0 (i.e. every data item is represented by the mean), 1, 2, 3, and 4 principal components.

You should produce:

a one-page PDF showing your numbers filled in a table set out as below, where "N" columns represents the components calculated via the noiseless dataset and the "c" columns the noisy datasets.:
upload to Kaggle your reconstruction of the dataset of version II, expanded onto 2 principal components, where mean and principal components are computed from the dataset of version II
Number of PCs->	0N	1N	2N	3N	4N	0c	1c	2c	3c	4c
Dataset I										
Dataset II										
Dataset III										
Dataset IV										
Dataset V										
Submission details
Submit to Kaggle The reconstruction of the second noisy version as a CSV file in the same format as the datasets. The CSV file should be named "yournetid-recon.csv", and I will frown on humorists who do not replace the "yournetid" with their netid. Each row is your reconstructed version of the data item. The first line should be a header reading:
"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"

Each following line is your reconstruction of a data item, in order (so first data item first, etc).

Submit to Kaggle A CSV file containing your numbers for the table. The CSV file should be named "yournetid-numbers.csv", and I will frown on humorists who do not replace the "yournetid" with their netid. The first line should read
"0N, 1N, 2N, 3N, 4N, 0c, 1c, 2c, 3c, 4c"

The following lines should be the rows of the table, in order, and contain only numbers. You should provide your numbers to at least three digits.

Submit to Gradescope: A two-page PDF file. The first page should contain your 50 MSE numbers filled in a table (see the course page for what it should look like), and the second page should be a one-page screenshot of your code.