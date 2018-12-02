# HW7

- Linear Model: Poisson Model + Lasso
- Cross-Validated
- Regression

You may use any programming language that amuses you for this homework, BUT much experience shows that the python lasso code leads to problems (the versions used in the past have not been smart about the regularization path). This is really straightforward if you use R. You'll need to use glmnet (which will do the heavy lifting for you; read the manual). If you plot the result of an cv.glmnet you'll get the plots I showed in class.

## Pre-problem
There is no pre-problem. Each is an exercise in using a powerful regression tool.

# Problem 1
Do problem 12.3 in the 16 Oct version of the textbook.
UC Irvine hosts a dataset of blog posts at https://archive.ics.uci.edu/ml/datasets/
BlogFeedback. There are 280 independent features which measure various
properties of the blog post. The dependent variable is the number of comments
that the blog post received in the 24 hours after a base time. The zip
file that you download will have training data in blogData train.csv, and
test data in a variety of files named blogData test-*.csv.
- (a) Predict the dependent variable using all features, a generalized linear
model (I’d use a Poisson model, because these are count variables), and
the Lasso. For this exercise, you really should use glmnet in R. Produce
a plot of the cross-validated deviance of the model against the regularization
variable (cv.glmnet and plot will do this for you). Use only the
data in blogData train.csv.
- (b) Your cross-validated plot of deviance likely doesn’t mean all that much to
you, because the deviance of a Poisson model takes a bit of getting used
to. Choose a value of the regularization constant that yields a strong
model, at least by the deviance criterion. Now produce a scatter plot
of true values vs predicted values for data in blogData train.csv. How
well does this regression work? keep in mind that you are looking at
predictions on the training set.
- (c) Choose a value of the regularization constant that yields a strong model,
at least by the deviance criterion. Now produce a scatter plot of true
values vs predicted values for data in blogData test-*.csv. How well
does this regression work?
- (d) Why is this regression difficult?

## Problem 2
Do problem 12.4 in the 16 Oct version of the textbook.
find a dataset giving the expression of 2000 genes in tumor and normal colon
tissues. Build a logistic regression of the label (normal vs tumor) against the
expression levels for those genes. There are a total of 62 tissue samples, so
this is a wide regression. For this exercise, you really should use glmnet in R.
Produce a plot of the classification error of the model against the regularization
variable (cv.glmnet – look at the type.measure argument – and plot will
do this for you). Compare the prediction of this model with the baseline of
predicting the most common class.

Problem 3
Do problem 12.5 in the 16 Oct version of the textbook.
The Jackson lab publishes numerous datasets to do with genetics and phenotypes
of mice. At https://phenome.jax.org/projects/Crusio1, you can find a
dataset giving the strain of a mouse, its gender, and various observations (click
on the “Downloads” button). These observations are of body properties like
mass, behavior, and various properties of the mouse’s brain.
- (a) We will predict the gender of a mouse from the body properties and the
behavior. The variables you want are columns 4 through 41 of the dataset
(or bw to visit time d3 d5; you shouldn’t use the id of the mouse). Read
the description; I’ve omitted the later behavioral measurements because
there are many N/A’s. Drop rows with N/A’s (there are relatively few).
How accurately can you predict gender using these measurements, using
a logistic regression and the lasso? For this exercise, you really should
use glmnet in R. Produce a plot of the classification error of the model
against the regularization variable (cv.glmnet – look at the type.measure
argument – and plot will do this for you). Compare the prediction of this
model with the baseline of predicting the most common gender for all
mice.
- (b) We will predict the strain of a mouse from the body properties and the
behavior. The variables you want are columns 4 through 41 of the dataset
(or bw to visit time d3 d5; you shouldn’t use the id of the mouse). Read
the description; I’ve omitted the later behavioral measurements because
there are many N/A’s. Drop rows with N/A’s (there are relatively few).
This exercise is considerably more elaborate than the previous, because
multinomial logistic regression does not like classes with few examples.
You should drop strains with fewer than 10 rows. How accurately can
you predict strain using these measurements, using multinomial logistic
regression and the lasso? For this exercise, you really should use glmnet
in R. Produce a plot of the classification error of the model against the
regularization variable (cv.glmnet – look at the type.measure argument
– and plot will do this for you). Compare the prediction of this model
with the baseline of predicting a strain at random.

 
