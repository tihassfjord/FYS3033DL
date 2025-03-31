# PDF Copy in Markdown

## Page 1

Home Exam

Home Exam in FYS-3033 - Deep Learning

Hand-out:

Monday March 31, 2025, 09:00

Hand-in:

Monday April 28, 2025, 14:00

The Home Exam contains 8 pages including this cover page

Contact person: Kristoﬀer Wickstrøm

Email:

kristoﬀer.k.wickstrom@uit.no

PO Box 6050 Langnes, NO-9037 Tromsø/ +47 77 64 40 00 / postmottak@uit.no / uit.no

## Page 2

Before You Start

Portfolio instructions

Your code should be submitted together with your report (see instructions below). Further, please
include a discussion of the results obtained and a discussion of the implementation in your
report, which show that you understand what you are doing.

The code should be commented in such a way that any person with programming knowledge should be
able to understand how the program works. Like your report, the code must be your own individual work.

You are permitted to use deep learning frameworks such as Pytorch and Tensorﬂow. As there is a lot of
code available online, please make sure that your report and code clearly show that you understand what
you are doing.

Hand-in format

Please submit your report (in pdf format) to WISEﬂow and attach one single .zip ﬁle that contains two
folders, one called doc that contains your report, and another one called src containing the code. The
ﬁle name of the .zip ﬁle should follow the format homeexam_candidateXX.zip (replace XX with
your candidate number obtained from WISEﬂow) for anonymity.

Please include your candidate number and the course name on the frontpage of your report.

Follow the hand-in instruction in Wiseﬂow. Upload the pdf as the main ﬁle and then attach your zip as
an attachment. Note, the reports will be processed by a plagiarism checker and the pdf ﬁle size must
not exceed 15MB.

## Page 3

FYS-3033 - Mandatory Assignment

Problem 1

In this problem, you will derive some of the theoretical results that are found in unsupervised deep learning.

(1a) The goal of generative models is to sample from the underlying distribution, pdata, of data x. Generative
Adversarial Networks (GANs) achieve this goal by constructing a generator G, which tries to fool a
discriminator D. The generator transforms a sample z from a prior noise distribution, pz, into a new
distribution pg. Given the loss that the discriminator is tasked to maximize

V (G, D) =

Z

x

pdata(x) log(D(x))dx +

Z

z

pz(z) log(1 − D(g(z)))dz,

(1)

and assuming that the optimal solution has been reached (pg = pdata). Show that the optimal solution for
the discriminator given a ﬁxed generator g is

D(x) =

1
2 .

(2)

Finally, illustrate that this optimum corresponds to a value of − log(4).

(1b) The reparameterization trick is a key component of Variational Autoencoders (VAEs). What is its purpose
in VAEs? For VAEs the latent variable is sampled from a normal distribution with parameters µz|x and
σz|x, where the parameters are learned by the encoder of the VAE. For the reparameterization trick, we
perform our sampling operation as

z = µz|x + σz|x(cid:15) ,

(3)

where (cid:15) ∼ N (0, 1). Show that z follows a normal distribution with parameters µz|x and σz|x.

(1c) Show that the KL divergence between two univariate Bernoulli distributions p(x) = Bern(p1) and

q(x) = Bern(p2) can be expressed as

KL(p, q) = log

(1 − p1)
(1 − p2)

+ p1 log p1 (1 − p2)
p2 (1 − p1) .

(4)

Discuss how this result can be used in VAEs.

3

## Page 4

FYS-3033 - Mandatory Assignment

Problem 2

You are working for the company Planes, Ships, and Trucks Inc., and have been tasked with developing an
image classiﬁer to automatically classify images of planes, ships and trucks. Your co-worker Elisabeth Wetzer
has worked day and night to gather and label images that your company is interested in, and has provided
you with a dataset (Canvas/Files/homeexam/problem2.zip (training_data.npz, evaluation_data.npz))
with the following content:

• 1500 labeled color images of size 96x96 for training.

• 3 classes; plane, ship, and truck.

• 2400 labeled color images of size 96x96 for validation.

Your job is to develop an algorithm that can successfully tackle the classiﬁcation task and undoubtedly
quadruple the revenue stream of Planes, Ships, and Trucks Inc.

Note: The data is stored in Numpy arrays (.npz extension), and can be loaded like shown below. Remember
to replace the path your local directory.

import numpy a s np
t r a i n i n g _ d a t a = np . l o a d ( ’ your / path / t r a i n i n g _ d a t a . npz ’ )
t r a i n i n g _ i m a g e s = t r a i n i n g _ d a t a [ ’ a ’ ]
t r a i n i n g _ l a b e l s = t r a i n i n g _ d a t a [ ’ b ’ ]

(2a) Based on the task and characteristics of the dataset, you decide that a VGG-11 with Batch Normalization
after the convolutional layers should be a good start. Explain why. Note: Do not use pre-trained
weights/ﬁnetuning in this exercise, nor the premade model from e.g. Torchvision. For details about the
VGG-11 architecture, see column A of Table 1 in Simonyan and Zisserman (1).

(2b) Explain Dropout (2).

(2c) Include Dropout in the network from problem (2a) with a dropout rate of 0.5 after each of the two layers

that contain 4096 ﬁlters, discuss results.

4

## Page 5

FYS-3033 - Mandatory Assignment

Your co-worker Harald Lykke Joakimsen is a hobby photographer and wants to help out with increasing
the size of the database to improve the evaluation of the algorithm. Therefore, he has gone out and
gathered new images (Canvas/Files/homeexam/problem2.zip (new_evaluation_data_with_labels.npz,
new_evaluation_data_without_labels.npz’)). However, in his eagerness to help, he also took images of
birds, which is not interesting for your company (Planes, Ships, and Trucks Inc.). He managed to label some
of the images, but a large portion of the dataset remains unlabeled. The content of new dataset is:

• 1600 labeled color images of size 96x96, containing both the known classes and a new unwanted class.

• 3 known classes; plane, ship, and truck, and one unwanted class; bird.

• 1600 unlabeled color images of size 96x96 containing both the known classes and the unwanted class.
To use the new dataset for evaluation in the future, you must remove as many images of new the
class as possible, while keeping as much as possible of the known classes. You get an idea that since
your trained model has never seen this new class, it should be more uncertain about those samples.
Therefore, you will investigate if uncertainty modeling can be used to ﬁlter out the unwanted images in
the new dataset.

Note: The new data is stored as before, but the unlabeled dataset only has images and no labels.

import numpy a s np
new_evaluation_data = np . l o a d (
’ your / path / new_evaluation_data_with_labels . npz ’ )

new_evaluation_images = new_evaluation_data [ ’ a ’ ] ,
n e w _ e v a l u a t i o n _ l a b e l s = new_evaluation_data [ ’ b ’ ]

u n l a b e l e d _ e v a l u a t i o n _ d a t a = np . l o a d (
’ your / path / new_evaluation_data_without_labels . npz ’ )

u n l a b e l e d _ e v a l u a t i o n _ i m a g e s = u n l a b e l e d _ e v a l u a t i o n _ d a t a [ ’ a ’ ] # no ’ b ’ .

(2d) Leverage the Dropout-mechanism in your trained network to create uncertainty estimates for each of the
labeled images in the new dataset. You will use the the concept of entropy from information theory to
indicate uncertainty as follows:

• For a single sample, do 10 forward passes with Dropout applied and calculate the mean softmax

output across the 10 predictions.

• Calculate the entropy of the mean softmax output ˆy as:

H(ˆy) =

NcX

i=1

ˆyi log ˆyi,

where Nc is the number of classes in your trained model.

Explain why the entropy of the softmax output can be interpreted as a notion of uncertainty. Make a
histogram where you plot the entropy of the known classes as one distribution and the entropy of the
new class as a second distribution. Calculate the mean of each distribution and mark it in the histogram.
What do you observe?

Disclaimer: Uncertainty modeling is a challenging topic in deep learning. It is unlikely that you will be

5

## Page 6

FYS-3033 - Mandatory Assignment

able to ﬁnd all images of the new class, and some images from the known classes will also most likely be
removed.

(2e) Using the results from problem (2d), ﬁnd a threshold where you detect as many images with the new
unwanted class but keep as many as possible of the images with the desired classes. Report the number of
unwanted images you detect and the number of wanted images you remove in the process.

(2f) Bonus problem: Pass the unlabeled part of the dataset through your trained model and predict known
and unknown classes based on the threshold from problem (2e). Store the predictions as a binary vector
in a Numpy array and upload it is part of the exam submission. After the exam, the predictions will be
evaluated and the one who detects the most samples of the new class while keeping as much as possible
of the known classes will receive a lot of glory and a little prize. The Numpy array can be saved using
the numpy.save function, and please use the .npz extension. Also remember not to shuﬄe the data when
doing your predictions.

Problem 3

In this problem, you will train an image classiﬁer on the provided traﬃc sign dataset and explore several
approaches to explain the models predictions and detect backdoors. The data can be found on Canvas in
Canvas/Files/homeexam/problem3.zip.

(3a) Explain one approach that can be used to interpret/explain a speciﬁc prediction of a deep learning model

and one that can be used to interpret/explain the model as a whole.

(3b) Train a ResNet-18 to classify the data. In this problem, you are allowed to use existing frameworks like
e.g. Torchvision, but without pretrained weights. Describe the network and your implementation and
report the obtained accuracy for the training and validation data. For this, use the data contained in the
"train" folder and split it into a 80%/20% train/validation split. Split the data per class so that you keep
the same class distribution.

(3c) Both quantitatively and qualitatively inspect the results on the provided test images (in the "test" folder)
and compare these results to what you obtained on the validation set. Discuss what you observe.

(3d) For the class where the model fails on the test-set, produce Class Saliency Maps for the training images
as described in (3) (Section 3.1) to discover what the model bases its predictions on. Provide the saliency
maps of 10 training images and comment on the observed results.

(3e) For the same training images, perform also an Occlusion analysis as described in (4) (Section 4.2) where
you occlude a 10x10 region and monitor the classiﬁer output as you occlude diﬀerent parts of the image.
Provide illustrations of the same examples as in 2c) that mimic Figure 7 (e) in (4) and comment on the
observed results.

(3f) Based on your observations, propose an approach to modify the training data to remove the backdoor in

the training dataset.

6

## Page 7

FYS-3033 - Mandatory Assignment

(3g) Retrain your classiﬁer on the modiﬁed data and report the new results for the test images. Discuss the

results.

(3h) Provide the saliency maps and the occlusion analysis maps for the new model for the same images as in

2d) and 2e). Discuss the results.

7

## Page 8

FYS-3033 - Mandatory Assignment

References

[1] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image

recognition. In International Conference on Learning Representations, 2015.

[2] Nitish Srivastava, Geoﬀrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
Dropout: a simple way to prevent neural networks from overﬁtting. The Journal of Machine Learning
Research, 15(1):1929–1958, 2014.

[3] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks:
Visualising image classiﬁcation models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.

[4] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In Computer
Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings,
Part I 13, pages 818–833. Springer, 2014.

8

