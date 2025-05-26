# CatVDog
CNN for the Cats Vs Dogs database from Kaggle

Dogs vs. Cats dataset from the kaggle competition https://www.kaggle.com/competitions/dogs-vs-cats.
Experimented with 2 models, a baseline CNN and the VGG-16 model used to actually win the competition. (https://doi.org/10.48550/arXiv.1409.1556)



## Building the model yourself
1. Download the data set from Kaggle
2. Run FProcess.py to process the photos into subdirectories.
3. Run FVGGModel.py to train the model (may take some time (took me 2 hours)).
4. Rename any photo containing a dog or cat to test.jpg
5. Run Predict.py to find out which animal is in the photo!

I have provided FModelCatdDog.keras so you can skip straight to step 4.

The VGG-16 Model achieved a 98%+ accuracy during my testing.

## Improvements
In order to improve the model further, we can use the following techniques 

Dropout

Data augmentation

Weight decay

Early stopping
