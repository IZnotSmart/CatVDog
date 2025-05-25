from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

#seed random number gen
seed(1)

#create directories
dataHome = 'dataCatDog/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    #label the subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labeldir in labeldirs:
        newdir = dataHome + subdir + labeldir
        makedirs(newdir, exist_ok=True)

#Ratio of Training:Testing data
valRatio = 0.25
#Copy the training data into subdirectories
srcDirec = 'train/'
for file in listdir(srcDirec):
    src = srcDirec + '/' + file
    #Destination directory
    destDir = 'train/'
    if random() < valRatio:
        destDir = 'test/'
    if file.startswith('cat'):
        dest = dataHome + destDir + 'cats/' + file
        copyfile(src,dest)
    elif file.startswith('dog'):
        dest = dataHome + destDir + 'dogs/' + file
        copyfile(src,dest)



print('end')
