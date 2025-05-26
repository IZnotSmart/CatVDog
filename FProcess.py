from os import makedirs
from os import listdir
from shutil import copyfile

#create directories
dataHome = 'FdataCatDog/'
labeldirs = ['dog/', 'cat/']
for labeldir in labeldirs:
    newdir = dataHome + labeldir
    makedirs(newdir, exist_ok=True)

#Copy the training data into subdirectories
srcDirec = 'train/'
for file in listdir(srcDirec):
    src = srcDirec + '/' + file
    #Destination directory
    if file.startswith('cat'):
        dest = dataHome + 'cat/' + file
        copyfile(src,dest)
    elif file.startswith('dog'):
        dest = dataHome + 'dog/' + file
        copyfile(src,dest)

print("end")