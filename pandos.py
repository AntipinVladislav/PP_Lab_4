import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random


def histo_lists(datafr: pd.DataFrame, class_name: str):
    index = random.randrange(0, 1000)
    if class_name == 'Cat':
        image = cv2.imread(f'dataset/Cats/{str(index).zfill(4)}.jpg')
        b, r, g = cv2.split(image)
        return b, r, g
    else:
        image = cv2.imread(f'dataset/Dogs/{str(index+1000).zfill(4)}.jpg')
        b, r, g = cv2.split(image)
        return b, r, g


def filter(datafr: pd.DataFrame, class_name: str):
    return datafr.loc[datafr["class"] == class_name]


def advanced_filter(datafr: pd.DataFrame, class_name: str, max_width: int, max_height: int):
    return datafr.loc[(datafr["class"] == class_name) & (datafr["width"] <= max_width) & (datafr["height"] <= max_height)]


datafr = pd.read_csv('dataset_new.csv')
# datafr = datafr.drop(['rel'], axis=1)

datafr.loc[datafr["class"] == "Cat", "class_num"] = 0
datafr.loc[datafr["class"] == "Dog", "class_num"] = 1


# datafr[['width', 'height', 'depth', 'pixels']] = datafr['path'].apply(
# lambda x: pd.Series((lambda img: (*img.shape, img.size))(cv2.imread(x))))
print(f'width:  average = {datafr['width'].mean()}   min = {
      datafr["width"].min()}   max = {datafr["width"].max()}')
print(f'height:  average = {datafr['height'].mean()}   min = {
      datafr["height"].min()}   max = {datafr["height"].max()}')
print(f'depth:  average = {datafr['depth'].mean()}   min = {
      datafr["depth"].min()}   max = {datafr["depth"].max()}')


print(datafr.groupby(['class'])['pixels'].mean())
print(datafr.groupby(['class'])['pixels'].min())
print(datafr.groupby(['class'])['pixels'].max())


print(histo_lists(datafr, 'Cat'))
b, r, g = histo_lists(datafr, 'Cat')

fig = plt.figure(figsize=(15, 10))
plt.ylabel('blue pos')

plt.hist(b, bins=3)
plt.show()


# plt.xlabel('y')
# plt.title('hist')

# plt.show()
# print(filter(datafr, 'Dog'))
# print(advanced_filter(datafr, 'Cat', 1200, 12000))
# datafr.to_csv('dataset_new.csv', index=False)
