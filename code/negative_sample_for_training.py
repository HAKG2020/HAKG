# This is used to sample negative movies that user has not interactions with, so as to balance model training process

import argparse
import math
import numpy as np
from random import randint


def load_train_data(file):
    train_dict = {}
    all_artist_list = []
    for line in file:
        lines = line.replace('\n', '').replace('\r', '').split('\t')
        user = lines[0]
        artist = lines[1]
        if user not in train_dict:
            train_dict.update({user:[artist]})
        else:
            train_dict[user].append(artist)
        if artist not in all_artist_list:
            all_artist_list.append(artist)
    return train_dict, all_artist_list


def negative_sample(all_dict,all_artist_list,train_dict, train_artist_list, shrink, negative_sample_file):

    all_artist_num = len(all_artist_list)
    print('all_artist in training ', all_artist_num)
    for user in train_dict:

        positive_artist_list = all_dict[user]
        positive_num = len(positive_artist_list)
        negative_num = math.ceil(positive_num * shrink)

        if positive_num + negative_num >= all_artist_num:
           negative_num = math.ceil(positive_num * 1)


        negative_artist_list = []
        print('u'+str(user), positive_num, negative_num)
        while (len(negative_artist_list) < negative_num):
            negative_index = np.random.randint(0, (all_artist_num - 1))
            negative_artist = str(all_artist_list[negative_index])
            if negative_artist not in positive_artist_list not in negative_artist_list:
                negative_artist_list.append(negative_artist)

        for negative_artist in negative_artist_list:
            line = user + '\t' + negative_artist + '\n'
            negative_sample_file.write(line)


if __name__ == '__main__':

    user_artists_file_name = 'ml/user_movies.txt'
    train_file_name = 'ml/training.txt'

    shrink = 1
    negative_sample_file_name = 'ml/negative.txt'

    all_dict, all_artist_list = load_train_data((open(user_artists_file_name, 'r')))
    train_dict, train_artist_list = load_train_data((open(train_file_name,'r')))

    negative_sample(all_dict,all_artist_list,train_dict, train_artist_list, shrink, open(negative_sample_file_name,'w'))

