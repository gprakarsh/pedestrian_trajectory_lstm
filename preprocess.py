from trajnetplusplustools import Reader
import numpy as np
from torch.autograd import Variable


def get_trajectories(file_path = "./train/real_data/biwi_hotel.ndjson"):
    trajnet_reader = Reader(file_path)
    tracks_by_frame = trajnet_reader.tracks_by_frame
    df = {}
    for frame_num in tracks_by_frame.keys():
        for track in tracks_by_frame[frame_num]:
            if track.pedestrian not in df:
                df[track.pedestrian] = []
            df[track.pedestrian].append([track.x, track.y])

    for pedestrian in df.keys():
        df[pedestrian] = np.array(df[pedestrian])

    return df


def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return x, y



def process_dataset(file_path):
    df = get_trajectories(file_path)
    seq_length = 8
    x, y = [], []
    for pedestrian_data in df.values():
        currX, currY = sliding_windows(pedestrian_data, seq_length)
        x.extend(currX)
        y.extend(currY)

    return x, y


def process_datasets(file_paths):
    trainX, trainY = [], []
    
    for file_path in file_paths[0:2]:
        currX, currY = process_dataset(file_path)
        print("Loaded", file_path)
        trainX.extend(currX)
        trainY.extend(currY)

    return trainX, trainY


file_paths = ["./train/real_data/biwi_hotel.ndjson", "./train/real_data/cff_06.ndjson", "./train/real_data/cff_07.ndjson", "./train/real_data/cff_08.ndjson", "./train/real_data/cff_09.ndjson", "./train/real_data/cff_10.ndjson", "./train/real_data/cff_12.ndjson", "./train/real_data/cff_13.ndjson", "./train/real_data/cff_14.ndjson", "./train/real_data/cff_15.ndjson", "./train/real_data/cff_16.ndjson", "./train/real_data/cff_17.ndjson", "./train/real_data/cff_18.ndjson", "./train/real_data/crowds_students001.ndjson", "./train/real_data/crowds_students003.ndjson", "./train/real_data/crowds_zara01.ndjson", "./train/real_data/crowds_zara03.ndjson", "./train/real_data/lcas.ndjson"]

x, y = process_datasets(file_paths[0:1])
seq_length = 8