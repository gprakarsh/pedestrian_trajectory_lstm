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
    x, y, dataset = [], [], []
    for pedestrian_data in df.values():
        currX, currY = sliding_windows(pedestrian_data, seq_length)
        x.extend(currX)
        y.extend(currY)
        dataset.extend(pedestrian_data)

    return x, y, dataset


def process_datasets(file_paths):
    trainX, trainY, dataset = [], [], []
    
    for file_path in file_paths:
        currX, currY, currDataset = process_dataset(file_path)
        print("Loaded", file_path)
        trainX.extend(currX)
        trainY.extend(currY)
        dataset.extend(currDataset)

    return trainX, trainY, dataset