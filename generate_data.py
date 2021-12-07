import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer

def create_data_for_causal_vae(data_dir, buffer_path):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    buffer = VectorReplayBuffer.load_hdf5(buffer_path)
    actions = []
    concepts = []
    obs_paths = []
    for i in tqdm(range(len(buffer))):
        xp = buffer[i]

        obs_path = os.path.join(data_dir, f'{i}.npy')
        with open(obs_path, 'wb') as f:
            np.save(f, xp.obs)

        obs_paths.append(obs_path)

        actions.append(xp.act)
        concepts.append(xp.info.concepts)

    actions = np.stack(actions, axis=0)
    if len(actions.shape) < 2:
        actions = np.expand_dims(actions, axis=-1)
    concepts = np.stack(concepts, axis=0)
    xp_df = pd.DataFrame(dict({'Observation': obs_paths},
                              **{f'Action_{a}': actions[:, a] for a in range(actions.shape[1])},
                              **{f'Concept_{c}': concepts[:, c] for c in range(concepts.shape[1])}))
    xp_df.to_csv(os.path.join(data_dir, 'xp_df.csv'), index=False)
    return

if __name__=='__main__':
    data_dir = 'B:/Datasets/School/CausalVAE/cartpole2/train'
    buffer_path = 'data/experience/cartpole2/train/buf.hdf5'
    create_data_for_causal_vae(data_dir, buffer_path)