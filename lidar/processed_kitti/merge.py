import os
import numpy as np

root = './'

all_ds  = []
for env in ['residential', 'city', 'road']:

    for recording in os.listdir(os.path.join(root, env)):

        all_chunks = os.listdir(os.path.join(root, env, recording))

        all_chunks = [x for x in all_chunks if 'processed_' in x]

        all_array = []

        for chunk_id in range(len(all_chunks)):
            array = np.load(os.path.join(root, env, recording, 'processed_%d.npz' % chunk_id))

            array = [array[i] for i in array.keys()]

            all_array += array

        all_ = {str(i):all_array[i] for i in range(len(all_array))}
        path_out = os.path.join(root, env, recording, 'processed.npz')
        np.savez_compressed(path_out, **all_)

        print('saved %s' % path_out)

# delete the split files
os.system('rm -rf */*/processed_*')

# remove github traces
os.sytem('rm -rf .git')
