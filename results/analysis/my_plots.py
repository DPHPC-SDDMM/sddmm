from pathlib import Path

from collections import defaultdict

import seaborn as sns
import pandas as pd

from data import Data

def update_dataset_collection(collection: dict, new_ds_name: str, datasets_root: Path):
    new_ds_root = datasets_root / new_ds_name

    for filepath in new_ds_root.iterdir():
        filename = filepath.name
        if filename != 'readme.txt': # Ignore `readme.txt` files. They do not contain data information.

            dataset = Data('/' + str(new_ds_root) + '/', str(filename))
            collection[(new_ds_name, False, dataset.params['K'])] = dataset

    # Almost identical code with the above
    new_ds_root.with_name(new_ds_name + '_companion') # Change to complementary "companion" folder

    for filepath in (new_ds_root).iterdir():
        filename = filepath.name
        if filename != 'readme.txt': # Ignore `readme.txt` files. They do not contain data information.

            dataset = Data('/' + str(new_ds_root) + '/', str(filename))
            collection[(new_ds_name, True, dataset.params['K'])] = dataset # THE CHANGE IS HERE (`True`)!


def generate_datapoints_dataframe(collection: dict):
    temp_dict = defaultdict(list)

    for k, v in collection.items():
        database_name, is_companion, K = k

        for algorithm_name, measurements in v.data.items():

            for id, measurement in enumerate(measurements):
                
                
                temp_dict['ds_name'].append(database_name)
                temp_dict['is_companion'].append(is_companion)
                temp_dict['K'].append(K)
                temp_dict['algorithm'].append(algorithm_name)
                temp_dict['datapoint_id'].append(id)
                temp_dict['value'].append(measurement)

    return pd.DataFrame(temp_dict)


nr_iterations = 100
datasets_root = Path(f'./sddmm_data_results_{nr_iterations}_iterations/data_sets/')

# name_comp_k_to_data[(<dataset name>, <bool: "companion" or not?>, <inner dimension K>) ] -> `Data` object.
name_comp_k_to_data = defaultdict(Data)
# Gather all datasets to be used in a single dictionary.
update_dataset_collection(name_comp_k_to_data, 'IMDB', datasets_root)
update_dataset_collection(name_comp_k_to_data, 'patents', datasets_root)
update_dataset_collection(name_comp_k_to_data, 'patents_main', datasets_root)
# Create corresponding dataframe for easier manipulation when choosing what to plot.
# Highly recommended in case where plots are created with `Seaborn`,
# which directly support Pandas dataframes. 
df = generate_datapoints_dataframe(name_comp_k_to_data)
print(df)

