from pathlib import Path

from collections import defaultdict

import matplotlib.pyplot as plt
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
    new_ds_root = new_ds_root.with_name(new_ds_name + '_companion') # Change to complementary "companion" folder

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
                temp_dict['runtime'].append(measurement)

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

# Create a <dataset_names> x <K> tableau of subplots
dataset_unique_names = df['ds_name'].unique()
K_unique_values = df['K'].sort_values().unique()

fig, axes = plt.subplots(len(dataset_unique_names), len(K_unique_values), figsize=(20, 20))
# Adjust the spacing between subplots
fig.subplots_adjust(hspace=0.5, wspace=0.2)

for i, dataset_str in enumerate(dataset_unique_names):
    for j, K in enumerate(K_unique_values):
        # Create the boxplots in each subfigure.
        s = sns.boxplot(data=df[(df['ds_name'] == dataset_str) & (df['K'] == K)],
                    x = 'algorithm',
                    y='runtime',
                    hue='is_companion',

                    ax=axes[i, j],
                    showfliers=False # Do not plot the outliers.
        )
        # s.set_title(s.get_title(), fontsize=22)
        s.set_yticklabels(s.get_yticklabels(), fontsize=16)
        s.set_xticklabels(s.get_xticklabels(), fontsize=16)
        s.set_ylabel(s.get_ylabel(), fontsize=16)
        # s.set_xlabel(s.get_xlabel(), fontsize=20)

        # Set custom title for each subplot
        # for easier readability.
        axes[i, j].set_title(f'Dataset: {dataset_str}, K = {K}')
        # Change the y-label to include the time unit,
        # as instructed by the project mentor.
        axes[i, j].set_ylabel('log(runtime) (ns)')
        # NOTE: Remove 'log' in case that the y-scaling below is removed.

        # Set a log scale on the y-axis (runtime)
        # It help visiblity for most cases.
        axes[i,j].set_yscale('log') 

        # Draw vertical lines between the x-axis ("algorithm")
        # for better visibility.
        for alg in range(len(df['algorithm'].unique())):
            axes[i, j].axvline(x=alg + 0.5, color='gray', linestyle='--')  # Adjust the position of the vertical lines

# save_name = "3x3-plot"
save_name = "3x3-plot"
if save_name == "":
    plt.show()
else:
    save_name += ".png"
    plt.savefig(save_name, bbox_inches="tight")