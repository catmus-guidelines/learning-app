from datasets import load_dataset, DatasetDict, Dataset
from collections import defaultdict
import pandas as pd
import tqdm


def filter_samples_in_batches(dataset_split, batch_size=10000):
    shelfmark_counts = defaultdict(int)
    new_samples = []

    try:
        # Iterate over the dataset and collect up to 10 samples per shelfmark
        for sample in tqdm.tqdm(dataset_split):
            shelfmark = sample['shelfmark']
            if shelfmark_counts[shelfmark] < 10:
                new_samples.append(sample)#.to_dict())
                shelfmark_counts[shelfmark] += 1
            # if shelfmark_counts[shelfmark] > 2:
            #     break
    except Exception as E:
        print(E)

    return new_samples

train_dataset = load_dataset('../medieval', split='train', streaming=True)
validation_dataset = load_dataset('../medieval', split='validation', streaming=True)
test_dataset = load_dataset('../medieval', split='test', streaming=True)

# Filter each split of the dataset
new_train_samples = filter_samples_in_batches(train_dataset)
new_validation_samples = filter_samples_in_batches(validation_dataset)
new_test_samples = filter_samples_in_batches(test_dataset)

# Convert samples to dictionary format suitable for Dataset.from_dict
def samples_to_dict(samples):
    # Initialize dictionary with empty lists for each key
    sample_dict = defaultdict(list)
    
    for sample in samples:
        for key, value in sample.items():
            # if isinstance(value, np.ndarray):
            #     sample_dict[key].append(value.tolist())  # Convert numpy arrays to lists
            # elif hasattr(value, 'convert'):
            #     sample_dict[key].append(np.array(value.convert('RGB')).tolist())  # Convert PIL images to lists
            # else:
            sample_dict[key].append(value)
    
    return dict(sample_dict)

new_train_dict = samples_to_dict(new_train_samples)
new_validation_dict = samples_to_dict(new_validation_samples)
new_test_dict = samples_to_dict(new_test_samples)


# Convert the filtered DataFrames back to Dataset objects
new_train_dataset = Dataset.from_dict(new_train_dict)
new_validation_dataset = Dataset.from_dict(new_validation_dict)
new_test_dataset = Dataset.from_dict(new_test_dict)

# Combine the new datasets into a DatasetDict
new_dataset = DatasetDict({
    'train': new_train_dataset,
    'validation': new_validation_dataset,
    'test': new_test_dataset
})

# Save the new dataset (optional)
new_dataset.save_to_disk('filtered_dataset')

# Print some statistics for verification
print(f"New train size: {len(new_dataset['train'])}")
print(f"New validation size: {len(new_dataset['validation'])}")
print(f"New test size: {len(new_dataset['test'])}")

print(new_dataset)