import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

class CustomRandomLinkSplit(RandomLinkSplit):
    def __init__(self, num_val: float = 0.1, num_test: float = 0.2,
                 is_undirected: bool = False, add_negative_train_samples: bool = True):
        super().__init__(num_val=num_val, num_test=num_test, 
                         is_undirected=is_undirected, 
                         add_negative_train_samples=add_negative_train_samples)
    
    def __call__(self, data: Data):
        # Custom pre-processing or checks can be added here
        print("Custom preprocessing before split")
        
        # Call the parent class's __call__ method to perform the actual split
        train_data, val_data, test_data = super().__call__(data)
        
        # Custom post-processing can be added here
        print("Custom postprocessing after split")
        
        return train_data, val_data, test_data

# Example usage:
data = Data()  # Assume this is your graph data object
transform = CustomRandomLinkSplit(num_val=0.15, num_test=0.25)
train_data, val_data, test_data = transform(data)

# Verify the splits
print(train_data)
print(val_data)
print(test_data)



