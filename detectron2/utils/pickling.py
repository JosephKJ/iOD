from store import Store
import torch
import os
import pickle

store = Store(10, 3)
store.add(('a', 'b', 'c', 'd', 'e', 'f'), (1, 1, 9, 1, 0, 1))
store.add(('h',), (4,))
print(store.retrieve())

IMAGE_STORE_LOC = '/home/joseph/detectron2'

file_path = os.path.join(IMAGE_STORE_LOC, "image_store.pth.726047903")

# torch.save(store, file_path)
obj = torch.load(file_path)

# with open(file_path, 'wb') as f:
#     pickle.dump(store, f)

# with open(file_path, 'rb') as f:
#     obj = pickle.load(f)

print(len(obj))
