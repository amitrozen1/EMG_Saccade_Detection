A=0
B=1

current_batch = 1
data_paths = [r"C:\Liron\DataEmg\try", r"C:\Liron\DataEmg\new_batch"]
data_path = data_paths[current_batch-1]

#data_path = "/mnt/c/Liron/DataEmg/try"
original_frequency = 4000
num_components = 16
saved_images_extention = ".jpeg"

def set_current_batch(batch):
    global current_batch
    current_batch = batch
    global data_path
    data_path = data_paths[current_batch-1]

def get_current_batch():
    return current_batch

def get_data_path():
    return data_path