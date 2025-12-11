import random
import shutil
import os 

target = 30
dataset_dir = "./DataSet"

for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    images = [f for f in os.listdir(person_path) if f.endswith(".jpg")]
    if len(images) > target:
        # randomly keep only target images
        keep = random.sample(images, target)
        for img in images:
            if img not in keep:
                os.remove(os.path.join(person_path, img))
