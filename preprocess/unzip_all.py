# unzip all dfdc_train_part_xx.zip and merge them
import zipfile
import pandas as pd
import os
import shutil

all_data_path = "/data1/data/deepfake/"

if not os.path.isdir(os.path.join(all_data_path, "dfdc_train")):
    os.mkdir(os.path.join(all_data_path, "dfdc_train"))
for i in range(50):
    print(str(i)+"/50")
    base = all_data_path + "dfdc_train_part_%02d" % i
    base2 = all_data_path + "dfdc_train_part_%d" % i
    with zipfile.ZipFile(base + ".zip", "r") as zip_ref:
        zip_ref.extractall(all_data_path)
        os.rename(base2 + "/metadata.json", base2 + "/metadata_%02d.json" % i)

        for path, dir_list, file_list in os.walk(base2):
            for file in file_list:
                shutil.move(os.path.join(path, file), os.path.join(all_data_path, "dfdc_train", file))
        os.rmdir(base2)

base = os.path.join(all_data_path, "dfdc_train")
meta_data = pd.read_json(os.path.join(base, "metadata_00.json"))
for i in range(1, 50):
    print(str(i)+"/50")
    meta_data = meta_data.join(pd.read_json(os.path.join(base, "metadata_%02d.json" % i)))

print("shape: ", meta_data.shape)
print(meta_data.head())
meta_data.to_json(os.path.join(base, "metadata_all.json"))
