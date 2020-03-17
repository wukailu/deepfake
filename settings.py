import glob

BASE = "/data/deepfake/"
DATA_DIR = "/data1/data/deepfake/dfdc_train"
meta_data_path = [BASE + "metadata/metadata_40_49_dropped.json"]
# meta_data_path = [BASE + "metadata/metadata_half_dropped.json"]
# meta_data_path = [BASE + "metadata/metadata_075.json"]
# meta_data_path = glob.glob(BASE+"metadata/metadata_*.json")
bbox_path = BASE + "bbox_real.csv"
face_cache_path = BASE + "faces/"
test_face_cache_path = BASE + "tests/"
img_cache_path = BASE + "shots/"
video_cache_path = BASE + "video/"
video_cache_path2 = BASE + "video2/"
continue_cache_path = BASE + "continue/"
USE_FOUNDATIONS = True
un_normal_list = []
test_model_path = '~/deepfake/model1/result/checkpoints/best_model.pth'
diff_dict_path = BASE + "diff.csv"
