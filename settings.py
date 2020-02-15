import glob

BASE = "/data/deepfake/"
DATA_DIR = BASE + "dfdc_train"
meta_data_path = glob.glob(BASE+"metadata/metadata_*.json")
bbox_path = BASE + "bbox_real.csv"
face_cache_path = BASE + "faces/"
img_cache_path = BASE + "shots/"
USE_FOUNDATIONS = True
un_normal_list = []
test_model_path = '~/deepfake/model1/result/checkpoints/best_model.pth'

