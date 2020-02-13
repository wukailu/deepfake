BASE = "/data1/data/deepfake/"
DATA_DIR = BASE + "dfdc_train"
meta_data_path = [
    DATA_DIR + "/metadata_kailu.json",
    DATA_DIR + "/metadata_kailu_hard.json",
    DATA_DIR + "/metadata_kailu_460.json",
]
bbox_path = BASE + "bbox_real.csv"
cache_path = BASE + "faces/"
USE_FOUNDATIONS = True
un_normal_list = []
test_model_path = '~/deepfake/model1/result/checkpoints/best_model.pth'


# {'batch_size': 64, 'n_epochs': 20, 'weight_decay': 0.0001, 'dropout': 0.7, 'augment_level': 3, 'max_lr': 0.0003, 'use_lr_scheduler': 0, 'scheduler_gamma': 0.96, 'use_hidden_layer': 0, 'backbone': 'resnet34', 'val_rate': 1, 'data_path': '/data1/data/deepfake/dfdc_train', 'metadata_path': '/data1/data/deepfake/dfdc_train/metadata_kailu.json', 'bbox_path': '/data1/data/deepfake/bbox_real.csv', 'cache_path': '/data1/data/deepfake/face/', 'seed': 1378744497}
