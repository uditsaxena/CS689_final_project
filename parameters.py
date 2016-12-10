root = './'
deploy = root + 'cnn_models/bvlc_reference_caffenet/deploy.prototxt'
model = root + 'cnn_models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
mean = root + 'imagenet/ilsvrc_2012_mean.npy'

video_path = root + 'data'
video_data_path = root + 'corpus.csv'
# video_feat_path = '/Users/Udit/Downloads/Datasets for ML FP/YouTubeClips/save'
video_features_path = root + 'data/save'

model_path = './models/'
############## Train Parameters #################

dim_image = 4096
dim_hidden = 256
n_frame_step = 80
n_epochs = 1000
batch_size = 100
learning_rate = 0.001
num_frames = 80
