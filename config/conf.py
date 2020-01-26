# Choose your classifier
# from sklearn.linear_model import LogisticRegression as classifier
# from sklearn.ensemble import RandomForestClassifier as classifier
from xgboost import XGBClassifier as classifier

# Set source for opencv VideoCapture: usb index, mp4, rtsp
stream = 0

# Set path to staged data, raw images, pickled sklearn model
csv_path = './data/data_3mdad.csv'
csv_keypoints_path = './data/train_3mdad_keypoints.csv'

images_dir = './data/raw/front_views/train/'
classifier_model = './models/3mdad_randomforestmodel.pkl'

# Set path to pose estimation model asset and body keypoint map
pose_model = './models/pose.tflite'  # resnet18_baseline_att_224x224_A_epoch_249.pth

body_dict = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder', 6: 'right_shoulder',
             7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
             13: 'left_knee', 14: 'right_knee', 15: 'left_akle', 16: 'right_ankle', 17: 'neck'}

"""
{0:'head', 1: 'neck', 2: 'left_shoulder', 3:'lelbow',
4:'left_wrist', 5:'right_shoulder', 6:'relbow', 7:'right_wrist',
8:'left_hip', 9:'left_knee', 10:'left_akle', 11:'right_hip', 12:'right_knee', 13:'right_ankle'}
"""
