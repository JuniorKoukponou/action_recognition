import os
import pandas as pd
from transformer import PoseExtractor
import cv2

if __name__ == '__main__':
    import argparse
    import importlib
    parser = argparse.ArgumentParser(description='Generate csv from image/video data')
    parser.add_argument('--config', type=str, default='conf',
                        help="name of config.py file inside config/ directory, default: 'conf'")
    args = parser.parse_args()
    config = importlib.import_module('config.' + args.config)

    labeled_images, labeled_keypoints = [], []
    for subdir, dirs, files in os.walk(config.images_dir):
        for img in files:
            image_path = os.path.join(subdir, img)
            label = subdir.split('/')[-1]
            labeled_images.append([image_path, label])

            extractor = PoseExtractor()
            image = cv2.imread(image_path)
            sample = extractor.transform([image])
            labeled_keypoints.append([', '.join(map(str, sample)), label])


    df = pd.DataFrame(labeled_images, columns=['image', 'label'])
    df.to_csv(config.csv_path, encoding='utf-8', index=False)

    df_keypoints = pd.DataFrame(labeled_keypoints)
    # df_keypoints.columns = ['keypts', 'label']
    # df_keypoints = pd.concat([df_keypoints['keypts'].str.split(expand=True,), df_keypoints['label']], axis=1)
    df_keypoints.to_csv(config.csv_keypoints_path, encoding='utf-8', index=False)
