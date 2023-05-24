from acvl_utils.instance_segmentation.instance_matching import compute_all_matches
import numpy as np
from argparse import ArgumentParser
import tifffile
import glob
import os


def load_fn(file_path):
    return tifffile.imread(file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="./pred5",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/resized_256", 
    )
    args = parser.parse_args()

    pred_list = sorted(glob.glob(os.path.join(args.pred_dir, '*/*.tif')))
    gt_list = sorted(glob.glob(os.path.join(args.gt_dir, 'c_GT_resized_256/*.tif')))
    
    # mean instance dice for each image over all images
    print('Computing Scores...')
    results = compute_all_matches(gt_list, pred_list, load_fn, num_processes=12)
    #print('Done')
    ID = np.array([np.array(([r[2] for r in results[i]])).mean() for i in range(len(results))]).mean()
    print('Mean Instance Dice:', ID)

    obj_level_f1_per_img = []
    for img in range(len(results)):
        TP = np.sum([i[0] is not None and i[1] is not None for i in results[img]])
        FP_PLUS_FN = np.sum([i[0] is None or i[1] is None for i in results[img]])
        obj_level_f1 = 2*TP / (2*TP + FP_PLUS_FN)
        obj_level_f1_per_img.append(obj_level_f1)
    
    ObjF1 = np.array(obj_level_f1_per_img).mean()

    print('Obj Level F1:', ObjF1)

    print('Score:', np.mean([ID, ObjF1]))

