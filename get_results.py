import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

from omegaconf import OmegaConf

def load_result_files(cfg):
    with open(cfg.artifacts.indexes_path, 'rb') as f:
        indexes = np.load(f)
    with open(cfg.artifacts.targets_path, 'rb') as f:
        targets = np.load(f)
    with open(cfg.artifacts.predicts_path, 'rb') as f:
        predicts = np.load(f)
    return indexes, targets, predicts

if __name__ == "__main__":

    cfg = OmegaConf.load("config.yaml")

    indexes, targets, predicts = load_result_files(cfg)

    print(f"Accuracy: {accuracy_score(targets, predicts)}")
    print("-----")

    classes = ["car", "airplane", "motorcycle", "elephant", "zebra"]
    labels = [cfg.class_to_id_mapping[cls_] for cls_ in classes]

    precision, recall, fscore, support = score(targets, predicts, labels=labels)
    for class_, pr, rc, f1, sup in zip(
        classes, precision, recall, fscore, support
    ):
        print(f"{class_} precision: {pr}")
        print(f"{class_} recall: {rc}")
        print(f"{class_} fscore: {f1}")
        print(f"{class_} support: {sup}")
        print("-----")

    print("Confusion matrix (row - target, column - predict) for car, airplane, motorcycle: ")
    print(confusion_matrix(targets, predicts, labels=labels))

    id_to_class_mapping = {value: key for key, value in cfg.class_to_id_mapping.items()}
    print("Errors:")
    for ind, target, predict in zip(indexes, targets, predicts):
        if target != predict:
            true_label = id_to_class_mapping[target]
            pred_label = id_to_class_mapping[predict]
            print(f"Image with index {ind}, target: {true_label}, predict: {pred_label}")
    
    #with open(cfg.artifacts.images_path, 'rb') as f:
    #    paths = np.load(f)
    #
    #paths_indexes_list = [348, 428, 692, 849, 964, 1201, 1426, 1473, 1478, 1510, 1604]
    #for path_ind in paths_indexes_list:
    #    print(paths[path_ind])