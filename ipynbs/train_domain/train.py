
import sys
import pickle as pkl
import numpy as np
from glob import glob
from tqdm import tqdm


train_datasets = ["cityscapes_train", "sim10k"]
test_datasets = ["cityscapes", "sim200k"]
ngc_id = sys.argv[1] #2838351

print("Load train features")
train_features = []
train_labels = []
for i, dset in tqdm(enumerate(train_datasets)):
    p = list(glob(f"/home/andliao/results/extract_features/*/{ngc_id}_{dset}.pkl"))
    assert len(p) == 1
    p = p[0]

    outs = pkl.load(open(p, "rb"))
    for out in outs:
        feat = out[-1]
        train_features.append(feat[0].mean(-1).mean(-1))     # average pooling
        train_labels.append(i)


train_features = np.array(train_features)
train_labels = np.array(train_labels)


print("Load test features")
test_features = []
test_labels = []
for i, dset in tqdm(enumerate(test_datasets)):
    p = list(glob(f"/home/andliao/results/extract_features/*/{ngc_id}_{dset}.pkl"))
    assert len(p) == 1
    p = p[0]

    outs = pkl.load(open(p, "rb"))
    for out in outs:
        feat = out[-1]
        test_features.append(feat[0].mean(-1).mean(-1))     # average pooling
        test_labels.append(i)

test_features = np.array(test_features)
test_labels = np.array(test_labels)


print("Fit classifier")
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


avg_acc = []
for seed in [1, 2, 3, 4, 5]:
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, random_state=seed))
    clf.fit(train_features, train_labels)

    acc = np.mean(clf.predict(test_features) == test_labels)
    avg_acc.append(acc)

print(f"[{ngc_id}] Accuracy: {np.mean(avg_acc):.3f}")
