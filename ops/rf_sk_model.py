import os
from sklearn.ensemble import RandomForestClassifier as rf
from ops.extract_tfrecords import get_records


def train_and_eval(config):
    """Train and evaluate the model."""
    print 'model directory = %s' % config.model_output

    model = rf(
        n_estimators=config.num_trees, max_depth=config.tree_depth,
        max_leaf_nodes=config.max_nodes)

    img, label, feat = get_records(
        os.path.join(config.tfrecord_dir, 'train.tfrecords'), config)
    model.fit(
        x=feat, y=label)

    test_img, test_label, test_feat = get_records(
        os.path.join(config.tfrecord_dir, 'val.tfrecords'), config)
    results = model.predict(
        x=test_img, y=test_label)
    return results, model
