import os
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.tensor_forest.client import eval_metrics, random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from ops.extract_tfrecords import get_records


def train_rf(num_features, config):
    """Build the tf model."""
    params = tensor_forest.ForestHParams(
        num_classes=config.num_classes, num_features=num_features,
        num_trees=config.num_trees, max_nodes=config.max_nodes)
    graph_builder_class = tensor_forest.RandomForestGraphs
    if config.use_training_loss:
        graph_builder_class = tensor_forest.TrainingLossForest
    return random_forest.TensorForestEstimator(
        params, graph_builder_class=graph_builder_class,
        model_dir=config.model_output)


def train_and_eval(config):
    """Train and evaluate the model."""
    print 'model directory = %s' % config.model_output

    num_features = 1e3
    model = train_rf(num_features, config)

    # Early stopping if the forest is no longer growing.
    monitor = random_forest.TensorForestLossHook(config.early_stopping_rounds)

    # TFLearn doesn't support tfrecords; extract them by hand for now
    img, label, feat = get_records(
        os.path.join(config.tfrecord_dir, 'train.tfrecords'))
    model.fit(
        x=feat, y=label,
        batch_size=config.batch_size, monitors=[monitor])

    metric_name = 'accuracy'
    metric = {metric_name: metric_spec.MetricSpec(
                eval_metrics.get_metric(metric_name),
                prediction_key=eval_metrics.get_prediction_key(metric_name))}

    test_img, test_label, test_feat = get_records(
        os.path.join(config.tfrecord_dir, 'val.tfrecords'))
    results = model.evaluate(
        x=test_img, y=test_label,
        batch_size=config.batch_size, metrics=metric)
    return results
