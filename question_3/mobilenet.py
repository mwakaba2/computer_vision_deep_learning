from argparse import ArgumentParser
import pickle
import os

import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

META_GRAPH = f"../mobilenet/mobilenet_orig/model.ckpt-906808.meta"
GRAPH_INF_FILENAME = f"../mobilenet/mobilenet_inference/frozen_graph.pb"


def preprocess_img(image_path):
    # Read image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.expand_dims(img, 0)

    # Preprocess
    img = img / 255.
    img = img - 0.5
    img = img * 2.

    return img


def get_label_map(custom_labels_path):
    with open('../mobilenet/imagenet_1k_label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    if custom_labels_path is not None:
        custom_label_map = {}
        with open(custom_labels_path, 'r') as f:
            labels = set(f.read().splitlines())
        for key, value in label_map.items():
            values = value.split(',')
            for value in values:
                if value in labels:
                    custom_label_map[key] = value
        label_map = custom_label_map

    return label_map


def get_graph_def(train=False):
    if train:
        filename = GRAPH_FILENAME
    else:
        filename = GRAPH_INF_FILENAME

    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def get_preds(*, image_path, top_k, custom_labels_path):
    img = preprocess_img(image_path)
    label_map = get_label_map(custom_labels_path)
    graph_def = get_graph_def()

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        input_node = graph.get_tensor_by_name('import/MobileNet/input_images:0')
        output_node = graph.get_tensor_by_name('import/MobileNet/Predictions/Softmax:0')

        with tf.Session() as sess:
            predictions = sess.run(output_node, feed_dict={input_node: img})[0]
            if custom_labels_path is None:
                top_k_predictions = predictions.argsort()[-top_k:][::-1]
                top_k_probabilities = predictions[top_k_predictions]
                prediction_names = [label_map[i] for i in top_k_predictions]
            else:
                custom_label_indices = list(label_map.keys())
                filtered_predictions = np.take(predictions, custom_label_indices)
                top_k_predictions =  filtered_predictions.argsort()[-top_k:][::-1]
                top_k_orig_indices = [custom_label_indices[idx] for idx in top_k_predictions]
                top_k_probabilities = filtered_predictions[top_k_predictions]
                prediction_names = [label_map[i] for i in top_k_orig_indices]

            pred_probs = zip(prediction_names, top_k_probabilities)

            for idx, (prediction, probability) in enumerate(pred_probs, 1):
                print((f'Prediction {idx}: {prediction}, '
                      f'Probability: {probability}\n'))


def print_layers():
    saver = tf.train.import_meta_graph(META_GRAPH)
    imported_graph = tf.get_default_graph()
    graph_operations = imported_graph.get_operations()

    for op in graph_operations:
        if op.name.startswith('MobileNet/conv_ds_3'):
            # only print the first three layers
            # up until MobileNet/conv_ds_2
            break
        if op.type == 'VariableV2':
            name = op.name
            values = op.values()
            shape = values[0].shape if values else None
            print(f'{name}:0', shape)


def get_args():
    arg_parser = ArgumentParser()
    subparsers = arg_parser.add_subparsers(dest='command')

    parser_get_preds = subparsers.add_parser('get-predictions')
    parser_get_preds.add_argument('--img-path',
                                  required=True,
                                  type=str,
                                  help=('Input image.'))
    parser_get_preds.add_argument('--top-k',
                                  default=5,
                                  type=int,
                                  help=('Get the top k predictions.'))
    parser_get_preds.add_argument('--custom-labels-path',
                                  type=str,
                                  help=('Text file to list of custom labels'))
    parser_get_preds.set_defaults(func=get_preds)

    parser_update_orgs = subparsers.add_parser(
        'print-layers', help=('Print the first 3 layers of the network.'))
    parser_update_orgs.set_defaults(func=print_layers)

    args = arg_parser.parse_args()
    return args


def main():
    args = get_args()

    if args.command == 'get-predictions':
        args.func(image_path=args.img_path,
                  top_k=args.top_k,
                  custom_labels_path=args.custom_labels_path)
    elif args.command == 'print-layers':
        args.func()


if __name__ == '__main__':
    main()
