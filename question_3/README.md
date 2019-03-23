## Question 3: Slightly More Complicated Neural Networks
1. Scan through the paper on depthwise-separable factorization using MobileNets: [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)
2. Fork the TF implementation of MobileNets and download the pre-trained weights: [https://github.com/Zehaos/MobileNet](https://github.com/Zehaos/MobileNet)
3. Given the weights, write a test script to restore the TF session and run image classification on the polarr image.
4. Find the top 5 labels and their respective probabilities.
5. Using TensorFlow, dump out the name (including BN parameters) and shape of the first 3 layers. I've completed part of it for you, stop at

        conv_ds_2/pw_batch_norm/moving_variance:
        u'global_step:0', ()
        u'MobileNet/conv_1/weights:0', (3, 3, 3, 32),
        u'MobileNet/conv_1/biases:0', (32,),
        ...
        u'MobileNet/conv_ds_2/pw_batch_norm/moving_variance:0', (64,)

6. Instead of doing separate convolution and batch normalization step for each layer, we would like to combine the two steps so that convolution takes care of the normalization. Combine the batch normalization parameters and weights to calculate the batch-normalized weights and biases for only the layer conv_1 of shape 3x3x3x32. You can ignore the biases provided in the model. The formula for calculating BN weights and bias is here: [https://forums.developer.apple.com/thread/65821](https://forums.developer.apple.com/thread/65821). Assume gamma = 1, calculate the weights modifiers and biases for each output channel (32). Do it only for the first convolution layer, conv_1.
7. The current model contains 1000 labels. Using the same model, reduce it to only the following 10 labels: `["pickup", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate", "pitcher", "plane"]` Run the test program again with the photo from question 3, what are the top 5 labels and their probabilities this time?

### Steps to get inference from MobileNet
#### 1. Insert placeholder first
1. I updated the `input_checkpoint` and `save_path` in the `tools/insert_placeholder.py` script
2. Ran the following command to insert placeholder for inference.
```bash
python insert_placeholder.py
```

#### 2. Freeze the graph
Inference graph
```bash
python tools/freeze_graph.py \
    --input_graph ~/Projects/polarr_assignment/data/mobilenet/mobilenet_inference/graph.pbtxt \
    --input_checkpoint ~/Projects/polarr_assignment/data/mobilenet/mobilenet_inference/mobilenet_inference \
    --output_graph ./frozen_graph.pb \
    --output_node_names MobileNet/Predictions/Softmax
```

#### 3. Download original imagenet labels
```python
# In forked mobilenet repository
import pickle
from datasets import imagenet
# URL to synset was broken so update it in the module.
# --> https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt
label_map = imagenet.create_readable_names_for_imagenet_labels()
with open('imagenet_1k_label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)
```

### How to run the script.
#### First, please untar original model and the inference model.
```bash
cd mobilenet
tar -xvf mobilenet_inference.tar.gz
tar -xvf mobilenet_orig.tar.gz
--> you should see a new directory called 'mobilenet_inference' and 'mobilenet_orig'
```

#### Run the script/notebook
Before running the script, please update the checkpoint files to use your path.

**Steps 3 and 4**
```bash
python mobilenet.py get-predictions --img-path ../images/polarr.png --top-k 5

--> Outputs
Prediction 1: coil, spiral, volute, whorl, helix, Probability: 0.5526436567306519

Prediction 2: acoustic guitar, Probability: 0.07745817303657532

Prediction 3: lampshade, lamp shade, Probability: 0.05849877744913101

Prediction 4: lens cap, lens cover, Probability: 0.02784486673772335

Prediction 5: mixing bowl, Probability: 0.026009973138570786
```

**Step 5**
```bash
python mobilenet.py print-layers

--> Outputs
global_step:0 ()
MobileNet/conv_1/weights:0 (3, 3, 3, 32)
MobileNet/conv_1/biases:0 (32,)
MobileNet/conv_1/batch_norm/beta:0 (32,)
MobileNet/conv_1/batch_norm/moving_mean:0 (32,)
MobileNet/conv_1/batch_norm/moving_variance:0 (32,)
MobileNet/conv_ds_2/depthwise_conv/depthwise_weights:0 (3, 3, 32, 1)
MobileNet/conv_ds_2/depthwise_conv/biases:0 (32,)
MobileNet/conv_ds_2/dw_batch_norm/beta:0 (32,)
MobileNet/conv_ds_2/dw_batch_norm/moving_mean:0 (32,)
MobileNet/conv_ds_2/dw_batch_norm/moving_variance:0 (32,)
MobileNet/conv_ds_2/pointwise_conv/weights:0 (1, 1, 32, 64)
MobileNet/conv_ds_2/pointwise_conv/biases:0 (64,)
MobileNet/conv_ds_2/pw_batch_norm/beta:0 (64,)
MobileNet/conv_ds_2/pw_batch_norm/moving_mean:0 (64,)
MobileNet/conv_ds_2/pw_batch_norm/moving_variance:0 (64,)
```

**Step 6**
```bash
Implementation in python notebook `notebooks/mobilenet_conv_batch.ipynb`
cd notebooks
jupyter notebook
```

**Step 7**

Note: I'm not retraining the original model to output the 10 labels. I'm using the original model from question 3.3 to pick the top 5 predictions from the 10 labels.
```bash
python mobilenet.py get-predictions --img-path ../images/polarr.png --top-k 5 --custom-labels-path ../mobilenet/custom_labels.txt

--> Outputs
Prediction 1: pitcher, Probability: 0.02051965519785881

Prediction 2: pillow, Probability: 0.0013868175446987152

Prediction 3: ping-pong ball, Probability: 0.00022996838379185647

Prediction 4: pill bottle, Probability: 0.00012150723341619596

Prediction 5: piggy bank, Probability: 1.167048503702972e-05
 ```