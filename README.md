# LSTM-Mobility-Model

The Tensorflow implementation of LSTM Mobility Model learns trajectories of human activity trajectories that have spatial and temporal features.

#### What are human activity trajectories?
Human activity trajectories are sequences of stationary activities. Stationary activities refer to someone stay at a location for sometime doing something. So the stationary activities have features include, **start time**, **duration**, **location** and **purpose**.

<img src="https://github.com/zihenglin/LSTM-Mobility-Model/blob/2_layer_models/images/activity_flow.png" height="120px" align="middle">


#### Have other type of time series problems?
The implementations of the LSTM Mobility Model provide good examples of combining discrete and multidimensional continuous features in one model. The implementations can be easily modified for your problem. Feel free to contact me if you need some extra help.

## How does it work?

To model human like decision making for activty choices, I design the two layer model structure shown below. 
The first layer models the activity type choices, such as Home, Work, or Shopping. The second layer makes decision of
activity location and duration according to the activity type decided by the first layer.

<img src="https://github.com/zihenglin/LSTM-Mobility-Model/blob/2_layer_models/images/LSTM_diagram_2_layers.png" height="320px" align="middle">

Layer are represented as ***h***. The input to both layers ***x_t*** is the feature of previous activity. 
The softmax output from the first layer models the activity type choices. The output from the second layer is a mixture distribution that models the spatial and temporal choices.

<img src="https://github.com/zihenglin/LSTM-Mobility-Model/blob/2_layer_models/images/two_layers_sampling_other.png" height="420px" align="center">

## Examples

The `examples` directory contains Jupyter Notebook examples using artifitial data sample:

* [2 Layer Structure with Lat-Lon Location](https://github.com/zihenglin/LSTM-Mobility-Model/blob/2_layer_models/examples/Two%20Layers%20Lat%20Lng%20Location.ipynb) Model activity latitude and longitude location using Guassian mixture distributions.

* [2 Layer Structure with Location ID](https://github.com/zihenglin/LSTM-Mobility-Model/blob/2_layer_models/examples/Two%20Layers%20Categorical%20Location.ipynb) Prepocess location into location categoreis and model location categories.



