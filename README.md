# Autonomous Driving Human Trajectory Prediction

This is a pilot project, our purpose is to train models to predict human trajectories given the past trajectory in developing autonomous driving. 

Our self-created datasets (default 6000 sets of map and trajectory, which can be made into >15k training samples) simulated human walking cross the crossroad through the zebra crossing. 

#### VRU prediction approaches specification

the challenge of this project is to combine context information with the trajectory information (x, y coordinate here) in predicting the future trajectory with RNN. Different approahces we've tried differ in their ways of encoding context info and combining with the trajecotory featuers. 


* x,y: the x, y coordniate feature on the original context

* delta x,y: the relative x, y coordinate of each point relative to its previous x, y position on the trajectory

* context: the original context map (default 1280 * 1280)

* context_patch: a patch of the orginal context cropped for each data point on a trajectory as the center. for a training sample, the shape is usually (None, sequence_length, patch_size * patch_size)



<pre>
model	 x,y    delta x,y   context  context_patch   CNN      RNN   fc

vru:	 -	    - 	      -		            context   all     -
vru2:               -                     -         context_patch     all     -
vru3:  	 -	   (-)	      -                     context   x/y (or x,y + delta x,y)    -
vru_s:   	    -	      -			              all     -

'-' means in use.
'all' in RNN means the x,y or delta x,y featuers with the output vector of CNN (or context itself).
</pre>

Here, I only included the code example of vru2 approach.

#### VAE for context 
Aside from using a simpe RNN to encode the context image, we also tried to use an autoencoder vae to encode the context patches and then combine with the delta x, y feature to train in RNN.The following is a visualization of a trained vae model for context images. The **left** blurried one is a vae reconstructed image, compared with its immediate **right** image which is the orginal one.

<img src="https://raw.githubusercontent.com/celisun/autonomous_driving_human_trajectory_prediction/master/img/I_reconstructed0.png" width="500">

#### Empirical results

So far we observed that a simple CNN encoding of context or a patch building still could not good enough combine context information with the trajectory information and influence the prediction resutls efficiently. The network seemed to predict future trajctory still mainly based the past trajectory line without considering how the corresponding context, e.g. at zebra crossing, could indicate the future trajectory. We also foud it hard hard to converge during training when concatenate context after x,y (and/or delta x,y) for RNN training. 

Generally, the results are not much better than learning from the x, y features on RNN alone. However, a neural network prediction would still slightly outperforme the prediction solely based on linear regression of trajectories.

Following are two exampels of our training results: the blue line us the human past trajectory, orange dots are ground gruth trajectory at the next 6 timestamps with 15Hz frequency, green dots are prediction results from neuron networks.
<img src="https://raw.githubusercontent.com/celisun/autonomous_driving_human_trajectory_prediction/master/img/VRU_2.png" width="500"> <img src="https://raw.githubusercontent.com/celisun/autonomous_driving_human_trajectory_prediction/master/img/VRU_3.png" width="500">

## Dependencies
* python2
* tensorflow
* numpy
* pickle
* pandas
* cv2
