# Autonomous Driving Human Trajectory Prediction

This is a pilot project, our purpose is to train models to predict human trajectories given the past trajectory in developing autonomous driving. 

Our self-created datasets (default 6000 sets of map and trajectory, which can be made into >10k training samples) simulated human walking cross the crossroad through the zebra crossing. 


#### VRU approaches specification

the challenge of this project is to combine context information with the trajectory information (x, y coordinate here) in predicting the future trajectory with RNN. Different approahces we've tried differ in their ways of encoding context info and combining with the trajecotory featuers. 


* x,y: the x, y coordniate feature on the original context

* delta x,y: the relative x, y coordinate of each point relative to its previous x, y position on the trajectory

* context: the original context map (default 1280 * 1280)

* context_patch: a patch of the orginal context cropped for each data point on a trajectory as the center. for a training sample, the shape is usually (None, sequence_length, patch_size * patch_size)



<pre>
	 x,y,   delta x,y,  context, context patch   CNN      RNN   fc

vru:	 -	    - 	      -		            context   all     -
vru2:               -                     -         patch     all     -
vru3:  	 -	   (-)	      -                     context   x/y (or x,y + delta x,y)    -
vru_s:   	    -	      -			              all     -

'-' means in use.
'all' in RNN means the x,y or delta x, y featuers with the output vector of CNN  if appicable.
</pre>

#### VAE 
Aside from using a simpe RNN to encode the context image, we also tried to use an autoencoder vae to encode the context patches and then combine with the x/y feature to feed in RNN.The following is a visualization of a trained vae model for context images. The **left** blurried one is a vae reconstructed image compared with its immediate **right** image which is the orginal one.

<img src="https://raw.githubusercontent.com/celisun/autonomous_driving_human_trajectory_prediction/master/img/I_reconstructed0.png" width="500">


