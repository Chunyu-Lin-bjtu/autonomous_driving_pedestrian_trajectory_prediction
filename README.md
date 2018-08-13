# Autonomous Driving Human Trajectory Prediction

This is a pilot project, our purpose is to train models to predict human trajectories given the past trajectory in developing autonomous driving. 

Our self-created datasets (default 6000 sets of map and trajectory, which can be made into >10k training samples) simulated human walking cross the crossroad through the zebra crossing. 


### VRU approaches specification

the challenge of this project is to combine context information with the trajectory information (x, y coordinate here) in predicting the future trajectory with RNN. Different approahces we've tried differ in their ways of encoding context info and combining with the trajecotory featuers. 


* x/y: the x, y coordniate feature on the original context

* delta x/y: the relative x, y coordinate of each point relative to its previous x, y position on the trajectory

* context: the original context map (default 1280 * 1280)

* context_patch: a patch of the orginal context cropped for each data point on a trajectory as the center. for a training sample, the shape is usually (None, sequence_length, patch_size * patch_size)



<pre>
	 x/y,   delta x/y,  context, context patch   CNN      RNN   fc

vru:	 -	    - 	      -		            context   all     -
vru2:               -                     -         patch     all     -
vru3:  	 -	   (-)	      -                     context   x/y (or x/y + delta x/y)    -
vru_s:   	    -	      -			              all     -

'-' means in use.
'all' in RNN means the x,y or delta x, y featuers with the output vector of CNN  if appicable.
</pre>

We lalso tried to use a autoencoder vae to encode the context patches and combine with the x/y feature.The following is a visualization of trained vae model for context images.



