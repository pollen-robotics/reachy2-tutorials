# reachy2-tutorials
Welcome to you, **new Reachy2 user** ! 

You've received your robot, you've got **lots of ideas** about what you'd like it to do, but you're feeling **helpless** and don't know how to go about it ?

Then you've come to the *right place*! 

Here you'll find **notebook tutorials** and **demo scripts** for Reachy2. 

### Notebooks
They will teach you **step-by-step** how to get to grips with Reachy and use the various libraries available to give it optimum control. The aim is to show how to manage **basic behaviors**, so that you can then create the more **complex** behaviors you wish to implement.

The notebooks present two libraries that are essential to Reachy's operation: 
- *ReachySDK*, for robot control,
- and *Pollen-Vision*, for object detection in the environment.

### Demo scripts 

They are the scripts used for our tests and videos. You can learn how to use the *ReachySDK* and how to implement models from the [*HuggingFace Hub*](https://huggingface.co/models). 

## Prerequisites

In both cases, you need to install **reachy2-sdk** to control the robot. 

You can find it on [GitHub](https://github.com/pollen-robotics/reachy2-sdk), with the installation procedures explained on the README. You can find [Getting Started notebooks](https://github.com/pollen-robotics/reachy2-sdk/tree/develop/src/examples) to show you the basics, so do not hesitate to follow them before starting the tutorials !

### Notebooks

As the tutorials are in notebook form, you'll need the *jupyter extension* in your code editor and the library *pykernel* (<code>pip install pykernel</code>).

For the tutorial n°3, you'll need **pollen-vision module**, available on [GitHub](https://github.com/pollen-robotics/pollen-vision) with the installation procedures explained on the README. 

### Demo scripts 

You can install the needed libraries by executing in a virtual environment : 
``` 
pip install -r scripts/requirements_scripts.txt
``` 

## Organization
### Notebooks
3 tutorials are available for now : 
1. Reachy's awakening (with SDK only)
2. Reachy the mime (with SDK only)
3. Reachy the greengrocer (with SDK & Pollen-Vision)

We recommend you carry them out **in this order**, from the most basic to the most complex, so you can become a **Reachy control expert** without even realizing it. 

### Demo scripts 

1 script is available for now : it's the fan tracking demo. It uses a face tracking system, to fan you wherever you want to go.

## Updates 

We'll be adding **new tutorials and scripts** to this library as we go along, with new functionalities and new libraries. **So stay tuned !**
