# Incorporating a-priori information in deep learning models for quantitative susceptibility mapping via adaptive convolution

This repository provides the official implementation of the Adaptive U-Net for QSM-based field-to-susceptibility inversion leveraging a-priori information.

**Paper:** [Frontiers in Neuroscience](https://doi.org/10.3389/fnins.2024.1366165 )


**The code will be available soon.**


# Adaptive Convolution

![](Figures/Figure1.png)

Figure 1: Schematic overview of adaptive convolution, adaptive layers and the used 3D U-Net architecture. (A) The filter manifold compresses the relationship between the side information $\vec{s}$ and the changes in the image onto a low dimensional filter manifold in the high dimensional filter weight space. By changing the side information, the filter kernel values itself change, sweeping along the smooth filter manifold. (B) Adaptive convolutional layers are built from the Filter Manifold Network (FMN) consisting of 4 fully connected linear layers that compute the weights *w* of the respective convolution operation of the input feature maps *X*, yielding the output feature maps *Y* (blue block). (C) The 3D U-Net is composed of an encoder (orange blocks) and a decoder (turquoise blocks) with the adaptive convolution layer (dashed red arrow) included in the first encoding stage (blue block).

# Results

# Instructions 


Please cite this paper when using the Adaptive U-Net: 

    Graf S, Wohlgemuth WA and Deistung A (2024). Incorporating a-priori information in deep learning models 
    for quantitative susceptibility mapping via adaptive convolution. Front. Neurosci. 18:1366165. doi: 10.3389/fnins.2024.1366165       
