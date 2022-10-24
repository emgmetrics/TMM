# TMM
Template Match with Binary Mask


We introduce an implementation of Template Match Algorithm intended to solve the problems of finding a Template image, where arbitrary regions are occluded with binary Mask, within a second large Source image.

It has the ability to find as much as possible occurrences of the Masked Template image within the Source image, that match (are similar) only on the areas outside the occluded binary mask.

For example, suppose we have a Source image that contains rectangles with text and other shapes, and a Template image with text on it, we want to find all the rectangles regardless the text on it.

In this case, the text regions of the Template are masked out, and a matching against the Source Image is performed. The result should be all the rectangles on the Source Image.


The core of this Template Matching algorithm is based on the well known Feature matching theory of image processing. The matching method is based on Template Descriptor correlation rather than Image correlation currently used by most matching applications.

The Template Descriptor is a signature that represents the Template image by a vector of N-bytes, and can be changed in such a simple way that it can represent the Masked Template Image.

The correlation of the Template descriptor is performed against a set of Template patches Descriptors within the Source Image computed in selected feature points.

The presented method allows applications to perform efficiently matching tasks with arbitrary mask regions.

Follow this [link]('https://github.com/emgmetrics/TMM/tree/main/Doc/Finding Templates Images with Masked Regions.pdf') for a detail description of the Algorithm.

