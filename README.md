# TMM
Template Match with Binary Mask


We introduce an implementation of Template Match Algorithm intended to solve the problems of finding a Template image, where arbitrary regions are occluded with binary Mask, within a second large Source image.

It has the ability to find as much as possible occurrences of the Masked Template image within the Source image, that match (are similar) only on the areas outside the occluded binary mask.

For example, suppose we have a Source image that contains rectangles with text and other shapes, and a Template image with text on it, we want to find all the rectangles regardless the text on it.

In this case, the text regions of the Template are masked out, and a matching against the Source Image is performed. The result should be all the rectangles on the Source Image (Fig. 1).



