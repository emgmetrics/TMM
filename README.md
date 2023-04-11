# TMM
## Template Match with Binary Mask - Overview


This Project introduces an implementation of Template Match Algorithm intended to solve the problems of finding a Template image, where arbitrary regions are occluded with binary Mask, within a second large Source image.

It has the ability to find as much as possible occurrences of the Masked Template image within the Source image, that match (are similar) only on the areas outside the occluded binary mask.

For example, suppose we have a Source image that contains rectangles with text and other shapes, and a Template image with text on it, we want to find all the rectangles regardless the text on it.

In this case, the text regions of the Template are masked out, and a matching against the Source Image is performed. The result should be all the rectangles on the Source Image.


The core of this Template Matching algorithm is based on the well known Feature matching theory of image processing. The matching method is based on Template Descriptor correlation rather than Image correlation currently used by most matching applications.

The Template Descriptor is a signature that represents the Template image by a vector of N-bytes, and can be changed in such a simple way that it can represent the Masked Template Image.

The correlation of the Template descriptor is performed against a set of Template patches Descriptors within the Source Image computed in selected feature points.

The presented method allows applications to perform efficiently matching tasks with arbitrary mask regions.

Follow this [link](https://github.com/emgmetrics/TMM/tree/main/Doc) for a detail description of the Algorithm on the "Finding Templates Images with Masked Regions.pdf" document.

## CONFIGURATION MANAGEMENT 

### Tools Requirements
- Microsoft Windows 64bit OS.
- Microsoft Visual Studio 2019.
- OpenCV 4.6.0.

### Pre-Build Requirements
- Install OpenCV 4.6.0.
- Build OpenCV.
- Add Environment Vairable 'opencv' pointing to the location of the OpenCV installation (ex. opencv=D:\opencv).


### Build Procedures
- Using Microsoft VS 2019 Open TMM.sln located on the TMM folder.
- Select Release Configuration and x64 Platform.
- On the VS IDE Solution Explorer enter the Solution Properties window.
- On the 'TMM Property Pages' window go to the Debugging page and locate the Command Arguments field.
- Update Command Arguments field with the location of one of the matching tests folders located in the Tests folder
- Build an Run the application.


### Runnig Specifications from command prompt
- Change to the folder whre the TMM.exe file is located (ex. D:\TMM\x64\Release).
- Type TMM.exe '<folder-name-of-one-of-the-tests-located-under-Tests-folder>' 1.
- A window with name Mask should open
- Mark the areas to mask from the Template image by using the mouse, and press Enter.
- Five windows shall open, including the Matching and key points of the Template image and the Source image.
  
