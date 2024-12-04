# ImageUnstitcher
*A simple yet powerful Python tool for splitting combined images into their original parts.*

This tool allows users to precisely restore individual images from a vertically combined composite image. With features like manual line-drawing assistance and automated edge detection, ImageSplitter ensures accurate separation of slices. Perfect for photographers, designers, and anyone working with composite images.

Features:

- **Interactive Line Drawing**: Easily draw guides to mark approximate borders.  
- **Smart Edge Detection**: Automatically refines borders for precise image slicing.  
- **User-Friendly Interface**: Resize large images for easy handling.  
- **Batch Output**: Quickly save all individual slices as separate files.  

***

## Instructions

1. Place the image to be split in the `image_path`.
2. Choose the appropriate mode for the image layout—vertical or horizontal. The default mode is vertical.
3. Run the program. A preview window will appear, allowing you to mark the approximate split points:
  - Use the **left mouse button** to draw reference lines at the edges of the areas to be separated.
4. Precision is not required when drawing the lines. The program automatically identifies the exact dividing lines within a 250-pixel range based on pixel differences.
5. Once all reference lines are drawn, close the preview window. The program will automatically process the image and save the separated parts in the same directory as the original image.
