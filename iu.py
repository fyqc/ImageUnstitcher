import cv2
import numpy as np
import os

# 12/3/2024
# Rilla @ Kariya Park

'''
Instructions

1. Place the image to be split in the image_path.
2. Choose the appropriate mode for the image layout—vertical or horizontal. The default mode is 
vertical.
3. Run the program. A preview window will appear, allowing you to mark the approximate split
points:
    * Use the left mouse button to draw reference lines at the edges of the areas to be 
    separated.
4. Precision is not required when drawing the lines. The program automatically identifies the exact 
dividing lines within a 250-pixel range based on pixel differences.
5. Once all reference lines are drawn, close the preview window. The program will automatically 
process the image and save the separated parts in the same directory as the original image.
'''


class ImageSegmenter:
    def __init__(self, image_path, split_direction='vertical'):
        """
        Initialize the image segmentation tool

        Args:
            image_path (str): Path to the input image
            split_direction (str): Direction of splitting 'vertical' or 'horizontal'
        """
        # Validate split direction
        if split_direction not in ['vertical', 'horizontal']:
            raise ValueError(
                "Split direction must be 'vertical' or 'horizontal'")

        self.split_direction = split_direction

        # Directly read the original image
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)

        if self.original_image is None:
            raise ValueError(f"Could not read the image at {image_path}")

        # Get original dimensions
        height, width = self.original_image.shape[:2]

        # Resize for display if needed
        self.display_scale = 1.0
        max_display_dimension = 980

        if self.split_direction == 'vertical':
            # Resize vertically
            if height > max_display_dimension:
                self.display_scale = max_display_dimension / height
                display_width = int(width * self.display_scale)
                self.display_image = cv2.resize(
                    self.original_image,
                    (display_width, max_display_dimension),
                    interpolation=cv2.INTER_AREA
                )
            else:
                self.display_image = self.original_image.copy()
        else:
            # Resize horizontally
            if width > max_display_dimension:
                self.display_scale = max_display_dimension / width
                display_height = int(height * self.display_scale)
                self.display_image = cv2.resize(
                    self.original_image,
                    (max_display_dimension, display_height),
                    interpolation=cv2.INTER_AREA
                )
            else:
                self.display_image = self.original_image.copy()

        # Store drawing-related variables
        self.image_for_drawing = self.display_image.copy()
        self.lines = []

        # Create a window and set mouse callback
        cv2.namedWindow('Image Segmentation')
        cv2.setMouseCallback('Image Segmentation', self.draw_line)

    def draw_line(self, event, x, y, flags, param):
        """
        Mouse callback function for drawing lines

        Args:
            event: OpenCV mouse event
            x, y: Coordinates of mouse event
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.split_direction == 'vertical':
                # Convert display coordinates back to original image coordinates
                original_y = int(y / self.display_scale)
                self.lines.append(original_y)

                # Draw line on display image
                cv2.line(
                    self.image_for_drawing,
                    (0, y),
                    (self.image_for_drawing.shape[1], y),
                    (0, 255, 0),
                    2
                )
            else:
                # Horizontal splitting
                original_x = int(x / self.display_scale)
                self.lines.append(original_x)

                # Draw line on display image
                cv2.line(
                    self.image_for_drawing,
                    (x, 0),
                    (x, self.image_for_drawing.shape[0]),
                    (0, 255, 0),
                    2
                )

            cv2.imshow('Image Segmentation', self.image_for_drawing)

    def find_precise_boundaries(self, manual_lines):
        """
        Find precise image boundaries using grayscale analysis
        """
        # Read full-resolution image
        full_image = self.original_image

        # Convert to grayscale based on split direction
        if self.split_direction == 'vertical':
            gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        else:
            # transpose for horizontal
            gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY).T
            image_length = gray.shape[0]

        # Sort and add image boundaries
        manual_lines = sorted(manual_lines)
        print("Sorted Manual Lines:", manual_lines)

        if self.split_direction == 'vertical':
            # Create boundaries list
            boundaries = [0] + manual_lines + [gray.shape[0]]
            print("Initial Boundaries:", boundaries)
            search_axis = 1  # for vertical splitting
        else:
            boundaries = [0] + manual_lines + [image_length]
            print("Initial Boundaries:", boundaries)
            search_axis = 0  # for horizontal splitting

        precise_boundaries = []
        search_range = 250  # pixels to search around manual line

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            print(f"Processing segment {i}: start={start}, end={end}")

            # Define search region with extended range
            search_top = max(0, start - search_range)

            if self.split_direction == 'vertical':
                search_bottom = min(gray.shape[0], end + search_range)
                region = gray[search_top:search_bottom]
            else:
                search_bottom = min(image_length, end + search_range)
                region = gray[:, search_top:search_bottom].T

            # If this is the first or last segment, use the boundary directly
            if i == 0 or i == len(boundaries) - 2:
                precise_boundary = start
            else:
                # Compute line-wise intensity gradient
                line_gradients = np.abs(np.diff(region.mean(axis=search_axis)))

                # Find significant edges
                gradient_threshold = line_gradients.mean() + line_gradients.std()
                edge_candidates = np.where(
                    line_gradients > gradient_threshold)[0]

                if len(edge_candidates) > 0:
                    # Prefer edges near the manual line
                    relative_manual_line = manual_lines[i-1] - search_top
                    best_edge = edge_candidates[np.argmin(
                        np.abs(edge_candidates - relative_manual_line))]
                    precise_boundary = search_top + best_edge
                else:
                    # Fallback to manual line
                    precise_boundary = start

            precise_boundaries.append(precise_boundary)

        # Always add the full image boundary
        if self.split_direction == 'vertical':
            precise_boundaries.append(gray.shape[0])
        else:
            precise_boundaries.append(image_length)

        print("Final Precise Boundaries:", precise_boundaries)
        return precise_boundaries

    def extract_and_save_images(self, boundaries):
        """
        Extract and save image segments
        """
        try:
            # Read full-resolution image
            full_image = cv2.imread(self.image_path)

            # Additional check for image reading
            if full_image is None:
                print(f"Error: Could not read image {self.image_path}")
                return []

            # Ensure boundaries are unique and sorted
            boundaries = sorted(set(boundaries))
            print("Processed Boundaries:", boundaries)

            # Extract and save each image segment
            extracted_images = []
            for i in range(len(boundaries) - 1):
                # Determine segment based on split direction
                if self.split_direction == 'vertical':
                    top, bottom = boundaries[i], boundaries[i+1]
                    print(f"Segment {i}: top={top}, bottom={bottom}")

                    # Ensure valid segment
                    if bottom - top <= 0:
                        print(f"Skipping invalid segment {
                              i}: top={top}, bottom={bottom}")
                        continue
                    try:
                        # Extract the image segment
                        segment = full_image[top:bottom, :]

                        print(f"Segment {i} dimensions: {segment.shape}")

                        extracted_images.append(segment)

                        # Save the segment
                        base_filename = os.path.splitext(
                            os.path.basename(self.image_path))[0]
                        output_filename = f"{base_filename}_{
                            i + 1}_vertical.jpg"

                        # Enhanced image writing with additional checks
                        if not cv2.imwrite(output_filename, segment):
                            print(f"Error: Failed to write {output_filename}")
                        else:
                            print(f"Successfully saved {output_filename}")

                    except Exception as e:
                        print(f"Error processing segment {i}: {e}")

                else:
                    # Horizontal splitting
                    left, right = boundaries[i], boundaries[i+1]
                    print(f"Segment {i}: left={left}, right={right}")

                    # Ensure valid segment
                    if right - left <= 0:
                        print(f"Skipping invalid segment {
                              i}: left={left}, right={right}")
                        continue

                    # Extract the image segment
                    segment = full_image[:, left:right]

                    # Skip invalid segments
                    if segment.size == 0:
                        print(f"Skipping empty segment {i}")
                        continue

                    extracted_images.append(segment)

                    # Save the segment
                    base_filename = os.path.splitext(
                        os.path.basename(self.image_path))[0]
                    output_filename = f"{base_filename}_{i + 1}_horizontal.jpg"

                    if not cv2.imwrite(output_filename, segment):
                        print(f"Error: Failed to write {output_filename}")
                    else:
                        print(f"Successfully saved {output_filename}")

            return extracted_images

        except Exception as e:
            print(f"Critical error in image processing: {e}")
            return []

    def run_segmentation(self):
        """
        Run interactive segmentation process
        """
        # Display the image
        cv2.imshow('Image Segmentation', self.image_for_drawing)

        # Wait for window to be closed
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find precise boundaries based on manual lines
        precise_boundaries = self.find_precise_boundaries(self.lines)

        # Extract and save images
        self.extract_and_save_images(precise_boundaries)

        return precise_boundaries


def main(image_path, split_direction='vertical'):
    """
    Main function to run interactive image segmentation

    Args:
        image_path (str): Path to the input image
        split_direction (str): Direction of splitting 'vertical' or 'horizontal'
    """
    # Create segmenter
    segmenter = ImageSegmenter(image_path, split_direction)

    # Run segmentation
    segmenter.run_segmentation()


# Example usage
if __name__ == "__main__":
    image_path = "test.jpeg"  # Replace with your image path

    # # Use vertical splitting (default)
    main(image_path, 'vertical')

    # Or use horizontal splitting
    # main(image_path, 'horizontal')
