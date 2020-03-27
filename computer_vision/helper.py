import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt


def show_image(img, name=None):
    """Shows an image until any key is pressed"""
    if name is None:
        name = 'image'

    cv2.imshow(str(name), img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows


def create_blank_image(width, height, grayscale=True, include_gray_channel=False):
    """Creates a blank image as a Numpy array that can be read by OpenCV function"""
    if grayscale:
        if include_gray_channel:
            return np.zeros((height, width, 1), np.uint8)
        else:
            return np.zeros((height, width), np.uint8)
    else:
        return np.zeros((height, width, 3), np.uint8)


def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def convert_when_colour(colour, img):
    """Dynamically converts an image to colour if the input colour is a tuple and the image is grayscale."""
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def display_lines(in_img, lines, colour=(0, 0, 255)):
    """
    Draws lines on a copy of the image in the given colour.

    Args:
        in_img (np.array): Input image (will be copied).
        lines (np.array): Array of lines to draw.
        colour (int|tuple): Integer if grayscale, BGR tuple for colour.

    Returns:
        np.array: Image with teh lines drawn on
    """
    img = convert_when_colour(colour, in_img.copy())

    height, width = img.shape[:2]
    l = height * width  # Length of the line (plot across the entire image)

    for i in range(len(lines)):  # Plot all lines
        for rho, theta in lines[i]:  # Data for each line
            p1, p2 = get_points_on_line(rho, theta, l)
            cv2.line(img, p1, p2, colour, 1)  # Draw line on image

    show_image(img)
    return img


def display_points(in_img, points, radius=5, colour=(0, 0, 255), show=False):
    """
    Draws circular points on an image.

    Args:
        in_img (np.array): Input image (will be copied).
        points (np.array): List of points to plot on the image.
        radius (int): Radius of the circle to draw.
        colour (int|tuple): Integer if grayscale, BGR tuple for colour.
        show (bool): If True, will display the image.

    Returns:
        np.array: Image with the circular points drawn.
    """
    img = convert_when_colour(colour, in_img.copy())

    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)

    if show:
        show_image(img)
    return img


def display_rects(in_img, rects, colour=(0, 0, 255), show=False):
    """
    Displays rectangles on the image.

    Args:
        in_img (np.array): Input image (will be copied).
        rects (np.array): List of rectangles (top left and bottom right corners) to plot on the image.
        colour (int|tuple): Integer if grayscale, BGR tuple for colour.
        show (bool): If True, will display the image.

    Returns:
        np.array: Image with the rectangles drawn.
    """
    img = convert_when_colour(colour, in_img.copy())
    for rect in rects:
        img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
    if show:
        show_image(img)
    return img


def display_contours(in_img, contours, colour=(0, 0, 255), thickness=2, show=False):
    """
    Displays contours on the image.

    Args:
        in_img (np.array): Input image (will be copied).
        contours (np.array): Array of contours to display on the image.
        colour (int|tuple): Integer if grayscale, BGR tuple for colour.
        thickness (int): Thickness in pixesl to display the contours.
        show (bool): If True, will display the image.

    Returns:

    """
    img = convert_when_colour(colour, in_img.copy())
    img = cv2.drawContours(img, contours, -1, colour, thickness)
    if show:
        show_image(img)


def sum_pixels_in_rects(img, rects):
    """Sums the total value of all the pixels contained in the list of all the rectangles."""
    return sum([np.sum(img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]) for rect in rects])


def plot_many_images(images, titles, rows=1, columns=2):
    """Plots each image in a given list in a grid format using Matplotlib."""
    for i, image in enumerate(images):
        plt.subplot(rows, columns, i+1)
        plt.imshow(image, 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # Hide tick marks
    plt.show()


def center_of_bbox(x, y, width, height):
    """Returns the point in the center of the box described by the top left point and width/height.

    Args:
        x: X co-ordinate of top left point.
        y: Y co-ordinate of top left point.
        width: Width of the box.
        height: Height of the box.

    Returns:
        tuple: X and Y co-ordinates of the center of the rectangle.
    """
    return x + (width / 2), y + (height / 2)


def find_largest_polygon(img, n=4, threshold=0.01):
    """
    Finds the largest contour in the image that approximates a polygon with `n` sides.

    Args:
        img (np.array): Input image to find the polygon in.
        n (int): Number of (approximate) sides the polygon is required to have.
        threshold (float): Percentage of the perimeter determining if sides are smoothed in the polygon or not.
    Returns:
        cv2.Contour: The contour that encloses the largest area, optionally filtered based on approximated number of sides.
    """
    _, contours, hier = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area

    if n > 0:  # Get the first contour that has the desired number of sides
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            sides = cv2.approxPolyDP(contour, threshold * perimeter, True)  # Approximate the polygon based on a threshold
            if len(sides) == n:
                return contour
    else:  # Otherwise simply return the largest
        return contours[0]


def find_four_corners_inner(img):
    """
    Finds the child contours of the feature in the image that has the largest area. Uses RETR_CCOMP.
    See: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
    Uses `get_four_corners` to find the four inner corners of the largest polygon in the image.

    Args:
        img (np.array): Input image to find the contours in.

    Returns:
        np.array: Array of contours that are immediate children of the largest contour.
    """
    _, contours, hier = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    largest_i, largest_area = max(enumerate([cv2.contourArea(c) for c in contours]), key=operator.itemgetter(1))

    def find_children(idx):
        children, corners = [], []
        while idx > -1:
            children.append(contours[idx])
            corners += [[c] for c in get_four_corners(contours[idx])]
            idx = hier[0][idx][0]
        return children, corners

    children_, corners_ = find_children(hier[0][largest_i][2])
    bbox = get_four_corners(corners_)
    area = cv2.contourArea(np.array([[c] for c in bbox]))
    return bbox, area, contours[largest_i], largest_area


def get_four_corners(polygon):
    """
    Gets the four most extreme corners of a polygon described a contour. Uses the logic that:

    * Bottom-right point has the largest (x + y) value
    * Top-left has point smallest (x + y) value
    * Bottom-left point has smallest (x - y) value
    * Top-right point has largest (x - y) value
    """

    # Use of operator.itemgetter here allows us to get the index of the maximum or minimum points
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """
    Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
    connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
    """

    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    img, bbox = get_bbox_from_seed(inp_img, seed_point, False)

    # Return the image as well as an array of points describing the bounding of the remaining feature
    return img, bbox, seed_point


def get_bbox_from_seed(inp_img, seed, bbox=True):
    """
    Using a seed point for a pixel connected fill, return a bounding box (or extreme point quadrilateral) for that
    connected pixel structure.

    Args:
        inp_img (np.array): Input image that the seed image should be extracted from
        seed (tuple): X, Y co-ordinate that has a pixel connected structure from which you want to extract the bounding
            box.
        bbox (bool): If True will mark the bounding box enclosing the whole seed feature. Otherwise will return a
            quadrilateral made of the extreme corners of that feature.

    Returns:
        tuple: First element is an `np.array` of the image with only the seed structure left in place.
        Second element is  an `np.array` that contains the top left, top right, bottom right and bottom left co-ordinates
        of a quadrilateral marking either the bounding box of the feature or the extreme point quadrilateral of that
        feature.
    """

    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

    # Colour everything grey
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    # Highlight the main feature
    if all([p is not None for p in seed]):
        cv2.floodFill(img, mask, seed, 255)

    top_left = [width, height]
    top_right = [0, height]
    bottom_left = [width, 0]
    bottom_right = [0, 0]

    top = height
    bottom = 0
    left = width
    right = 0

    # Loop again to fill in all the gray (temporary highlighted) areas, leaving just the grid
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

            # If it is a highlighted point, use it to determine the bounding position
            if img.item(y, x) == 255:
                if bbox:
                    if y < top:
                        top = y

                    if y > bottom:
                        bottom = y

                    if x < left:
                        left = x

                    if x > right:
                        right = x
                else:
                    if x + y < sum(top_left):
                        top_left = [x, y]

                    if x + y > sum(bottom_right):
                        bottom_right = [x, y]

                    if x - y > top_right[0] - top_right[1]:
                        top_right = [x, y]

                    if x - y < bottom_left[0] - bottom_left[1]:
                        bottom_left = [x, y]

    if bbox:
        top_left = [left, top]
        bottom_right = [right, bottom]
        rect = [top_left, bottom_right]  # Only need the top left and bottom right points
    else:
        rect = [top_left, top_right, bottom_right, bottom_left]

    return img, np.array(rect, dtype='float32')


def get_points_on_line(rho, theta, length=1000):
    """Gets two points (x1, y1), (x2, y2) on a line defined in the normal form by rho and theta."""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + (length * (-b)))
    y1 = int(y0 + (length * a))
    x2 = int(x0 - (length * (-b)))
    y2 = int(y0 - (length * a))
    return (x1, y1), (x2, y2)


def merge_related_lines(img, lines, threshold=64):
    """
    Based on all the lines input to the function, takes an average of any adjacent lines, effectively merging them.

    Args:
        img: Image the lines are from.
        lines (np.array): Array of arrays representing the lines in the image. Each line should have two elements, the
             first being rho, the second being theta.
        threshold (int): Distance to use to calculate if lines count as being adjacent.

    Returns:
        np.array: Array of only the merged lines.
    """
    height, width = img.shape[:2]

    def check_horiz_or_vert(theta, angular_threshold=30):
        """Checks if a line is approximately horizontal or vertical. Threshold in degrees."""

        # Checks horizontal
        if np.pi * ((90 - angular_threshold) / 180) < theta < np.pi * ((90 + angular_threshold) / 180):
            return True

        # Checks vertical
        if theta < np.pi * (angular_threshold / 180) or theta > np.pi * ((180 - angular_threshold) / 180):
            return True

        return False

    for current in lines:  # Plot all lines
        for rho1, theta1 in current:  # Data for each line

            # If we reach a line with these impossible values, skip them - they will only exist if we set them
            if rho1 == 0 and theta1 == -100:
                continue

            if not check_horiz_or_vert(theta1):
                current[0][0] = 0
                current[0][1] = -100
                continue

            low1, high1 = get_points_on_line(rho1, theta1,  (height + width) / 2)

            # Loop over all the lines again so we can compare them
            for next_line in lines:
                for rho2, theta2 in next_line:
                    # Skip if it's the same line as before, we won't merge those
                    if rho1 == rho2 and theta1 == theta2:
                        continue

                    low2, high2 = get_points_on_line(rho2, theta2, (height + width) / 2)

                    # If the end points are close together (within the threshold) we can establish the lines are adjacent
                    distances = [
                        low2[0] - low1[0],
                        low2[1] - low1[1],
                        high2[0] - high1[0],
                        high2[1] - high1[1]
                    ]

                    if (sum([x ** 2 for x in distances]) ** 0.5) < threshold:
                        # Merge the lines by altering rho and theta of the original line if they are close enough
                        current[0][0] = (rho1 + rho2) / 2
                        current[0][1] = (theta1 + theta2) / 2

                        # Set the merged line t o an impossible value so we ignore it in the outer loop
                        next_line[0][0] = 0
                        next_line[0][1] = -100

    # Remove invalid lines
    lines = list(filter(lambda x: not(x[0][0] == 0 and x[0][1] == -100), lines))
    return lines


def filter_related_lines(lines, thold=10, a_thold=1):
    """
    Filters out lines that are close to each other and that are more than 1 degree away from vertical or horizontal.
    """
    lines = np.array(sorted(lines, key=lambda x: x[0][0]))  # Sort lines by Rho

    pos_hori = -thold - 1
    pos_vert = -thold - 1

    out = []
    for line in lines:
        for rho, theta in line:
            if np.sin(theta) > 0.5:  # Horizontal lines
                if (rho - pos_hori) > thold and np.pi * ((90 - a_thold) / 180) < theta < np.pi * ((90 + a_thold) / 180):
                    pos_hori = rho
                    out.append(line)
            else:  # Vertical
                if (rho - pos_vert) > thold and (theta < np.pi * (a_thold / 180) or theta > np.pi * ((180 - a_thold)/180)):
                    pos_vert = rho
                    out.append(line)
    return out


def find_extreme_lines(img, lines):
    """
    Filters the array of input lines to only keep the top, bottom, left and right most edges.
    """
    height, width = img.shape[:2]

    # Set unrealistically extreme values that will be overwritten
    top_edge = [height, -100]
    bottom_edge = [0, -100]
    left_edge = [0, -100]
    right_edge = [0, -100]

    right_x_intercept = 0
    left_x_intercept = width

    for line in lines:
        for rho, theta in line:
            x_intercept = rho / np.cos(theta)  # Where the line crosses the X axis

            # If the line is (approximately) vertical
            if np.pi * (100/180) > theta > np.pi * (80/180):

                # Set the top and bottom edges based on rho
                if rho < top_edge[0]:
                    top_edge = line[0]

                if rho > bottom_edge[0]:
                    bottom_edge = line[0]
            elif theta < np.pi * (10/180) or theta > np.pi * (170/180):  # Or if approximately horizontal
                if x_intercept > right_x_intercept:
                    right_edge = line[0]
                    right_x_intercept = x_intercept

                if x_intercept < left_x_intercept:
                    left_edge = line[0]
                    left_x_intercept = x_intercept

    return [[left_edge], [right_edge], [top_edge], [bottom_edge]]


def intersection(line1, line2):
    """
    Returns a point (x, y) describing the intersection between the two input lines.
    Method: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    """
    p1, p2 = get_points_on_line(line1[0][0], line1[0][1])
    p3, p4 = get_points_on_line(line2[0][0], line2[0][1])

    def det(v1, v2):
        """Returns the determinant of a 2x2 matrix defined by two two-element vectors, v1 and v2."""
        return (v1[0] * v2[1]) - (v1[1] * v2[0])

    def numerator(i):
        return det([det(p1, p2), det([p1[i], 1], [p2[i], 1])], [det(p3, p4), det([p3[i], 1], [p4[i], 1])])

    x_num = numerator(0)
    y_num = numerator(1)
    denom = det([det([p1[0], 1], [p2[0], 1]),
                 det([p1[1], 1], [p2[1], 1])],
                [det([p3[0], 1], [p4[0], 1]),
                 det([p3[1], 1], [p4[1], 1])])
    x = x_num / denom
    y = y_num / denom
    return x, y


def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop(img, top_left, bottom_right):
    """Crops an image based on top left and bottom right points."""
    return img[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]


def crop_and_warp(img, crop_rect, square=False):
    """
    Crops an image and skews the aspect based on the bounding box defined by the input parameters.

    Args:
        img (np.array): Original image to crop
        crop_rect(np.array): Array representing a rectangle, with order: top_left, top_right, bottom_right, bottom_left
        square (bool): If True, will force the output image to be square.

    Returns:
        np.array: Cropped and skewed image.
    """
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    max_height = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left)
    ])

    max_width = max([
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    if square:
        max_length = max([max_width, max_height])
        max_width = max_length
        max_height = max_length

    # Define a square of side `max_length` to map the image on to
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    output = cv2.warpPerspective(img, m, (int(max_width), int(max_height)))

    return output, [top_left, top_right, bottom_right, bottom_left], m


def crop_image_from_edges(img, left_edge, right_edge, top_edge, bottom_edge, square=False):
    """
    Crops the image using the input lines to define a bounding box. The intersections of these four lines are obtained
    and then the image is cropped and the aspect un-skewed so that the edges are vertical and horizontal.


    Args:
        img (np.array): Original image to crop
        left_edge (np.array): Array defining the line for the left edge to crop
        right_edge (np.array): Array defining the line for the right edge to crop
        top_edge (np.array): Array defining the line for the top edge to crop
        bottom_edge (np.array): Array defining the line for the bottom edge to crop
        square (bool): If True, will force the output image to be square.

    Returns:
        np.array: Cropped and skewed image.
    """

    top_left = intersection(left_edge, top_edge)
    top_right = intersection(right_edge, top_edge)
    bottom_left = intersection(left_edge, bottom_edge)
    bottom_right = intersection(right_edge, bottom_edge)

    return crop_and_warp(img, [top_left, top_right, bottom_right, bottom_left], square)


def grid_intersections(lines):
    """Finds all intersection points between lines."""
    all_intersects = []
    for line1 in lines:
        for line2 in lines:
            if line1[0][0] != line2[0][0] and line1[0][1] != line2[0][1]:  # Don't compare with itself
                intersect = intersection(line1, line2)
                if intersect not in all_intersects:
                    all_intersects.append(intersect)
    return all_intersects


def grid_to_bbox(grid, padding=0):
    """Converts a 2D grid (from `extract_sudoku_grid`) into a list a rectangles describing the bounding boxes."""
    rects = []
    for i in range(9):
        for j in range(9):
            p1 = grid[i][j]  # Top left corner of a bounding box
            p2 = grid[i+1][j+1]  # Bottom right corner of the box
            rect = ((p1[0] + padding, p1[1] + padding), (p2[0] - padding, p2[1] - padding))
            rects.append(rect)
    return rects


def limit_scale(img, max_x, max_y):
    """
    Scales down a large image based on a minimum width or height, keeping the original aspect ratio. Images within the
    bounds of `max_x` and `max_y` are left untouched.
    """
    height, width = img.shape[:2]
    ratio = 1

    if height <= max_y and width <= max_x:
        pass
    elif height > width:
        ratio = max_y / height
    elif width > height:
        ratio = max_x / width

    return cv2.resize(img, (int(ratio * width), int(ratio * height)))


def adjust_contrast_brightness(img, contrast=1, brightness=0):
    """
    Adjusts the contrast and brightness of an image according to parameters alpha (contrast) and beta (brightness).

    Args:
        img (np.array): Input image
        contrast (float): Alpha > 1 will increase the contrast, less than 1 will reduce the contrast. Alpha should be > 0.
        brightness (int): Amount between -255 and 255 to increase the brightness.

    Returns:
        np.array: Image with the brightness and contrast changed.
    """
    height, width = img.shape[:2]
    new_img = create_blank_image(width, height)
    return cv2.convertScaleAbs(img, new_img, contrast, brightness)
