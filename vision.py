import math
import cv2
import numpy as np
from constants import *

def pre_process(img, is_thin = True):
    height, width, _ = img.shape

    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(IMAGE_SCALING*height),int(IMAGE_SCALING*width)

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = img[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height)) 
    gray = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2GRAY)         
        
    # possible future work here, thin lines benefit from this blur/divide code
    # while thick lines can sometimes be interpreted as larger shapes,
    # may want to an initial test to see which option interprests the board better,
    # then uses that is_thin in the future
    if is_thin:
        blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    else:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
    inverse = (255 - thresh)    
    thin = cv2.ximgproc.thinning(inverse)
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thin, kernel, iterations = 2)
    
    return resized_cropped, gray, gray, thresh, dilated

def get_points_sorted_by_distance(points, center_of_mass):
    distances = []
    
    for p in points:
        distances.append(get_distance(p, center_of_mass))
        
    arr1 = np.array(distances)
    arr2 = np.array(points)    
    indexes = arr1.argsort()
    
    return arr2[indexes]
    
def get_points_sorted_by_angle(points, center_of_mass):
    angles = []

    for p in points:
        angle = get_angle(center_of_mass, p)
        angles.append(angle)
        
    arr1 = np.array(angles)
    arr2 = np.array(points)    
    indexes = arr1.argsort()
    points_by_angle = arr2[indexes[::-1]]
    
    return points_by_angle

def get_rotation(img):   
    # attempt to determine how far off of orthogonality the currently viewed grid is
    # so that it can be rotated back into true (this may not be strictly necessary, 
    # maybe the cell quadrilaterals can be interpreted without a rotation?) 
    contour_points = get_board_contour_points(img)
    center_of_mass = get_center_of_mass(contour_points)
    points = get_points_sorted_by_distance(contour_points, center_of_mass)
    
    # there should be 4 interior intersection points and 8 external external endpoints on the grid
    if len(points) != 12:
        raise Exception(f'Failed to detect gameboard, {len(points)} out of expected 12 contour points were found')

    interior_points_by_distance = points[0:4]
    interior_points_by_angle = get_points_sorted_by_angle(interior_points_by_distance, center_of_mass)

    # get the interior rotated bounding box (TODO but why, you already have the points by angle?)
    interior = np.array(interior_points_by_angle, dtype=np.int32)
    rect = cv2.minAreaRect(interior)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    interior_rotation = get_angle(box[0], box[1])
    # offset the rotation by 90 degrees and flip the angle so that the angle is like a clock with 0deg at 12 o'clock
    corrected_rotation = (270 - interior_rotation) % 360
        
    # if the angle is between 270 and 320, we don't want to fully rotate that much but rather by a 90deg offset
    if corrected_rotation < 320 and corrected_rotation >= 270:
        corrected_rotation = (corrected_rotation + 90) % 360
            
    return corrected_rotation

def get_end_points(img):    
    center_of_mass = get_center_of_largest_contour(img)
    contour_points = get_board_contour_points(img)
    points = get_points_sorted_by_distance(contour_points, center_of_mass)
    
    # get the outermost 8 points (the maximal termini of the gridlines)
    exterior_points_by_distance = points[-8:]
    exterior_points_by_angle = get_points_sorted_by_angle(exterior_points_by_distance, center_of_mass)
    
    return exterior_points_by_angle

def add_channel_to_grayscale(img):
    return np.stack((img,)*3, axis=-1)

def get_distance(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    
def sort_end_points_by_angle(end_points, center_of_mass):
    angles = []
    
    for p in end_points:
        angle = get_angle(center_of_mass, p)
        angles.append(angle)
       
    arr1 = np.array(angles)
    arr2 = np.array(end_points)    
    indexes = arr1.argsort()
    sorted_end_points = arr2[indexes]
    
    return sorted_end_points

def rotate_image(img, angle, background):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    
    return result
                           
def get_bounding_box(img):    
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    rectangle = cv2.minAreaRect(contours[0])
    _, _, rot = rectangle
    box = np.intp(cv2.boxPoints(rectangle))    
    
    return [box, rot] 

def get_center_of_mass(points):
    x = 0
    y = 0

    for p in points:
        x = x + p[0]
        y = y + p[1]

    center_x = int(x / len(points))
    center_y = int(y / len(points)) 
    
    return [center_x, center_y]

def get_center_of_largest_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_area_contour = None

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h;
       
        if area > max_area:
            max_area = area
            max_area_contour = contour

    x, y, w, h = cv2.boundingRect(max_area_contour)
    center_of_mass = get_center_of_mass([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    
    return center_of_mass

def get_angle(point1, point2):
    angle = math.atan2((point2[0] - point1[0]), (point2[1] - point1[1]))
    degrees = math.degrees(angle)
    degrees = (180 + degrees) % 360
    
    return degrees

"""
def get_rotational_loss(end_points):    
    vert_line_0 = get_angle(end_points[4], end_points[7])
    vert_line_1 = get_angle(end_points[3], end_points[0])

    if vert_line_0 > 180: vert_line_0 = 360 - vert_line_0
    if vert_line_1 > 180: vert_line_1 = 360 - vert_line_1
    
    horiz_line_0 = get_angle(end_points[6], end_points[1])
    horiz_line_1 = get_angle(end_points[5], end_points[2])
    
    loss = (vert_line_0)**2 + (vert_line_1)**2 + (90 - horiz_line_0)**2 + (90 - horiz_line_1)**2
    
    return loss
"""

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    
    if div == 0:
       raise Exception('Failed to find a line intersection point')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    return int(x), int(y)

def get_cells(end_points):
    # determine the intersection points, virtual corners of the gameboard
    # and finally derive the cell bounding box quadrilaterals, so that when the user
    # plays a turn, the vision component can determine where it was played on the board
    inter_top_left = line_intersection((end_points[4], end_points[7]), (end_points[6], end_points[1]))
    inter_top_right = line_intersection((end_points[3], end_points[0]), (end_points[6], end_points[1]))
    inter_bottom_left = line_intersection((end_points[4], end_points[7]), (end_points[5], end_points[2]))
    inter_bottom_right = line_intersection((end_points[3], end_points[0]), (end_points[5], end_points[2]))

    corner_top_left = (min(end_points[5][0], end_points[6][0]), min(end_points[7][1], end_points[0][1]))
    corner_top_right = (max(end_points[1][0], end_points[2][0]), min(end_points[7][1], end_points[0][1]))
    corner_bottom_left = (min(end_points[5][0], end_points[6][0]), max(end_points[4][1], end_points[3][1]))
    corner_bottom_right = (max(end_points[1][0], end_points[2][0]), max(end_points[4][1], end_points[3][1]))

    cells = []
    cells.append(np.array([corner_top_left, end_points[7], inter_top_left, end_points[6]]));
    cells.append(np.array([end_points[7], end_points[0], inter_top_right, inter_top_left]));
    cells.append(np.array([end_points[0], corner_top_right, end_points[1], inter_top_right]));
    cells.append(np.array([end_points[6], inter_top_left, inter_bottom_left, end_points[5]]));
    cells.append(np.array([inter_top_left, inter_top_right, inter_bottom_right, inter_bottom_left]));
    cells.append(np.array([inter_top_right, end_points[1], end_points[2], inter_bottom_right]));
    cells.append(np.array([end_points[5], inter_bottom_left, end_points[4], corner_bottom_left]));
    cells.append(np.array([inter_bottom_left, inter_bottom_right, end_points[3], end_points[4]]));
    cells.append(np.array([inter_bottom_right, end_points[2], corner_bottom_right, end_points[3]]));
    
    contours = []
    
    for cell in cells:
        contour = np.array(cell).reshape((-1,1,2)).astype(np.int32)
        contours.append(contour)
   
    return contours;

def get_board_contour_points(img):
    points = []
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
    approx = cv2.approxPolyDP(selected_contour, 0.0068 * cv2.arcLength(selected_contour, True), True)

    for candidate in approx:
        p = ([candidate[0][0], candidate[0][1]])
        is_close = False
        
        for q in points:
            if get_distance(p, q) < 25:  ### variabilize
                is_close = True
                break
        
        if is_close == False:
            points.append(p)
    
    return points

"""
def test_end_point_orthogonality(points):
    tolerance = 15
        
    horiz_line_0 = get_angle(points[0], points[3])
    horiz_line_1 = get_angle(points[7], points[4])
    vert_line_0 = get_angle(points[6], points[1])
    vert_line_1 = get_angle(points[5], points[2])

    print(180 - horiz_line_0)
    print(180 - horiz_line_1)
    print(90 - vert_line_0)
    print(90 - vert_line_1)

    if abs(180 - horiz_line_0) > tolerance \
        or abs(180 - horiz_line_1) > tolerance \
        or abs(90 - vert_line_0) > tolerance \
        or abs(90 - vert_line_1) > tolerance:
        raise Exception("Failed to detect gameboard, lines do not appear orthogonal")
"""

def read_cells(current_board, img, cells):  
    board = [EMPTY] * 9 
    
    # find the interior contours of the board, which correpond to the Xs and Os (hopefully!)
    thin = cv2.ximgproc.thinning(img)
    contours, _ = cv2.findContours(thin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    _, _, w, h = cv2.boundingRect(cells[0])
    
    #contours have to be a certain size, but no too big
    min_area = .05 * w * h
    max_area = .8 * w * h
    x_contours = []
    o_contours = []
    
    # look for a contour that's completely inside one of the cell contours
    for j, cell in enumerate(cells):
        #if current_board[j] == EMPTY:
        for contour in contours:
            content = read_cell(contour, cell, min_area, max_area)
            
            if content != None:
                board[j] = content
                
                if content == X:
                    x_contours.append(contour)
                else:
                    o_contours.append(contour)
        #else:
        #    board[j] = current_board[j]
      
    return board, x_contours, o_contours

def read_cell(contour, cell, min_area, max_area):
    # determine if this contour is completely inside a cell's contour
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    
    if area > min_area and area < max_area:  
        # all four corners of the contour's bounding box need to be inside the cell         
        a1 = cv2.pointPolygonTest(cell, (x,y), False) 
        a2 = cv2.pointPolygonTest(cell, (x+w,y), False) 
        a3 = cv2.pointPolygonTest(cell, (x,y+h), False) 
        a4 = cv2.pointPolygonTest(cell, (x+w,y+h), False)
        
        if a1+a2+a3+a4 != 4:
            return None
        else:                     
            # if the contour is a closed loop with some "solidity" (bounded area),
            # then we're going to say this is an O, future updates could use more of an OCR
            # or trained NN approach
            contour_area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hullArea = cv2.contourArea(hull)
            solidity = (contour_area) / float(hullArea) 
                       
            return O if solidity > .5 else X
 
def get_empty_image(img, color):
    # sometimes we just want to show a blank image if the vision component can't detect a board
    h, w, c = img.shape
    empty = color * np.ones(shape=(h, w, c), dtype=np.uint8)
     
    return empty
           
def get_decorated_gameboard(img, end_points, cells, x_contours, o_contours):   
    # take the rotated gameboard and add the cell bounding lines, contours, enumeration of cells 
    # and endpoints, etc
    decorated = img.copy() 

    for contour in x_contours:
        cv2.drawContours(decorated, [contour], 0, (0, 255, 0), 3)
    
    for contour in o_contours:
        cv2.drawContours(decorated, [contour], 0, (255, 255, 0), 3)
    
    for i, p in enumerate(end_points):
        cv2.circle(decorated, p, 8, (200, 200, 200), 2)
        cv2.putText(decorated, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    for i, cell in enumerate(cells):
        cv2.polylines(decorated, [cell], True, (0, 0, 255), 2)
        center_of_mass = get_center_of_mass(cell[:,0])
        cv2.putText(decorated, str(i), center_of_mass, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    return decorated

def get_virtual_gameboard(img, board, frame, next_move_player = EMPTY, next_move = None, state = None, win_type = None, win_info = 0):
    # build a virtual view of the gameboard, TODO the numbers below really should be variablized
    # based on the dimensions of the image
    spacing = 120
    offset_x = 140
    offset_y = 70
    o_radius = int(spacing / 5)
    x_length = int(spacing / 7)
    stroke = 8
    green = (0, 255, 0)
    cyan = (255, 255, 0)
    red = (0, 0, 255)
    win_line_padding = 10
    win_line_padding_diag = 20
    win_line_stroke = 8
    
    h, w, c = img.shape
    virtual = 255 * np.ones(shape=(h, w, c), dtype=np.uint8)
    lines = [[[1, 0], [1, 3]], [[2, 0], [2, 3]], [[0, 1], [3, 1]], [[0, 2], [3, 2]]];

    # draw the grid, but if the game is over then hide the grid every other frame, 
    # in order to create a flashing effect
    if not ((state == GAME_ACTIVE or state == GAME_DRAWN) and frame < 0 and frame % 2 == 0):
        for line in lines: 
            cv2.line(virtual, (
                offset_x + line[0][0]*spacing, \
                offset_y + line[0][1]*spacing), (\
                offset_x + line[1][0]*spacing, \
                offset_y + line[1][1]*spacing), red, 4)

    if len(board) != 9:
        return virtual
    
    for i in range(0, 3):
        for j in range(0, 3):
            content = board[i * 3 + j]
            
            if content != EMPTY or next_move_player != EMPTY:
                x = int(spacing * 1 * j + offset_x + spacing / 2)
                y = int(spacing * 1 * i + offset_y + spacing / 2)
                
                # show an O or X in each cell, but color it red if it's the AI's next intended move
                if content == O or (next_move_player == O and next_move == i * 3 + j):
                    cv2.circle(virtual, (x,y), o_radius, red if next_move_player == O and next_move == i * 3 + j else cyan, stroke)
                elif content == X or (next_move_player == X and next_move == i * 3 + j):
                    cv2.line(virtual, (x-x_length, y-x_length), (x+x_length, y+x_length), red if next_move_player == O and next_move == i * 3 + j else green, stroke)
                    cv2.line(virtual, (x+x_length, y-x_length), (x-x_length, y+x_length), red if next_move_player == O and next_move == i * 3 + j else green, stroke)
    
    # show a winning three-in-a-row line, on a horizontal row
    if win_type == "row":
        cv2.line(virtual, 
                 (offset_x + win_line_padding, offset_y + int((win_info + .5) * spacing)),
                 (offset_x + 3 * spacing - win_line_padding, offset_y + int((win_info + .5) * spacing)), 
                 green if next_move_player == X else cyan, win_line_stroke)
        
    # show a winning three-in-a-row line, on a vertical column
    elif win_type == "col":
        cv2.line(virtual, 
                 (offset_x + int((win_info + .5) * spacing), offset_y + win_line_padding), 
                 (offset_x + int((win_info + .5) * spacing), offset_y + 3 * spacing - win_line_padding), 
                 green if next_move_player == X else cyan, win_line_stroke)
    
    # show a winning three-in-a-row line, on a diagonal
    elif win_type == "diag":
        if win_info == -1:
            cv2.line(virtual, 
                (offset_x + win_line_padding_diag, offset_y + win_line_padding_diag), 
                (offset_x + 3 * spacing - win_line_padding_diag, offset_y + 3 * spacing - win_line_padding_diag), 
                green if next_move_player == X else cyan, win_line_stroke)
        else:
            cv2.line(virtual, 
                (offset_x + spacing * 3 - win_line_padding_diag, offset_y + win_line_padding_diag), 
                (offset_x + win_line_padding_diag, offset_y + 3 * spacing - win_line_padding_diag), 
                green if next_move_player == X else cyan, win_line_stroke)
        
    return virtual

def interpret(img, current_board):
    copy, _, _, _, processed = pre_process(img)
    
    try:
        # process, rotate, and interpret the current image frame
        rotation = get_rotation(processed)        
        rotated = rotate_image(processed, rotation, (0,0,0))    
        
        # determine the geometry of the gameboard     
        end_points = get_end_points(rotated)
        cells = get_cells(end_points)

        # find the Xs and Os and read them into the board state,
        # perserve the contours in order to overlay them on top of the 
        # decorated gameboard
        board, x_contours, o_contours = read_cells(current_board, rotated, cells)

        grayscale = add_channel_to_grayscale(rotated)
        decorated = get_decorated_gameboard(grayscale, end_points, cells, x_contours, o_contours)
        
        return copy, grayscale, decorated, board
        
    except:
        empty = get_empty_image(copy, 0)
        
        return copy, add_channel_to_grayscale(processed), empty, None
    