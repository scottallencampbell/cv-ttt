import math
import cv2
import numpy as np

image_path = 'images/hash-9.png'

def pre_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    thin = cv2.ximgproc.thinning(thresh)
    return gray, blur, thresh, thin
    
def add_channel_to_grayscale(img):
    return np.stack((img,)*3, axis=-1)
    
def rotate_to_orthogonal(copy, processed, gray):    
    try:
        interior_box, _ = get_interior_bounding_box(processed, gray)
    except:
        raise Exception("Failed to find an interior region, gameboard does not appear to be complete")

    interior_box_rotation = get_angle(interior_box[0], interior_box[1])
    
    rotated = rotate_image(copy, interior_box_rotation)
    gray, _, thresh, processed = pre_process(rotated)
    return rotated, gray, thresh, processed

def get_interior_bounding_box(thin, gray): 
    contours, _= cv2.findContours(thin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    
    for i in contours:
        area = cv2.contourArea(i)
        
        if area > 1000 and area > max_area:
            max_area = area
            best_cnt = i

    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]

    out_blur = cv2.GaussianBlur(out, (5,5), 0)
    out_thresh = cv2.adaptiveThreshold(out_blur, 255, 1, 1, 11, 2)

    contours, _ = cv2.findContours(out_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = np.intp(cv2.boxPoints(rect))
    center_point = get_center_of_mass(box)
   
    
    return box, center_point

def get_end_points(points, center_of_mass, max_points):
    distances = []
        
    for i, p in enumerate(points):
        distance = get_distance(p, center_of_mass)
        distances.append(distance)

    arr1 = np.array(distances)
    arr2 = np.array(points)    
    indexes = arr1.argsort()
    points_by_distance = arr2[indexes[::-1]]        
    max_distance_points = points_by_distance[:max_points]
    points_by_angle = sort_end_points_by_angle(max_distance_points, center_of_mass)    
        
    if len(points_by_angle) != 8:
        raise Exception("Failed to find 8 line segment end points, gameboard does not appear to be complete")

    test_end_point_orthogonality(points_by_angle)

    return points_by_angle

def get_distance(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def get_bounding_contour(thresh):
    coords = cv2.findNonZero(thresh)
    x,y,w,h = cv2.boundingRect(coords)
    ctr = np.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h)]).reshape((-1,1,2)).astype(np.int32)
    return ctr

def get_farthest_end_points(end_points, count):
    distances = []    
    bounds = get_bounding_contour(thresh)
    
    for i, p in enumerate(end_points):
        distance = cv2.pointPolygonTest(bounds, (int(p[0]), int(p[1])), True)
        distances.append(distance)

    arr1 = np.array(distances)
    arr2 = np.array(end_points)    
    indexes = arr1.argsort()
    sorted_distances = arr2[indexes]
    
    return sorted_distances[:count]
    
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

def rotate_image(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
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
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    degrees = 360 - (540 + degrees) % 360
    return degrees

def get_rotational_loss(end_points):    
    vert_line_0 = get_angle(end_points[4], end_points[7])
    vert_line_1 = get_angle(end_points[3], end_points[0])

    if vert_line_0 > 180: vert_line_0 = 360 - vert_line_0
    if vert_line_1 > 180: vert_line_1 = 360 - vert_line_1
    
    horiz_line_0 = get_angle(end_points[6], end_points[1])
    horiz_line_1 = get_angle(end_points[5], end_points[2])
    
    loss = (vert_line_0)**2 + (vert_line_1)**2 + (90 - horiz_line_0)**2 + (90 - horiz_line_1)**2
    return loss

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

def find_solidity(contours):
    contourArea = cv2.contourArea(contours)
    convexHull = cv2.convexHull(contours)
    contour_hull_area = cv2.contourArea(convexHull)
    solidity = float(contourArea)/contour_hull_area
    return solidity

def find_equi_diameter(contours):
    area = cv2.contourArea(contours)
    equi_diameter = np.sqrt(4*area/np.pi)
    return equi_diameter

def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)
 
def get_board_contour_points(img):
    points = []
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
    approx = cv2.approxPolyDP(selected_contour, 0.0068 * cv2.arcLength(selected_contour, True), True)

    for p in approx:
        points.append([p[0][0], p[0][1]])
        
    return points

def test_end_point_orthogonality(points):
    tolerance = 15
        
    horiz_line_0 = get_angle(points[0], points[3])
    horiz_line_1 = get_angle(points[7], points[4])
    vert_line_0 = get_angle(points[6], points[1])
    vert_line_1 = get_angle(points[5], points[2])

    if abs(180 - horiz_line_0) > tolerance \
        or abs(180 - horiz_line_1) > tolerance \
        or abs(90 - vert_line_0) > tolerance \
        or abs(90 - vert_line_1) > tolerance:
        raise Exception("Failed to detect gameboard, lines do not appear orthogonal")
   
def read_cells(board, img, cells):   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(cells[0])
    min_area = .05 * w * h
    max_area = .8 * w * h
    x_contours = []
    o_contours = []
    
    for j, cell in enumerate(cells):
        if board[j] == " ":
            for i, contour in enumerate(contours):
                content = read_cell(contour, cell, min_area, max_area)                
                   
                if content != None:
                    board[j] = content
                    
                    if content == "X":
                        x_contours.append(contour)
                    else:
                        o_contours.append(contour)
    
    return board, x_contours, o_contours

def read_cell(contour, cell, min_area, max_area):
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    
    if area > min_area and area < max_area:                
        a1 = cv2.pointPolygonTest(cell, (x,y), False) 
        a2 = cv2.pointPolygonTest(cell, (x+w,y), False) 
        a3 = cv2.pointPolygonTest(cell, (x,y+h), False) 
        a4 = cv2.pointPolygonTest(cell, (x+w,y+h), False)
        
        if a1+a2+a3+a4 != 4:
            return None
        else:                        
            contour_area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hullArea = cv2.contourArea(hull)
            solidity = (contour_area) / float(hullArea)            
            return "O" if solidity > .5 else "X"
            
def get_decorated_gameboard(img, end_points, cells, x_contours, o_contours):    
    h, w, c = img.shape
    decorated = 0 * np.ones(shape=(h, w, c), dtype=np.uint8)

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

def get_virtual_gameboard(img, board):
    spacing = 100
    offset_x = 120
    offset_y = 70
    o_radius = int(spacing / 5)
    x_length = int(spacing / 7)
    stroke = 8

    h, w, c = img.shape
    virtual = 255 * np.ones(shape=(h, w, c), dtype=np.uint8)
    lines = [[[1, 0], [1, 3]], [[2, 0], [2, 3]], [[0, 1], [3, 1]], [[0, 2], [3, 2]]];

    for line in lines: 
        cv2.line(virtual, (
            offset_x + line[0][0]*spacing, \
            offset_y + line[0][1]*spacing), (\
            offset_x + line[1][0]*spacing, \
            offset_y + line[1][1]*spacing), (0, 0, 255), 4)

    for i in range(0, 3):
        for j in range(0, 3):
            content = board[i * 3 + j]
            
            if content != " ":
                x = int(spacing * 1 * j + offset_x + spacing / 2)
                y = int(spacing * 1* i + offset_y + spacing / 2)
                
                if content == "O":
                    cv2.circle(virtual, (x,y), o_radius, (255, 255, 0), stroke)
                elif content == "X":
                    cv2.line(virtual, (x-x_length, y-x_length), (x+x_length, y+x_length), (0, 255, 0), stroke)
                    cv2.line(virtual, (x+x_length, y-x_length), (x-x_length, y+x_length), (0, 255, 0), stroke)
    
    return virtual


    
original = cv2.imread(image_path)
copy = cv2.resize(original, (0,0), fx=0.5, fy=0.5) 
gray, blur, thresh, processed = pre_process(copy)
rotated, gray, thresh, processed = rotate_to_orthogonal(copy, processed, gray)

center_of_mass = get_center_of_largest_contour(thresh)
contour_points = get_board_contour_points(thresh)
end_points = get_end_points(contour_points, center_of_mass, 8)
cells = get_cells(end_points)

#corner_top_left = (min(end_points[5][0], end_points[6][0]), min(end_points[7][1], end_points[0][1]))
#corner_top_right = (max(end_points[1][0], end_points[2][0]), min(end_points[7][1], end_points[0][1]))
#corner_bottom_left = (min(end_points[5][0], end_points[6][0]), max(end_points[4][1], end_points[3][1]))
#corner_bottom_right = (max(end_points[1][0], end_points[2][0]), max(end_points[4][1], end_points[3][1]))

#board_width = (corner_top_right[0] + corner_bottom_right[0]) / 2 - (corner_top_left[0] + corner_bottom_left[0]) / 2 
#board_height = (corner_bottom_left[1] + corner_bottom_right[1]) / 2 - (corner_top_left[1] + corner_top_right[1]) / 2

board = [' '] * 9
board, x_contours, o_contours = read_cells(board, thresh, cells)

decorated = get_decorated_gameboard(rotated, end_points, cells, x_contours, o_contours)
virtual = get_virtual_gameboard(rotated, board)

left = np.concatenate((copy, decorated), axis=0)
right = np.concatenate((add_channel_to_grayscale(thresh), virtual), axis=0)
final = np.concatenate((left, right), axis=1)
cv2.imshow(f'Tic tac toe', final) 

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()
    
    
    