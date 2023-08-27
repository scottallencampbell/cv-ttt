import math
import cv2
import numpy as np
from engine import getBestMove

def pre_process(img):
    max_width, max_height = 640, 480
    f1 = max_width / img.shape[1]
    f2 = max_height / img.shape[0]
    f = min(f1, f2) 
    dim = (int(img.shape[1] * f), int(img.shape[0] * f))
    copy = cv2.resize(img, dim)
    
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)         
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(gray, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    inverse = (255 - thresh)    
    thin = cv2.ximgproc.thinning(inverse)
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thin, kernel, iterations = 2)
   
    return copy, gray, blur, thresh, dilated

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
    contour_points = get_board_contour_points(img)
    center_of_mass = get_center_of_mass(contour_points)
    points = get_points_sorted_by_distance(contour_points, center_of_mass)
    
    if len(points) != 12:
        raise Exception(f'Failed to detect gameboard, {len(points)} out of expected 12 contour points were found')

    interior_points_by_distance = points[0:4]
    interior_points_by_angle = get_points_sorted_by_angle(interior_points_by_distance, center_of_mass)
    
    interior = np.array(interior_points_by_angle, dtype=np.int32)
    rect = cv2.minAreaRect(interior)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    interior_rotation = get_angle(box[0], box[1])
    
    return interior_rotation

def get_end_points(img):    
    center_of_mass = get_center_of_largest_contour(img)
    contour_points = get_board_contour_points(img)
    points = get_points_sorted_by_distance(contour_points, center_of_mass)
    
    exterior_points_by_distance = points[-8:]
    exterior_points_by_angle = get_points_sorted_by_angle(exterior_points_by_distance, center_of_mass)
    
    return exterior_points_by_angle

def add_channel_to_grayscale(img):
    return np.stack((img,)*3, axis=-1)

def get_distance(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    
def get_bounding_contour(thresh):
    coords = cv2.findNonZero(thresh)
    x,y,w,h = cv2.boundingRect(coords)
    ctr = np.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h)]).reshape((-1,1,2)).astype(np.int32)
    
    return ctr

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
    degrees = (180 + degrees) % 360
    
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
   
def read_cells(current_board, img, cells):  
    board = [' '] * 9 
    thin = cv2.ximgproc.thinning(img)
    contours, _ = cv2.findContours(thin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    _, _, w, h = cv2.boundingRect(cells[0])
    min_area = .05 * w * h
    max_area = .8 * w * h
    x_contours = []
    o_contours = []
    
    for j, cell in enumerate(cells):
        if current_board[j] == " ":
            for contour in contours:
                content = read_cell(contour, cell, min_area, max_area)
                
                if content != None:
                    board[j] = content
                    
                    if content == "X":
                        x_contours.append(contour)
                    else:
                        o_contours.append(contour)
        else:
            board[j] = current_board[j]
      
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
 
def get_empty_image(img, color):
    h, w, c = img.shape
    empty = color * np.ones(shape=(h, w, c), dtype=np.uint8)
     
    return empty
           
def get_decorated_gameboard(img, end_points, cells, x_contours, o_contours):    
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

def get_virtual_gameboard(img, board):
    spacing = 120
    offset_x = 140
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

    if len(board) != 9:
        return virtual
    
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

def interpret(img, current_board):
    copy, gray, blur, thresh, processed = pre_process(img)
    
    try:
        rotation = get_rotation(processed)
        rotated = rotate_image(processed, (630 - rotation) % 360, (0,0,0))
        colored = add_channel_to_grayscale(rotated)

        end_points = get_end_points(rotated)
        cells = get_cells(end_points)

        board = [' '] * 9
        board, x_contours, o_contours = read_cells(current_board, rotated, cells)

        decorated = get_decorated_gameboard(colored, end_points, cells, x_contours, o_contours)
        virtual = get_virtual_gameboard(colored, current_board)

        left = np.concatenate((copy, decorated), axis=0)
        right = np.concatenate((colored, virtual), axis=0)
        final = np.concatenate((left, right), axis=1)
        
        return final, board

    except:
        empty = get_empty_image(copy, 0)
        virtual = get_virtual_gameboard(copy, current_board)
        left = np.concatenate((copy, empty), axis=0)
        right = np.concatenate((add_channel_to_grayscale(processed), virtual), axis=0)
        final = np.concatenate((left, right), axis=1)
        
        return final, None
        
    
# Example board state as a nine-element array


current_board = [' '] * 9 
#interpret(cv2.imread('images/hash-8a.png'))
final, board = interpret(cv2.imread('images/hash-9b.png'), current_board)

board = [' '] * 9
game_state = [board[0:3], board[3:6], board[6:9]]
   

for i in range(0,9):
    player = "X" if i % 2 == 0 else "O"
    best, best_move = getBestMove(game_state, player)
    board[best_move] = player
    game_state = [board[0:3], board[3:6], board[6:9]]
    print(game_state[0])
    print(game_state[1])
    print(game_state[2])
    print("")
#cv2.imshow("ttt", final)

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()
