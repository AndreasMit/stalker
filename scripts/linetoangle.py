    lines = np.zeros(4)
    lines[0] = np.linalg.norm(box[0] - box[1])
    lines[1] = np.linalg.norm(box[1] - box[2])
    lines[2] = np.linalg.norm(box[2] - box[3])
    lines[3] = np.linalg.norm(box[3] - box[0])
    long_line = np.argmax(lines) # we assume that the long line is always the one we want to follow 
    angle = np.arctan2(box[long_line], box[(long_line+1)%4]) #returns [-pi,pi] , i want -pi/2 to pi/2 
    angle = abs(angle) # [0,pi] 
    angle -+ pi/2 # [-pi/2 , pi/2]
    cv.line(image, box[long_line], box[(long_line+1)%4], (0, 255, 0), 1)
    angle += offset #adjust to your needs