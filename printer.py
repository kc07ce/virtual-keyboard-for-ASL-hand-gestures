import cv2 
    
#org is the coordinates of the starting point
#image is the numpy array of the image on which it has to be printed
#string is the string that has to be printed
def notepad(image,org,string):
    
    font = cv2.FONT_HERSHEY_SIMPLEX 

    # fontScale 
    fontScale = 1

    # Blue color in BGR 
    color = (0, 0, 0) 

    # Line thickness of 2 px 
    thickness = 0

    # Using cv2.putText() method 
    image = cv2.putText(image, string, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 

    return image
