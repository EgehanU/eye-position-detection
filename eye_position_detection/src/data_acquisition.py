import cv2
import os
import time

def capture_screenshot():
    cam = cv2.VideoCapture(0)
    class_dict = {'r': 'right', 'l': 'left', 'u': 'up', 'd': 'down'}

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Capturing", frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('s'):  # press 's' to take a screenshot
            cv2.imshow("Screenshot", frame)
            class_key = input("Enter the class ('r' for right, 'l' for left, 'u' for up, 'd' for down, 'b' for bin): ")
            
            if class_key in class_dict.keys():
                class_name = class_dict[class_key]
                if not os.path.exists(class_name):
                    os.makedirs(class_name)  # Creates a directory if doesn't exist
                img_name = os.path.join(class_name, "{}.png".format(int(time.time())))
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
            else:
                print("Discarding image.")
                
            cv2.destroyWindow("Screenshot")
                
        elif key == ord('q'):  # press 'q' to quit
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()
capture_screenshot()