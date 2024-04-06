import cv2
import numpy as np
import csv
 

class Detect:
    def __init__(self) -> None:
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        self.cap = cv2.VideoCapture('iceberg.mp4')
        frame_width = int(self.cap.get(3)) 
        frame_height = int(self.cap.get(4))

        size= (2*frame_width, frame_height)
        
        # Object to save Video_file
        self.result = cv2.VideoWriter('ORB.mp4',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10,size)
        
        #Object to write data to csv file for post processing.
        self.f = open('ORB.csv', 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["FRAME_NO, MEAN, STD_DEV"])
        self.f.close()
        self.f = open('ORB.csv', 'a')
        self.writer = csv.writer(self.f)
        
        ##CHANGE TO SIFT, SURF, ETC
        self.orb = cv2.ORB_create()

        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")
        ret, self.first_frame = self.cap.read()
        self.first_frame = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)

        ##CHANGE TO SIFT, SURF, ETC
        self.kp1, self.des1 = self.orb.detectAndCompute(self.first_frame, None)

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        cv2.imshow("First Frame", self.first_frame)
    
    def play(self):
        # Read until video is completed
        frame_count = 0
        while(self.cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret == True:
            
                # CHANGE TO SIFT, SURF, ETC
                kp2, des2  = self.orb.detectAndCompute(frame, None)
                cv2.imshow('Frame',frame)

                matches = self.bf.match(self.des1, des2)
                matches = sorted(matches, key= lambda x:x.distance)
                # We select the top 5 strongest matches. Lower distance means less difference.
                matching_result = cv2.drawMatches(self.first_frame, self.kp1, frame, kp2, matches[:10], None, flags=2)
                ##Save as mp4
                self.result.write(matching_result)

                cv2.imshow("Matching result", matching_result)
                
                distance = []
                for m in matches [:10]:
                    distance.append(m.distance)

                #Find the mean and std_dev of the top 5 matches.
                mean = np.mean(distance)
                std_dev = np.std(distance)
                print(mean)
                self.writer.writerow([frame_count, mean, std_dev])
                frame_count +=1

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
            # Break the loop
            else: 
                break
        
        # When everything done, release the video capture object
        self.cap.release()
        self.f.close()
        self.result.release()
        # Closes all the frames
        cv2.destroyAllWindows()

if __name__ == "__main__":
    c = Detect()
    c.play()