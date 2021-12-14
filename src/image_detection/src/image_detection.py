#!/usr/bin/env python

import rospy
import math
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker


class ImageDetection:
    def __init__(self):
        self.subscriber = rospy.Subscriber("/vrep/image", Image, self.callback, queue_size=100)
        self.publisher = rospy.Publisher("image_marker", Marker, queue_size=100)
	
        self.bridge = CvBridge()

        # Load the image
        path = "/home/user/catkin_ws/src/image_detection/image/"
        pictures_name = ["pic001", "pic002", "pic003", "pic004", "pic005"]
        self.pictures = [cv2.imread(path+name+".jpg") for name in pictures_name]
        #pictures_name = ["pic001"]
        #self.pictures = [cv2.imread(path+name+".jpg") for name in pictures_name]
	
        self.orb_cnt = [0] * len(self.pictures)
        self.square_cnt = [0] * len(self.pictures)
        self.marked = [False] * len(self.pictures)

        self.bf = cv2.BFMatcher()
        self.orb = cv2.ORB_create(nfeatures=400)
        self.keypoints, self.descriptors = [], []
        ##okprint('test hello world1')
        
        for i, picture in enumerate(self.pictures):
            size = (400, 400)
            ##okprint('test hello world',i)
            if picture is None:
                print('Wrong path:', path)
            else:
                picture = cv2.resize(picture, size, interpolation = cv2.INTER_AREA)
                
            # find the keypoints and descriptors with ORB
            kp = self.orb.detect(picture, None)
            kp, des = self.orb.compute(picture, kp)
            # some pictures in env map is flipped
            picture = cv2.flip(picture, 1)
            kp2 = self.orb.detect(picture, None)
            kp2, des2 = self.orb.compute(picture, kp)
            
            self.keypoints.append((kp, kp2))
            self.descriptors.append((des, des2))
            
        self.markers = [Marker() for i in range(len(pictures_name))]
        ##okprint('test hello worldxx1')
        
        for marker in self.markers:
            self.markers = []
        ##okprint('test hello worldxx2')
        for i in range(len(pictures_name)):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "/camera_link"
            marker.type = marker.TEXT_VIEW_FACING
            marker.action = marker.ADD
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            ##okprint('test hello worldxx3')  
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
            marker.color.a = 1
            marker.pose.orientation.w = 1.0
            marker.text = pictures_name[i]
            self.markers.append(marker)
            print(marker.text)
            

    def _mark(self, id, img):
        print('test hello world def mark')
        img_gray = cv2.cvtColor(img, cv2.rCOLOR_BGR2GRAY)
        _, img_BW = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, contours, _ = cv2.findContours(img_BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        ratio = float(w)/h

        if ratio > 1.1 and ratio < 0.5:
            self.square_cnt[id] = 0
        else:
            self.square_cnt[id] += 1

        if (self.square_cnt[id] >= 15):
            self.markers[id].pose.position.x = 1 / math.tan(math.pi / 8 * h / img.shape[1]) * 0.5
            self.markers[id].pose.position.y = (x + w/2) / img.shape[1]
            self.markers[id].pose.position.z = 0
            print(self.markers[id].pose.position)

            self.publisher.publish(self.markers[id])
            self.marked[id] = True
            

    def _best_fit(self, img):
        try:
            best_id = -1
            best = -1
            keypoint, descriptor = self.orb.detect(img, None)

            for i, d in enumerate(self.descriptors):
                cur_cnt = 0

                for dd in d:
                    result = self.bf.knnMatch(dd, descriptor, k=2)
                    for m, n in result:
                        if m.distance < 0.75 *n.distance:
                            cur_cnt += 1

                if cur_cnt > best:
                    best = cur_cnt
                    best_id = i

            return best_id, best
        except Exception as err:
            return -1, -1

    def callback(self, img):
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        id, cnt = self._best_fit(img)

        if id == -1:
            return

        if (cnt < 120):
            self.orb_cnt[id] = 0
        else:
            self.orb_cnt[id] += 1

        if(self.orb_cnt[id] >= 10 and not self.marked[id]):
            self._mark(id, img)
        self.publisher.publish(img)
        print('test hello worldxxx')


def main():
    rospy.init_node("image_detection")
    node = ImageDetection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("KeyboardInterrupted")
        
if __name__ == "__main__":
    main()
