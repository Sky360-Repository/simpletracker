#from typing_extensions import ParamSpecArgs
import cv2
import sys
import numpy as np
import time
#from object_detection.utils import config_util

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

font_size=1

#
# Tracks multiple objects in a video
#
class VideoTracker():

    def __init__(self,video):
        #print(f'VideoTracker called {video}')
        self.video=video
        self.total_trackers_finished=0
        self.total_trackers_started=0
        self.live_trackers=[]

        self.listeners=[]
        

    def listen(self,listener):
        self.listeners.append(listener)

    def create_trackers_from_keypoints(self,tracker_type,keypoints,frame):
        trackers=[]    
        for kp in keypoints:
            bbox=kp_to_bbox(kp)
            print(bbox)
            # Initialize tracker with first frame and bounding box
            self.create_and_add_tracker(tracker_type,frame,bbox)
        
        print(self.live_trackers)

    def create_and_add_tracker(self,tracker_type,frame,bbox):
        if not bbox:
            raise Exception("null bbox")

        self.total_trackers_started+=1

        tracker=Tracker(self.total_trackers_started,tracker_type)
        tracker.cv2_tracker.init(frame, bbox)
        tracker.update_bbox(bbox)
        self.live_trackers.append(tracker)

    def update_trackers(self,tracker_type,keypoints,masked):

        # cache kp -> bbox mapping for removing failed trackers
        kp_bbox_map={}
        for kp in keypoints:
            bbox=kp_to_bbox(kp)
            kp_bbox_map[kp]=bbox

        
        failed_trackers=[]        
        for idx,tracker in enumerate(self.live_trackers):
            # Update tracker
            ok, bbox = tracker.cv2_tracker.update(masked)
            
            if ok:
                # Tracking success
                #tracker_success(tracker, bbox, masked)
                tracker.update_bbox(bbox)
            else:
                # Tracking failure
                failed_trackers.append(tracker)
                next

            #Try to match the new detections with this tracker
            for idx,kp in enumerate(keypoints):
                if kp in kp_bbox_map:
                    overlap=bbox_overlap(bbox,kp_bbox_map[kp])
                    #print(f'Overlap: {overlap}; bbox:{bbox}, new_bbox:{new_bbox}')
                    if overlap>0.2:
                        del(kp_bbox_map[kp])

        #remove failed trackers from live tracking
        for tracker in failed_trackers:
            self.live_trackers.remove(tracker)
            self.total_trackers_finished+=1

        #Add new detections to live tracker
        max_trackers = 10
        for kp,new_bbox in kp_bbox_map.items():
            #Hit max trackers?
            if len(self.live_trackers) < max_trackers:
                self.create_and_add_tracker(tracker_type,masked,new_bbox)


    def detect_and_track(self, trackers_updated_callback=None, record=True, demo_mode=False, two_by_two=False):

        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'DASIAMRPN']
        tracker_type = tracker_types[7]

        # Open output video
        source_width=int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height=int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer=get_writer("outputvideo.mp4",source_width,source_height)

        font_size=source_height/1000.0
            
        # Read first frame.
        ok, frame = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        

        backSub=createBackgroundSubtractorKNN()
        output_image=fisheye_mask(frame)

        def bg_subtract(frame,background_subtractor):
            a_masked=fisheye_mask(frame)
            a_fgMask = background_subtractor.apply(a_masked)
            return (a_masked, cv2.bitwise_and(a_masked, a_masked, mask=a_fgMask))

        output_image, bgMasked = bg_subtract(frame,backSub)

        for i in range(5):
            ok, frame = self.video.read()
            output_image, bgMasked = bg_subtract(frame,backSub)   

        keypoints=detectSBD(bgMasked)

        print(keypoints)
        im_with_keypoints = cv2.drawKeypoints(output_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        msg=f"Detected {len(keypoints)} keypoints: "
        #print(msg)
        cv2.putText(im_with_keypoints, msg, (100,50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50,170,50), 2);
        #cv2.imshow('mask', im_with_keypoints)
        #cv2.waitKey()

        if demo_mode:
            for i in range(150):
                cv2.putText(frame, "First frame, look for BLOB's: ", (100,50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50,170,50), 2);
                writer.write(frame)


            for i in range(150):
                cv2.putText(im_with_keypoints, "Identified here with in red: ", (100,50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50,170,50), 2);
                writer.write(im_with_keypoints)


        #Create Trackers
        self.create_trackers_from_keypoints(tracker_type,keypoints,output_image)
        
        
        frame_count=0
        while True:
            # Read a new frame
            ok, frame = self.video.read()
            if not ok:
                break

            
            # Start timer
            timer = cv2.getTickCount()

            output_image, bgMasked = bg_subtract(frame,backSub)

            # Detect new objects of interest to pass to tracker
            keypoints=detectSBD(bgMasked)
            
            im_with_keypoints = cv2.drawKeypoints(bgMasked, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            msg=f"Detected {len(keypoints)} keypoints: "
            #print(msg)
            cv2.putText(im_with_keypoints, msg, (100,50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50,170,50), 2);

            self.update_trackers(tracker_type,keypoints,output_image)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                          
            for listener in self.listeners:
                listener.trackers_updated_callback(output_image,self.live_trackers,fps)

            for tracker in self.live_trackers :
                tracker.add_bbox_to_image(output_image,(0,255,0))

            msg=f"Trackers: started:{self.total_trackers_started}, ended:{self.total_trackers_finished}, alive:{len(self.live_trackers)}"
            print(msg)
            cv2.putText(output_image, msg, (100,200), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50,170,50), 2);
            cv2.putText(output_image, "FPS : " + str(int(fps)), (100,300), cv2.FONT_HERSHEY_SIMPLEX, font_size, (50,170,50), 2);

            if two_by_two:
                im_h1 = cv2.hconcat([frame, output_image])
                im_h2 = cv2.hconcat([bgMasked,im_with_keypoints])
            
                final_image=cv2.vconcat([im_h1,im_h2])
            else:
                final_image = output_image
            #final_image=im_v2
            # Display result
            cv2.imshow("Tracking", final_image)

            if record or demo_mode:
                writer.write(final_image)
            
            #keyboard = cv2.waitKey()
            
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

            frame_count+=1

        writer.release()
        for listener in self.listeners:
            listener.finish(self.total_trackers_started,self.total_trackers_finished)
            


#
# Tracks a single object
#
class Tracker():
    
    def __init__(self, id, tracker_type):
        self.cv2_tracker = Tracker.create_cv2_tracker(tracker_type)
        self.id=id
        self.bboxes=[]

    @staticmethod
    def create_cv2_tracker(tracker_type):
        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                param_handler = cv2.TrackerCSRT_Params()
                param_handler.use_gray=True
                #print(f"psr_threshold: {param_handler.psr_threshold}")
                param_handler.psr_threshold=0.06
                #fs = cv2.FileStorage("csrt_defaults.json", cv2.FileStorage_WRITE)
                #param_handler.write(fs)
                #fs.release()

            
                #param_handler.use_gray=True

                tracker = cv2.TrackerCSRT_create(param_handler)
            
            if tracker_type == 'DASIAMRPN':
                tracker = cv2.TrackerDaSiamRPN_create()

            
        return tracker

    def update_bbox(self,bbox):
        return self.bboxes.append(bbox)

    def get_bbox(self):
        return self.bboxes[-1]

    def add_bbox_to_image(self,frame,color):
        bbox=self.get_bbox()
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, color, 2, 1)
        cv2.putText(frame, str(self.id), (p1[0],p1[1]-4), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)

def get_writer(output_filename,width,height):
    print(f'source w,h:{(width,height)}')
    return cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*"AVC1"),30,(width,height))

def fisheye_mask(frame):
    shape=frame.shape[:2]
    #print(f'shape: {shape}')
    mask = np.zeros(shape, dtype="uint8")
    cv2.circle(mask, (int(shape[1]/2), int(shape[0]/2)), int(min(shape[0],shape[1])*0.46), 255, -1)
    return cv2.bitwise_and(frame, frame, mask=mask)
    
def createBackgroundSubtractorKNN():
    #defaults: samples:2, dist2Threshold:400.0, history: 500
    backSub = cv2.createBackgroundSubtractorKNN()
    samples=backSub.getkNNSamples()
    dist2Threshold=backSub.getDist2Threshold()
    history=backSub.getHistory()
    print(f'samples:{samples}, dist2Threshold:{dist2Threshold}, history, {history}')
    backSub.setHistory(1) #large gets many detections
    backSub.setkNNSamples(2) #1 doesn't detect small object
    backSub.setDist2Threshold(5000) #small gets many detections, large misses small movements
    return backSub


def kp_to_bbox(kp):
    (x,y)=kp.pt
    size=kp.size
    scale=6
    return (int(x-scale*size/2), int(y-scale*size/2), int(scale*kp.size), int(scale*kp.size))


def bbox_overlap(bbox1,bbox2):

#    bb1 : dict
#        Keys: {'x1', 'x2', 'y1', 'y2'}
#        The (x1, y1) position is at the top left corner,
#        the (x2, y2) position is at the bottom right corner
#    bb2 : dict
#        Keys: {'x1', 'x2', 'y1', 'y2'}
#        The (x, y) position is at the top left corner,
#        the (x2, y2) position is at the bottom right corner

    bb1={}
    bb1['x1']=bbox1[0]
    bb1['y1']=bbox1[1]
    bb1['x2']=bbox1[0] + bbox1[2]
    bb1['y2']=bbox1[1] + bbox1[3]


    bb2={}
    bb2['x1']=bbox2[0]
    bb2['y1']=bbox2[1]
    bb2['x2']=bbox2[0] + bbox2[2]
    bb2['y2']=bbox2[1] + bbox2[3]
    
    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']


    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def detectSBD(frame):
    
    params = cv2.SimpleBlobDetector_Params()
    #print(f"original sbd params:{params}")

    params.minThreshold = 3;

    params.filterByArea = 1
    params.minArea = 5

    params.filterByColor=0
#    params.blobColor=255

    detector = cv2.SimpleBlobDetector_create(params)
    #params.write('params.json')
    #print("created detector")
    #blobframe=cv2.convertScaleAbs(frame)
    #print("blobframe")
    keypoints = detector.detect(frame)
    #print("ran detect")
    return keypoints





if __name__ == '__main__' :

    # Read video
    #video = cv2.VideoCapture("videos/unknownclip.mp4") # the angles movement of uap
#    video = cv2.VideoCapture("videos/plane.mp4")
#    video = cv2.VideoCapture("videos/unknown_long.mp4")
#    video = cv2.VideoCapture("videos/unknown_direction_change.mp4")
#    video = cv2.VideoCapture("videos/samples/48301cdd-3947-43be-b0c1-01164f112625.mkv") # detecting alot of fast moving cloud (error)
#    video = cv2.VideoCapture("videos/samples/a994adff-8867-41e8-8241-6814257f202d.mkv") # detecting rain error
#    video = cv2.VideoCapture("videos/samples/577f38f5-de96-4928-af7c-121c3a78f1b1.mkv")# ok empty
#    video = cv2.VideoCapture("videos/samples/edb5ef90-62cf-4f6e-9e6c-e5409effddb3.mkv") #ok empty sky
#    video = cv2.VideoCapture("videos/samples/51201f30-75da-4c3f-bb1d-893f9a28ff68.mkv") #black bird, misses bird black??
#    video = cv2.VideoCapture("videos/samples/9e9bb2e5-4fd7-46ab-be98-0cd67f455ece.mkv") #rain, ok empty
#    video = cv2.VideoCapture("videos/samples/edd6edf4-96d0-45c8-a9cc-bf23a295e55a.mkv") #good, nothing no tracks
#    video = cv2.VideoCapture("videos/samples/a618e162-fc97-46fb-9eb5-f91e5cb6cc47.mkv") #nothing
#    video = cv2.VideoCapture("videos/samples/a683e74a-489e-4eb7-b443-fa9cc97b94d8.mkv") #nothing
#    video = cv2.VideoCapture("videos/samples/a994adff-8867-41e8-8241-6814257f202d.mkv")
#    video = cv2.VideoCapture("videos/samples/a150066f-f2ee-477e-b0ba-e8b49fb70127.mkv")
#    video = cv2.VideoCapture("videos/samples/aaaddcc6-687d-4483-9ce0-ac63211db7ef.mkv")
#    video = cv2.VideoCapture("videos/samples/aca44dd0-fdb3-4bc1-a76f-c3d85831ca88.mkv") # crane moving
#    video = cv2.VideoCapture("videos/samples/ae0e4eeb-c5c5-47bf-bd36-84a263a78e8c.mkv")
#    video = cv2.VideoCapture("videos/pelican.mp4")
#    video = cv2.VideoCapture("videos/cloud_plane.mp4")
# '03b53a8a-b5a0-4192-834e-48f2f56c007a.mkv' #insect
    input_file = "videos/unknown_long.mp4"

    video =  cv2.VideoCapture(input_file)
        
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    video_tracker = VideoTracker(video)
    video_tracker.detect_and_track()


