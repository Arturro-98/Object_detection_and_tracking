import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

import pandas as pd ### Added to save list to csv
from pathlib import Path  

from pandas._libs.tslibs.period import validate_end_alias 
import cv2  ######


from _collections import deque # Need for motion path
num_of_vid_frames = 1745
path_ball = [deque(maxlen=num_of_vid_frames*2) for _ in range(10)] # Seems like maxlen is the number of last detected frames to display. Range restrics num of appended instances 
path_player = [deque(maxlen=num_of_vid_frames*2) for _ in range(10)]



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

#######################################
count = 0 # Counting the occurance of tennis ball on bottom side of the court
data = [] # list to store counted objects to don't count twice

current_frame_no = 0  # Want to know the current frame number for stat_data() function
count_top = 0 # Counting the occurance of tennis ball on top side of the court

speed_top_player = 0 # Storing the current (each frame) speed value for 'top side player'
speed_bottom_player = 0 # Storing the current (each frame) speed value for 'top side player'
speed_ball = 0 # Storing the current (each frame) speed value for the ball object

total_dist_top_player = 0
total_dist_bottom_player = 0

#Temporary variables to display speed for certain object only every certain frame
temp_top = 0 #
temp_bottom = 0
temp_ball = 0

# Initializing lists to store coordinates of all detected objects 
Ball_coord_X = []
Ball_coord_Y = []
Top_player_X = []
Top_player_Y = []
Bottom_player_X = []
Bottom_player_Y = []

Total_top_play_det_time = [] # List with all time detections for 'top' side player 
                    

#####################   for stat_data() function    ##################################

# Top side player
Curr_fr_t_X_top_play = []    # Exact time of the target frame, to could compare later and get the time difference (Here the exact time for 'top' player if changed distance on X axis)
Curr_fr_t_Y_top_play = []    # Exact time of the target frame, to could compare later and get the time difference (Here the exact time for 'top' player if changed distance on Y axis)                   

Speed_diff_X_top_play = [] # All the distance differences for X axis 
Speed_diff_X_frame_top_play = [] # Occurance for every frame when the distance difference on X axis 
Curr_fr_t_diff_X_top_play = []   # Current time of the object as a difference between previous occurance

Speed_diff_Y_top_play = [] # All the distance differences for Y axis 
Speed_diff_Y_frame_top_play = [] # Occurance for every frame when the distance difference on Y axis 

Curr_fr_t_diff_Y_top_play = [] # Current time of the object as a difference between previous occurance   

Speed_top_player = [] # All the calculated speed values
Total_dist_top_player = [] # All the calculated distance values

# Bottom side player
Curr_fr_t_X_bottom_play = []    # Exact time of the target frame, to could compare later and get the time difference (Here the exact time for 'bottom' player if changed distance on X axis)
Curr_fr_t_Y_bottom_play = []    # Exact time of the target frame, to could compare later and get the time difference (Here the exact time for 'bottom' player if changed distance on Y axis)

Speed_diff_X_bottom_play = []   # All the distance differences for X axis 
Speed_diff_X_frame_bottom_play = [] # Occurance for every frame when the distance difference on X axis 

Curr_fr_t_diff_X_bottom_play = [] # Current time of the object as a difference between previous occurance
Speed_diff_Y_bottom_play = [] # All the distance differences for Y axis 
Speed_diff_Y_frame_bottom_play = [] # Occurance for every frame when the distance difference on Y axis 

Curr_fr_t_diff_Y_bottom_play = [] # Current time of the object as a difference between previous occurance   

Speed_bottom_player = [] # All the calculated speed values
Total_dist_bottom_player = [] # All the calculated distance values

# Ball
Curr_fr_t_X_ball = []    # Exact time of the target frame, to could compare later and get the time difference (Here the exact time for a ball if changed distance on X axis)
Curr_fr_t_Y_ball = []    # Exact time of the target frame, to could compare later and get the time difference (Here the exact time for a ball if changed distance on Y axis)

Speed_diff_X_ball = []  # All the distance differences for X axis 
Speed_diff_X_frame_ball = [] # Occurance for every frame when the distance difference on X axis 

Curr_fr_t_diff_X_ball = [] # Current time of the object as a difference between previous occurance
Speed_diff_Y_ball = []  # All the distance differences for Y axis 
Speed_diff_Y_frame_ball = [] # Occurance for every frame when the distance difference on Y axis 

Curr_fr_t_diff_Y_ball = [] # Current time of the object as a difference between previous occurance   


Current_frame_time_list = [] # all time values corresponding to a certain frame/ Storing time of each frame #



if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)

    
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
        

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    

    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources


    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        #Object speed, distance, etc
        file_name = path
        
        global current_frame_no # Want to know the current frame number for stat_data() function
        current_frame_no +=1 

        
        from pandas._libs.tslibs.period import validate_end_alias
        import cv2  ######

        # Want to get info about input video to display it to the user, as well as further extraction for time occurance of tracked objects (Need exact frame etc.)
        video_file = cv2.VideoCapture(file_name)

        global total_frames
        video_fps = video_file.get(cv2.CAP_PROP_FPS)
        total_frames = video_file.get(cv2.CAP_PROP_FRAME_COUNT)
        height = video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = video_file.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_time = total_frames/video_fps

        current_frame_time = current_frame_no/video_fps # Defining the actual time for specific/current frame
        Current_frame_time_list.append(current_frame_time) # appending all time values corresponding to certain frame
     

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            
            curr_frames[i] = im0 #Each frame of the video



            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            #########################################################
            #Determining the size of the frame: width, height 
            # w & h: width and hight of the frame/image (video resolution)
            w, h = im0.shape[1], im0.shape[0]

            global line_height
            line_height = h - 380 # Need to determine different position/height of the line as the center of the court for different videos /380, 570, 630 for another vid

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4




                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    

                        #output array of 7 objects: coordinates/bounding box 1-4 (counting from 1), id of tracked object (5),
                        # class of the object (6) 'cls', and confidence score of tracked object

                        bboxes = output[0:4] # coordinates of the object
                        id = output[4] # id of tracked object

                        
                        cls = output[5] # class of the object given from detection
                        
                        ###############################
                        #count
                        count_obj(bboxes, w, h, id, cls, current_frame_time)

                        image_frame = annotator.result() # Need to define frame/image for motion_path function
                        motion_path(bboxes, cls, image_frame)

                        stat_data(file_name, cls, bboxes, w, h, image_frame, current_frame_time)

                        ####


                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)


                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                #########################################
                #Creating the line in the middle / can count etc
                global count

                global temp_top
                global temp_bottom
                global temp_ball
                
                color=(0,255,0)
                start_point = (0, line_height) #h - 630
                end_point = (w, line_height)
                cv2.line(im0, start_point, end_point, color, thickness=2)
                thickness = 3
                org = (150, 150)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 3 

                cv2.putText(im0, str(count), org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
################################################################################################################################
                                        #########################################
                # When results are save as a video file, edit here
                #Creating the line in the middle / can count etc        

                        
                #Tennis net position for some of the videos
                # start_point = (0, h-380) 
                # end_point = (w, h-380)

                # Defining colors in BGR
                color=(255, 0, 0)
                color_red = (0, 0, 255)
                color_yellow = (59, 235, 255) 
                color_yellow_2 = (102, 255, 255) 
                color_green = (0, 255, 0)
                color_blue =(255, 0, 0)
                color_orange = (0, 165, 255)


                # Defining coordinates to draw a line as the bottom of the 'net'
                start_point = (0, line_height) # width and hight positions  #530 & 579 for another video 
                end_point = (w, line_height)

                # Want to display the speed value more smoothly, therefore taking only N't 'speed value' that will be displayed for each frame (don't display each speed value for every frame)
                if current_frame_no % 3 == 0: # Want to display every N'th frame for speed value (don't display each speed value for every frame)
                    temp_top = speed_top_player
                    temp_bottom = speed_bottom_player
                    temp_ball = speed_ball


                cv2.line(im0, start_point, end_point, color, thickness=3)
                
                thickness = 2

                org_left = (round(w * 0.032), round(h * 0.05))
                org_left_ball = (round(w * 0.032), round(h * 0.1))
                org_speed_bottom_player = (round(w * 0.032), round(h * 0.15))
                org_tot_dist_bottom_player = (round(w * 0.032), round(h * 0.20))

                org_right = (round(w * 0.65), round(h * 0.05))
                org_right_ball = (round(w * 0.65), round(h * 0.1))
                org_speed_top_player = (round(w * 0.65), round(h * 0.15))
                org_tot_dist_top_player = (round(w * 0.65), round(h * 0.20))

                org_ball = ( round(w * 0.05), round(line_height + 29) )
                

                

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1 

                disp_bot = "Ball occurance: " + str(count) 
                disp_top = "Ball occurance: " + str(count_top) 


                #cv2.putText(image, string to display, X & Y coordinates of the bottom-left corner of the text, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
                cv2.putText(im0, "Bottom player: ", org_left, font, fontScale, color_red, thickness, cv2.LINE_AA)
                cv2.putText(im0, disp_bot, org_left_ball, font, fontScale, color_yellow, thickness, cv2.LINE_AA)
                cv2.putText(im0, "Speed (a.u.): " + str(temp_bottom), org_speed_bottom_player, font, fontScale, color_yellow, thickness, cv2.LINE_AA)
                cv2.putText(im0, "Total distance (a.u.): " + str( round(total_dist_bottom_player, 2) ), org_tot_dist_bottom_player, font, fontScale, color_yellow, thickness, cv2.LINE_AA)

                cv2.putText(im0, "Top player: ", org_right, font, fontScale, color_red, thickness, cv2.LINE_AA)
                cv2.putText(im0, disp_top, org_right_ball, font, fontScale, color_yellow, thickness, cv2.LINE_AA)
                cv2.putText(im0, "Speed (a.u.): " + str(temp_top), org_speed_top_player, font, fontScale, color_yellow, thickness, cv2.LINE_AA)
                cv2.putText(im0, "Total distance (a.u.): " + str( round(total_dist_top_player, 2) ), org_tot_dist_top_player, font, fontScale, color_yellow, thickness, cv2.LINE_AA)


                cv2.putText(im0, "Speed ball (a.u.): " + str(temp_ball), org_ball, font, fontScale, color_green, thickness, cv2.LINE_AA)


                    
                        ###############################################################################

                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video




                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

        

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # To print info about input video file
        print(f" \n Frames per second: {video_fps } \n Total Frames: {total_frames} \n Height: {height} \n Width: {width} \n Video length: {video_time}")

    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def count_obj(box, w, h, id, cls,current_frame_time):
    # w & h: width and hight of the frame/image (video resolution)
    #wanna take center and line
    global count, data
    global count_top 

    global ball_bottom_occur

    center_ball_coord = (int(box[0] + (box[2] - box[0])/2), int(box[1] + (box[3] - box[1])/2)) #Gives width and hight (X & Y) coordinates respectively
    players_coordinates = ( int(box[0] + (box[2] - box[0])/2), int(box[3])) 
    

    # Implementing counting only for tennis ball (class 1) and storing its coordinates in the list at the same time
    if cls == 1: 
        #Appending the coordinates of the ball to the list
        Ball_coord_X.append(center_ball_coord[0]) #From center coordinate, the X coordinate
        Ball_coord_Y.append(center_ball_coord[1]) #From center coordinate, the Y coordinate

        # COUNTING occurance of the ball
        # if the hight position of tracked object is greater than the current hight position of the line, then count the object
        if center_ball_coord[1] > line_height: # Here counting the ball occurance on bottom half of the field
            count +=1
            #add id to data to don't count again
            #data.append(id)
        # if the hight position of tracked object is lower than the current hight position of the line, then count the object
        if center_ball_coord[1] < line_height: # Here counting the ball occurance on the top half of the field
            count_top +=1 
 
    # Want to extract coordinates for both players and store it in the list. For player class only (class 0)
    if cls == 0:
        if players_coordinates[1] > line_height: # For bottom player
            Bottom_player_X.append(players_coordinates[0])
            Bottom_player_Y.append(players_coordinates[1])

        if players_coordinates[1] < line_height: # For 'top' side player
            Top_player_X.append(players_coordinates[0])
            Top_player_Y.append(players_coordinates[1]) 
            Total_top_play_det_time.append(current_frame_time) # List with all time detections for the player (currently not used but can be useful in further implementation)

    #saving objects coordinates in csv format
    # For bottom player object
    df_X = pd.DataFrame(Bottom_player_X, columns=['X'])
    filepath = Path('Objects_coordinates/Bottom_player_X.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_X.to_csv(filepath)

    df_Y = pd.DataFrame(Bottom_player_Y, columns=['Y'])
    filepath = Path('Objects_coordinates/Bottom_player_Y.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_Y.to_csv(filepath)

    #Saving coordinates for top player object
    df_X = pd.DataFrame(Top_player_X, columns=['X'])
    filepath = Path('Objects_coordinates/Top_player_X.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_X.to_csv(filepath)

    df_Y = pd.DataFrame(Top_player_Y, columns=['Y'])
    filepath = Path('Objects_coordinates/Top_player_Y.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_Y.to_csv(filepath)

    #Saving coordinates for ball object
    df_X = pd.DataFrame(Ball_coord_X, columns=['X'])
    filepath = Path('Objects_coordinates/Ball_coord_X.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_X.to_csv(filepath)

    df_Y = pd.DataFrame(Ball_coord_Y, columns=['Y'])
    filepath = Path('Objects_coordinates/Ball_coord_Y.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_Y.to_csv(filepath)

    df = pd.DataFrame(Total_top_play_det_time, columns=['time'])
    filepath = Path('Objects_coordinates/Total_top_play_det_time.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath)  


def motion_path(bboxes, cls, image_frame):

    # Want to center coordinates for a ball and a 'bottom'/X (feet approximation) for the player
    center_ball_coord = (int(bboxes[0] + (bboxes[2] - bboxes[0])/2), int(bboxes[1] + (bboxes[3] - bboxes[1])/2)) #Gives width and hight (X & Y) coordinates respectively
    players_coordinates = ( int(bboxes[0] + (bboxes[2] - bboxes[0])/2), int(bboxes[3])) 
    
    path_ball[int(cls)].append(center_ball_coord)
    path_player[int(cls)].append(players_coordinates)
                          
    thickness = 3

    # For tennis player object
    if cls == 0:
        for tr in range(1, len(path_player[int(cls)])):
            cv2.line(image_frame, (path_player[int(cls)][tr-1]), (path_player[int(cls)][tr-1]), (255,255,0), thickness=thickness) 

    # For tennis ball object
    if cls == 1:
        for tr in range(1, len(path_ball[int(cls)])):
            cv2.line(image_frame, (path_ball[int(cls)][tr-1]), (path_ball[int(cls)][tr-1]), (55,255,0), thickness=thickness) 


def stat_data(file_name, cls, bboxes, w, h, image_frame, current_frame_time):

    global speed_top_player # giving global permission to the current (each frame) speed value variable for 'top side player'
    global speed_bottom_player # giving global permission to the current (each frame) speed value variable for 'bottom side player'
    global speed_ball # giving global permission to the current (each frame) speed value variable for the ball object

    global total_dist_top_player
    global total_dist_bottom_player
   
    players_coordinates = ( int(bboxes[0] + (bboxes[2] - bboxes[0])/2), int(bboxes[3])) # Need players coordinates to define their position first and then distance for correct player

       
    video_file = cv2.VideoCapture(file_name)

    video_fps = video_file.get(cv2.CAP_PROP_FPS)
    total_frames = video_file.get(cv2.CAP_PROP_FRAME_COUNT) #
    height = video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video_file.get(cv2.CAP_PROP_FRAME_WIDTH)

    video_time = total_frames/video_fps # Total time of the video
    
    frame_time_const = video_time/total_frames # Time for each frame (constant), need it while no distance change later 

    speed_norm = 0.04 # Since the distance base only on pixels values and don't have real speed/distance measurement. Want to normalize speed to get lower values and muliply nominator

    # Here taking data only for players (class: 0)
    if cls==0:
        if players_coordinates[1] < line_height: # For 'top' side player
            
            index = len(Top_player_X) - 1 # Need index to get correct coordinate at certain time for players
            
            # This checks the current length of the list with all time occurance of "top player" occurance at certain frame. 
            # Need it to could correctly take take a time difference between two frames
            index_time_X = len(Curr_fr_t_X_top_play) 
            index_time_Y = len(Curr_fr_t_Y_top_play)

            # Here only for 'top side player'
            # Want to be sure if there is distance/pixel's difference and the list is not empty
            if ( ( Top_player_X[index] != Top_player_X[index - 1] ) and (len(Top_player_X) != 0)) or ( ( Top_player_Y[index] != Top_player_Y[index - 1] ) and (len(Top_player_Y) != 0) ):

                # Defining the distance difference for both X and Y axis knowing its coordinates and indexes
                difference_X = abs(Top_player_X[index] - Top_player_X[index-1]) # Need to avoid negative numbers, so taking absolute value
                difference_Y = abs(Top_player_Y[index] - Top_player_Y[index-1]) # Need to avoid negative numbers, so taking absolute value
                
          
                if difference_X >= difference_Y:                
                    
                    Speed_diff_X_top_play.append(difference_X)   # Appending all the distance differences for X axis to the list
                    Speed_diff_X_frame_top_play.append(current_frame_no) # Appending to the list all frames when the distance difference on X axis occures

                    # Need exact time of the target frame, store it in the list to could compare later and get the time difference
                    # Here only the exact time for player if changed distance on X axis
                    Curr_fr_t_X_top_play.append(current_frame_time) 
                    # Exact time length for changed on X axis, knowing previous (index) occurance. Time difference
                    current_frame_time_diff_X = Curr_fr_t_X_top_play[index_time_X] - Curr_fr_t_X_top_play[index_time_X - 1]
                    #current_frame_time_diff_X = 0.04 # 0.04 is ecact time difference between two frames if 25 fps
                    Curr_fr_t_diff_X_top_play.append(round(current_frame_time_diff_X, 3)) # Appending to list: the current time of the object as a difference between previous occurance

                    #Defining speed variable
                    if current_frame_time_diff_X == 0: # Avoiding division by 0
                        speed = 0
                    else:
                        speed = round( (difference_X * speed_norm / current_frame_time_diff_X) * 1.87, 1) # Knowing distance and time can calculate the speed (dist/time)
                    # Storing the current speed value for 'top side player'
                    speed_top_player = round(speed) #don't want to display decimal number on the screen
                    total_dist_top_player += (difference_X * speed_norm) * 1.65

                    Speed_top_player.append(speed_top_player) # storing all the calculated speed values
                    Total_dist_top_player.append(total_dist_top_player) # storing all the calculated distance values

                elif difference_Y >= difference_X:
                    Speed_diff_Y_top_play.append(difference_Y)   # Appending all the distance differences for Y axis to the list
                    Speed_diff_Y_frame_top_play.append(current_frame_no) # Appending to the list all frames when the distance difference on Y axis occures

                    # Need exact time of the target frame, store it in the list to could compare later and get the time difference
                    # Here only the exact time for player if changed distance on Y axis
                    Curr_fr_t_Y_top_play.append(current_frame_time) 
                    # Exact time length for changed on Y axis, knowing previous (index) occurance. Time difference
                    current_frame_time_diff_Y = Curr_fr_t_Y_top_play[index_time_Y] - Curr_fr_t_Y_top_play[index_time_Y - 1]
                    #current_frame_time_diff_Y = 0.04 # 0.04 is ecact time difference between two frames if 25 fps
                    Curr_fr_t_diff_Y_top_play.append(round(current_frame_time_diff_Y, 3)) # Appending to list: the current time of the object as a difference between previous occurance

                    #Defining speed variable
                    if current_frame_time_diff_Y == 0: # Avoiding division by 0
                        speed = 0
                    else:
                        speed = round( (difference_Y * speed_norm / current_frame_time_diff_Y) *1.87, 1) # Knowing distance and time can calculate the speed (dist/time)
                    #print("SPEED ", speed)
                    # Storing the current speed value for 'top side player'
                    speed_top_player = round(speed) #don't want to display decimal number on the screen
                    total_dist_top_player += (difference_Y * speed_norm) * 1.65

                    Speed_top_player.append(speed_top_player) # storing all the calculated speed values
                    Total_dist_top_player.append(total_dist_top_player) # storing all the calculated distance values

            # Need this statement to append all '0' values when there is no change in distance. The same for time difference between frames 
            # Want to have the same list length as number of occurances
            else:
                speed_top_player = 0
                temp_dist = total_dist_top_player
                Speed_top_player.append(speed_top_player) # storing all the calculated speed values
                Total_dist_top_player.append(temp_dist) # storing all the calculated distance values

                # Passing frame_time_const , if 25 fps then it's 0.04
                Curr_fr_t_diff_X_top_play.append(round(frame_time_const, 3)) 
                Curr_fr_t_diff_Y_top_play.append(round(frame_time_const, 3))


#######################################################################################3
        # For 'bottom' side player
        if players_coordinates[1] > line_height: 
            
            index = len(Bottom_player_X) - 1 # Need index to get correct coordinate at certain time for players
            
            # This checks the current length of the list with all time occurance of "bottom player" occurance at certain frame. 
            # Need it to could correctly take take a time difference between two frames
            index_time_X = len(Curr_fr_t_X_bottom_play) 
            index_time_Y = len(Curr_fr_t_Y_bottom_play)

            # Here only for 'bottom side player'
            # Want to be sure if there is distance/pixel's difference and the list is not empty
            if ( ( Bottom_player_X[index] != Bottom_player_X[index - 1] ) and (len(Bottom_player_X) != 0)) or ( ( Bottom_player_Y[index] != Bottom_player_Y[index - 1] ) and (len(Bottom_player_Y) != 0) ):

                # Defining the distance difference for both X and Y axis knowing its coordinates and indexes
                difference_X = abs(Bottom_player_X[index] - Bottom_player_X[index-1]) # Need to avoid negative numbers, so taking absolute value
                difference_Y = abs(Bottom_player_Y[index] - Bottom_player_Y[index-1]) # Need to avoid negative numbers, so taking absolute value
                
          
                if difference_X >= difference_Y:                
                    
                    Speed_diff_X_bottom_play.append(difference_X)   # Appending all the distance differences for X axis to the list
                    Speed_diff_X_frame_bottom_play.append(current_frame_no) # Appending to the list all frames when the distance difference on X axis occures

                    # Need exact time of the target frame, store it in the list to could compare later and get the time difference
                    # Here only the exact time for player if changed distance on X axis
                    Curr_fr_t_X_bottom_play.append(current_frame_time) 

                    # Exact time length for changed on X axis, knowing previous (index) occurance. Time difference
                    current_frame_time_diff_X = Curr_fr_t_X_bottom_play[index_time_X] - Curr_fr_t_X_bottom_play[index_time_X - 1]
                    #current_frame_time_diff_X = 0.04 # 0.04 is ecact time difference between two frames if 25 fps
                    Curr_fr_t_diff_X_bottom_play.append(round(current_frame_time_diff_X, 3)) # Appending to list: the current time of the object as a difference between previous occurance

                    #Defining speed variable
                    if current_frame_time_diff_X == 0: # Avoiding division by 0
                        speed = 0
                    else:
                        speed = round(difference_X * speed_norm / current_frame_time_diff_X, 1) # Knowing distance and time can calculate the speed (dist/time)
                    # Storing the current speed value for 'top side player'
                    speed_bottom_player = round(speed) #don't want to display decimal number on the screen
                    total_dist_bottom_player += (difference_X * speed_norm) 

                    Speed_bottom_player.append(speed_bottom_player) # storing all the calculated speed values
                    Total_dist_bottom_player.append(total_dist_bottom_player) # storing all the calculated distance values

                elif difference_Y >= difference_X:
                    Speed_diff_Y_bottom_play.append(difference_Y)   # Appending all the distance differences for Y axis to the list
                    Speed_diff_Y_frame_bottom_play.append(current_frame_no) # Appending to the list all frames when the distance difference on Y axis occures

                    # Need exact time of the target frame, store it in the list to could compare later and get the time difference
                    # Here only the exact time for player if changed distance on Y axis
                    Curr_fr_t_Y_bottom_play.append(current_frame_time) 

                    # Exact time length for changed on Y axis, knowing previous (index) occurance. Time difference
                    current_frame_time_diff_Y = Curr_fr_t_Y_bottom_play[index_time_Y] - Curr_fr_t_Y_bottom_play[index_time_Y - 1]
                    #current_frame_time_diff_Y = 0.04 # 0.04 is ecact time difference between two frames if 25 fps
                    Curr_fr_t_diff_Y_bottom_play.append(round(current_frame_time_diff_Y, 3)) # Appending to list: the current time of the object as a difference between previous occurance

                    #Defining speed variable
                    if current_frame_time_diff_Y == 0: # Avoiding division by 0
                        speed = 0
                    else:
                        speed = round(difference_Y * speed_norm / current_frame_time_diff_Y, 1) # Knowing distance and time can calculate the speed (dist/time)
                    #print("SPEED ", speed)
                    # Storing the current speed value for 'top side player'
                    speed_bottom_player = round(speed) #don't want to display decimal number on the screen

                    total_dist_bottom_player += (difference_Y * speed_norm)

                    Speed_bottom_player.append(speed_bottom_player) # storing all the calculated speed values
                    Total_dist_bottom_player.append(total_dist_bottom_player) # storing all the calculated distance values

            # Need this statement to append all '0' values when there is no change in distance. The same for time difference between frames 
            # Want to have the same list length as number of occurances
            else:
                speed_bottom_player = 0
                temp_dist = total_dist_bottom_player
                Speed_bottom_player.append(speed_bottom_player) # storing all the calculated speed values
                Total_dist_bottom_player.append(temp_dist) # storing all the calculated distance values     

                # Passing frame_time_const , if 25 fps then it's 0.04
                Curr_fr_t_diff_X_bottom_play.append(round(frame_time_const, 3))
                Curr_fr_t_diff_Y_bottom_play.append(round(frame_time_const, 3))

   
    #### Estimating the speed for a ball object (class: 1)

    if cls == 1:         
        index = len(Ball_coord_X) - 1 # Need index to get correct coordinate at certain time for a ball object
        
        # This checks the current length of the list with all time occurance of ball occurance at certain frame. 
        # Need it to could correctly take a time difference between two frames
        index_time_X = len(Curr_fr_t_X_ball) 
        index_time_Y = len(Curr_fr_t_Y_ball)

        # Here only for a ball object
        # Want to be sure if there is distance/pixel's difference and the list is not empty
        if ( ( Ball_coord_X[index] != Ball_coord_X[index - 1] ) and (len(Ball_coord_X) != 0)) or ( ( Ball_coord_Y[index] != Ball_coord_Y[index - 1] ) and (len(Ball_coord_Y) != 0) ):

            # Defining the distance difference for both X and Y axis knowing its coordinates and indexes
            difference_X = abs(Ball_coord_X[index] - Ball_coord_X[index-1]) # Need to avoid negative numbers, so taking absolute value
            difference_Y = abs(Ball_coord_Y[index] - Ball_coord_Y[index-1]) # Need to avoid negative numbers, so taking absolute value
            
      
            if difference_X >= difference_Y:                
                
                Speed_diff_X_ball.append(difference_X)   # Appending all the distance differences for X axis to the list
                Speed_diff_X_frame_ball.append(current_frame_no) # Appending to the list all frames when the distance difference on X axis occures

                # Need exact time of the target frame, store it in the list to could compare later and get the time difference
                # Here only the exact time for the ball if changed distance on X axis
                Curr_fr_t_X_ball.append(current_frame_time) 

                # Exact time length for changed on X axis, knowing previous (index) occurance. Time difference
                #current_frame_time_diff_X = Curr_fr_t_X_ball[index_time_X] - Curr_fr_t_X_ball[index_time_X - 1]
                current_frame_time_diff_X = 0.04 # 0.04 is ecact time difference between two frames if 25 fps
                Curr_fr_t_diff_X_ball.append(round(current_frame_time_diff_X, 3)) # Appending to list: the current time of the object as a difference between previous occurance

                # Defining speed variable
                if current_frame_time_diff_X == 0: # Avoiding division by 0
                    speed = 0
                else:
                    speed = round(difference_X * speed_norm / current_frame_time_diff_X, 1) # Knowing distance and time can calculate the speed (dist/time)
                #print("SPEED ", speed)
                # Storing the current speed value for the ball
                speed_ball = round(speed) #don't want to display decimal number on the screen

            elif difference_Y >= difference_X:
                Speed_diff_Y_ball.append(difference_Y)   # Appending all the distance differences for Y axis to the list
                Speed_diff_Y_frame_ball.append(current_frame_no) # Appending to the list all frames when the distance difference on Y axis occures

                # Need exact time of the target frame, store it in the list to could compare later and get the time difference
                # Here only the exact time for the ball if changed distance on Y axis
                Curr_fr_t_Y_ball.append(current_frame_time) 

                # Exact time length for changes on Y axis, knowing previous (index) occurance. Time difference
                #current_frame_time_diff_Y = Curr_fr_t_Y_ball[index_time_Y] - Curr_fr_t_Y_ball[index_time_Y - 1]
                current_frame_time_diff_Y = 0.04 # 0.04 is ecact time difference between two frames if 25 fps

                Curr_fr_t_diff_Y_ball.append(round(current_frame_time_diff_Y, 3)) # Appending to list: the current time of the object as a difference between previous occurance

                #Defining speed variable
                if current_frame_time_diff_Y == 0: # Avoiding division by 0
                    speed = 0
                else:
                    speed = round(difference_Y * speed_norm / current_frame_time_diff_Y, 1) # Knowing distance and time can calculate the speed (dist/time)
                #print("SPEED ", speed)
                # Storing the current speed value for 'top side player'
                speed_ball = round(speed) #don't want to display decimal number on the screen

    # Saving objects extracted data in csv format
    # For top player object
    df = pd.DataFrame( Speed_top_player, columns=['speed'])
    filepath = Path('Objects_coordinates/Speed_top_player.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath)

    df = pd.DataFrame(Total_dist_top_player, columns=['distance'])
    filepath = Path('Objects_coordinates/Total_dist_top_player.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath)

    # For bottom player object
    df = pd.DataFrame(Speed_bottom_player, columns=['speed'])
    filepath = Path('Objects_coordinates/Speed_bottom_player.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath)

    df = pd.DataFrame(Total_dist_bottom_player, columns=['distance'])
    filepath = Path('Objects_coordinates/Total_dist_bottom_player.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath)


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))





if __name__ == "__main__":
    opt = parse_opt()
    main(opt)