from ultralytics import YOLO
import cv2
# import argparse
import time
from datetime import datetime
from rcm import detectEdgeMax, detectEdgeAbs, cannyEdge
from plate_recognition import read_license_plate
from data_interpolation import perform_data_interpolation
from data_visualization import visualize_data, draw_license_plate_boundary_box, display_label, display_text
from util import create_outputs_dir, create_outputs_csv_dir, create_outputs_frames_dir, create_outputs_images_dir, create_outputs_videos_dir, create_outputs_performance_dir, write_csv, write_performance_csv
from configs import get_configs, get_keys, OPENIMAGESV7_VEHICLE_LICENSE_PLATE_CLASS_ID

debug_mode = False

lpd_model = None
class_id = OPENIMAGESV7_VEHICLE_LICENSE_PLATE_CLASS_ID
source = ''
fps = 30.0
label = ''
img_sz = ()
conf = 0.1

is_live = True
is_execute = False
is_record = False
csv_output_dir = ''
frames_output_dir = ''
images_output_dir = ''
videos_output_dir = ''
csv_filename = ''
frames_filename = ''
image_filename = ''
video_filename = ''
output_mp4_filename = ''
output_img_filename = ''
algorithm_name = ''

cap = None
input_vid_out = None

detect_edge = None


def setup():
    global lpd_model, source, fps, label, img_sz, conf, detect_edge
    global is_live, is_execute, is_record, debug_mode
    global csv_output_dir, frames_output_dir, images_output_dir, videos_output_dir
    global csv_filename, frames_filename, image_filename, video_filename, output_mp4_filename, output_img_filename, algorithm_name
    global cap, input_vid_out
    configs = get_configs()
    source, fps, label, model, img_sz, conf, ed_alg_num, is_live, is_execute, is_record, debug_mode = [
        configs[key] for key in get_keys()
    ]
    # 0.1) INSTANTIATE LICENSE PLATE DETECTION PRE-TRAINED MODEL
    # if is_live or is_execute:
    lpd_model = YOLO(model)
    # 0.2) INITIALIZE EDGE DETECTION ALGORITHM TO BE USED
    if ed_alg_num == 0:
        detect_edge = detectEdgeMax
        algorithm_name = 'rcm_max'
    elif ed_alg_num == 1:
        detect_edge = detectEdgeAbs
        algorithm_name = 'rcm_abs'
    elif ed_alg_num == 2:
        detect_edge = cannyEdge
        algorithm_name = 'canny'
    # 0.3) CHECK IF IN DEBUG MODE
    if debug_mode == 1:
        print(f'[DEBUG] Starting Debug mode {debug_mode}...')
        # Check using image
        lpd_model.predict(
            source=source, imgsz=img_sz,
            conf=0.01, classes=[class_id], show=True
        )
        # Don't continue with the program
        return False
    if debug_mode == 2:
        print(f'[DEBUG] Starting Debug mode {debug_mode}...')
        # Check using web cam
        lpd_model.track(
            source=source, imgsz=img_sz,
            conf=0.01, classes=[class_id], show=True
        )
        # Don't continue with the program
        return False
    if debug_mode == 3:
        print(f'[DEBUG] Starting Debug mode {debug_mode}...')
        # Create outputs directory
        create_outputs_dir()
        # Create directory for performance
        performance_dir = create_outputs_performance_dir(label)
        # Filenames for outputs
        input_img_filename = source.split('/')[-1][:-4]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_img_filename = performance_dir + input_img_filename + f'_img_output_{timestamp}.jpg'
        csv_filename = performance_dir + input_img_filename + f'_csv_output_{timestamp}.csv'
        return False
    print("This is source:", source)
    cap = cv2.VideoCapture(source)
    while not cap.isOpened(): print('Waiting to open device...')
    # If user will only record, then it will skip some of the setup
    if is_record: 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_sz[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_sz[0])
        print("FPS setting:", fps)
        print("(width,height):", img_sz[1], img_sz[0])
        status = cap.set(cv2.CAP_PROP_FPS, fps)
        print("Is setting up of FPS successful:", status)
        # Create outputs directory
        create_outputs_dir()
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Ready the directory and file for the video
        videos_output_dir = create_outputs_videos_dir(label)
        video_filename = videos_output_dir + label + '_' + timestamp + '.mp4'
        output_mp4_filename = videos_output_dir + label + '_' + timestamp + '_output.mp4'
        # 0.8) SETUP TO SAVE A VIDEO OUTPUT WITH ANNOTATED RESULTS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if input_fps == 0.0: input_fps = fps
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Recording at (fps):", input_fps)
        print("(width,height):", width, height)
        input_vid_out = cv2.VideoWriter(video_filename, fourcc, input_fps, (width, height))
        return 0
    # If recognition model is live, then it will override the web cam settings
    if is_live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_sz[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_sz[0])
        cap.set(cv2.CAP_PROP_FPS, fps)
        return 1
    # If recognition model is going to be executed on a pre-recorded video,
    # it will use default settings of video
    if is_execute:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if input_fps == 0.0: input_fps = fps
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # img_sz = (height, width)
        # 0.5) CREATE OUTPUT DIRECTORY FOR RESULTS
        create_outputs_dir()
        # 0.6) GET THE FILENAME OF THE VIDEO FILE
        input_filename = source.split('/')[-1][:-4]
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 0.7A) READY CSV OUTPUT FILE
        csv_output_dir = create_outputs_csv_dir(label)
        csv_filename = csv_output_dir + input_filename + '.csv'
        write_csv({}, csv_filename, 'w')
        # 0.7B) READY FRAMES OUTPUT DIRECTORY
        # if save_to_frames: 
        #     frames_output_dir = create_outputs_frames_dir(label)
        #     frames_filename = frames_output_dir + label + '_' + timestamp
        # 0.7C) READY IMAGES OUTPUT DIRECTORY
        # if save_to_images: 
        #     images_output_dir = create_outputs_images_dir(label)
        #     image_filename = images_output_dir + label + '_' + timestamp
        # 0.7D) READY VIDEO OUTPUT FILE
        videos_output_dir = create_outputs_videos_dir(label)
        video_filename = source
        output_mp4_filename = videos_output_dir + input_filename + '_output.mp4'
        # 0.8) SETUP TO SAVE A VIDEO OUTPUT WITH ANNOTATED RESULTS
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        # input_vid_out = cv2.VideoWriter(video_filename, fourcc, input_fps, (width, height))
        return 2
    return -1


def live_license_plate_recognition():
    # Storing results
    results = {}
    frame_num = -1
    ret = True
    # Continuously sample frames from video capturing device
    while ret:
        # 1.0) READ/SAMPLE FRAME
        ret, frame = cap.read()
        frame_num += 1
        # 2.0) CHECK IF A FRAME IS READ
        if ret:
            results[frame_num] = {}
            # 3.0) PERFORM LICENSE PLATE DETECTION INFERENCE TO DETECT THE LICENSE PLATES
            detections = lpd_model.track(
                frame, imgsz=img_sz,
                conf=conf, classes=[568]
            )[0]
            post_detections = []
            # 4.0) CHECK IF TRACKER STARTS TRACKING THE LICENSE PLATE
            is_tracking = detections.boxes.is_track
            # 5.0) GO OVER EACH DETECTED PLATES
            for detection in detections.boxes.data.tolist():
                # 5.1) CHECK IF CURRENT LICENSE PLATE IS BEING TRACKED
                if is_tracking:
                    x1, y1, x2, y2, track_id, score, class_id = detection
                    print(x1, y1, x2, y2, track_id, score, class_id)
                    lp_id = track_id
                else:
                    x1, y1, x2, y2, score, class_id = detection
                    print(x1, y1, x2, y2, score, class_id)
                    lp_id = -1
                # 5.2) APPEND THE DETECTED LICENSE PLATE TO THE LIST OF DETECTIONS FOR VIEWING
                post_detections.append([x1, y1, x2, y2, score])
                draw_license_plate_boundary_box(frame, x1, y1, x2, y2)
                # 5.3) CROP THE LICENSE PLATE ONLY
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                # 5.4) CONVERT CROPPED IMAGE OF LICENSE PLATE INTO A GRAYSCALE IMAGE
                lp_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # 5.5) APPLY IMAGE PROCESSING TO THE CROPPED LICENSE PLATE USING THE DESIRED EDGE DETECTION ALGORITHM
                lp_crop_rcm_dip = detect_edge(lp_crop_gray) # Edge Detection algorithm
                # 5.6) READ LICENSE PLATE
                license_plate_text, license_plate_text_score = read_license_plate(lp_crop_rcm_dip)
                # 5.7) STORE CSV RESULTS
                if license_plate_text is not None:
                    display_label(frame, x1, y1, x2)
                    display_text(
                        frame, 
                        license_plate_text,
                        x1, y1
                    )
                    results[frame_num][lp_id] = {
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'bbox_score': score,
                        'text': license_plate_text,
                        'text_score': license_plate_text_score,
                    }}
                    # Record results
                    # write_csv(results, csv_filename, 'a')
            # 5.8) SAVE FRAME AS PART OF AN INPUT VIDEO
            # if save_to_videos: input_vid_out.write(frame)
            cv2.imshow('frame', frame)
            results = {}
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def execute_license_plate_recognition():
    # Storing results
    results = {}
    frame_num = -1
    ret = True
    # Continuously sample frames from video capturing device
    while ret:
        # 1.0) READ/SAMPLE FRAME
        ret, frame = cap.read()
        frame_num += 1
        # 2.0) CHECK IF A FRAME IS READ
        if ret:
            results[frame_num] = {}
            # 3.0) PERFORM LICENSE PLATE DETECTION INFERENCE TO DETECT THE LICENSE PLATES
            detections = lpd_model.track(
                frame, imgsz=img_sz,
                conf=conf, classes=[568]
            )[0]
            post_detections = []
            # 4.0) CHECK IF TRACKER STARTS TRACKING THE LICENSE PLATE
            is_tracking = detections.boxes.is_track
            # 5.0) GO OVER EACH DETECTED PLATES
            for detection in detections.boxes.data.tolist():
                # 5.1) CHECK IF CURRENT LICENSE PLATE IS BEING TRACKED
                if is_tracking:
                    x1, y1, x2, y2, track_id, score, class_id = detection
                    print(x1, y1, x2, y2, track_id, score, class_id)
                    lp_id = track_id
                else:
                    x1, y1, x2, y2, score, class_id = detection
                    print(x1, y1, x2, y2, score, class_id)
                    lp_id = -1
                # 5.2) APPEND THE DETECTED LICENSE PLATE TO THE LIST OF DETECTIONS FOR VIEWING
                post_detections.append([x1, y1, x2, y2, score])
                # 5.3) CROP THE LICENSE PLATE ONLY
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                # 5.4) CONVERT CROPPED IMAGE OF LICENSE PLATE INTO A GRAYSCALE IMAGE
                lp_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # 5.5) APPLY IMAGE PROCESSING TO THE CROPPED LICENSE PLATE USING THE DESIRED EDGE DETECTION ALGORITHM
                lp_crop_rcm_dip = detect_edge(lp_crop_gray) # Edge Detection algorithm
                # 5.6) READ LICENSE PLATE
                license_plate_text, license_plate_text_score = read_license_plate(lp_crop_rcm_dip)
                # 5.7) STORE CSV RESULTS
                if license_plate_text is not None:
                    results[frame_num][lp_id] = {
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'bbox_score': score,
                        'text': license_plate_text,
                        'text_score': license_plate_text_score,
                    }}
                    # Record results
                    write_csv(results, csv_filename, 'a')
            # 5.8) SAVE FRAME AS PART OF AN INPUT VIDEO
            # input_vid_out.write(frame)
            # cv2.imshow('frame', frame)
            results = {}
    # input_vid_out.release()
    cap.release()
    cv2.destroyAllWindows()
    # Interpolate collected data
    interpolated_csv = perform_data_interpolation(csv_filename)
    # Add visualization of collected data to the video
    visualize_data(interpolated_csv, video_filename, output_mp4_filename)


def record_data():
    print(input_vid_out)
    ret = True
    start_time = time.time() 
    diff = 0
    # Continuously sample frames from video capturing device
    while ret:
        # 1.0) READ/SAMPLE FRAME
        ret, frame = cap.read()
        # 2.0) CHECK IF A FRAME IS READ
        if ret:
            # display current frame
            cv2.imshow('Live Recording', frame)
            # SAVE FRAME AS PART OF AN INPUT VIDEO
            input_vid_out.write(frame)
            end_time = time.time()
            diff = (end_time - start_time) * 1000
            print(diff)
        
        # if cv2.waitKey(int((1000 // fps) - diff)) & 0xFF == ord('q'): break
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    input_vid_out.release()
    cap.release()
    cv2.destroyAllWindows()


def image_license_plate_recognition():
    print(f'[DEBUG] Started Debug mode {debug_mode}.')
    # Storing results
    results = {}
    write_performance_csv(results, csv_filename, 'w')
    # Source image
    img_filename = source.split('/')[-1]
    image = cv2.imread(source)
    output_image = image.copy()
    results[img_filename] = {}
    detections = lpd_model.track(
        output_image, imgsz=img_sz,
        conf=conf, classes=[568]
    )[0]
    post_detections = []
    detected_license = False
    recognized_text = False
    is_tracking = detections.boxes.is_track
    for detection in detections.boxes.data.tolist():
        if not detected_license: detected_license = True
        if is_tracking:
            x1, y1, x2, y2, track_id, score, class_id = detection
            print(x1, y1, x2, y2, track_id, score, class_id)
            lp_id = track_id
        else:
            x1, y1, x2, y2, score, class_id = detection
            print(x1, y1, x2, y2, score, class_id)
            lp_id = -1
        post_detections.append([x1, y1, x2, y2, score])
        draw_license_plate_boundary_box(output_image, x1, y1, x2, y2)
        license_plate_crop = output_image[int(y1):int(y2), int(x1):int(x2), :]
        lp_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        # Check performance of Edge Detection Algorithm
        start_time = datetime.now()
        lp_crop_rcm_dip = detect_edge(lp_crop_gray) # Edge Detection algorithm
        end_time = datetime.now()
        cv2.imshow(f'Cropped vehicle registration/license plate of {img_filename}', lp_crop_rcm_dip)
        delta = end_time - start_time
        comp_time_s = delta.total_seconds()
        comp_time_ms = comp_time_s * 1000
        comp_time_us = comp_time_s * 1000000
        print('Edge detection time elapsed (s):', comp_time_s)
        print('Edge detection time elapsed (ms):', comp_time_ms)
        print('Edge detection time elapsed (us):', comp_time_us)
        license_plate_text, license_plate_text_score = read_license_plate(lp_crop_rcm_dip)
        if license_plate_text is not None:
            recognized_text = True
            display_label(output_image, x1, y1, x2)
            display_text(
                output_image, 
                license_plate_text,
                x1, y1
            )
            results[img_filename][lp_id] = {
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'bbox_score': score,
                    'text': license_plate_text,
                    'text_score': license_plate_text_score,
                },
                'edge_detection': {
                    'algotrithm': algorithm_name,
                    'time_exp': '{:e}'.format(comp_time_ms),
                    'time_s': '{:.4f}'.format(comp_time_s),
                    'time_ms': '{:.4f}'.format(comp_time_ms),
                    'time_us': '{:.4f}'.format(comp_time_us)
                }
            }
            # Record results
            write_performance_csv(results, csv_filename, 'a')
    if detected_license:
        # cv2.imshow(img_filename, image)
        cv2.imshow(f'Annotated {img_filename}', output_image)
        if recognized_text:
            print('Saving annotated image...')
            cv2.imwrite(output_img_filename, output_image)
        while True: 
            if cv2.waitKey(0) or cv2.waitKey(0) & 0xFF == ord('q'): break
    else:
        if not detected_license:
            print('[WARNING] No vehicle registration/license plate/s found!')
        if not recognized_text:
            print('[WARNING] Registration/license plate/s not recognized!')
    cv2.destroyAllWindows()
    print(f'[DEBUG] Stopped Debug mode {debug_mode}.')


if __name__ == "__main__":
    status = setup()
    if status == False and isinstance(status, bool):
        if debug_mode == 3:
            image_license_plate_recognition()
    elif status == -1:
        print("An error occured!")
    elif status == 0:
        record_data()
    elif status == 1:
        live_license_plate_recognition()
    elif status == 2:
        execute_license_plate_recognition()
