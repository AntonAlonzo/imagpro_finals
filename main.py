from ultralytics import YOLO
import cv2
import argparse
from datetime import datetime
from rcm import detectEdgeMax, detectEdgeAbs
from plate_recognition import read_license_plate
from data_interpolation import perform_data_interpolation
from data_visualization import visualize_data
from util import create_outputs_dir, create_outputs_csv_dir, create_outputs_frames_dir, create_outputs_images_dir, create_outputs_videos_dir, write_csv
from configs import get_configs, get_keys, OPENIMAGESV7_VEHICLE_LICENSE_PLATE_CLASS_ID

debug_mode = False

lpd_model = None
class_id = OPENIMAGESV7_VEHICLE_LICENSE_PLATE_CLASS_ID
source = ''
label = ''
img_sz = ()
conf = 0.5

save_to_csv = True
save_to_frames = False
save_to_images = False
save_to_videos = False
csv_output_dir = ''
frames_output_dir = ''
images_output_dir = ''
videos_output_dir = ''
csv_filename = ''
frames_filename = ''
image_filename = ''
video_filename = ''
output_mp4_filename = ''

cap = None
input_vid_out = None

detect_edge = None


def setup():
    global lpd_model, debug_mode, source, label, img_sz, conf, detect_edge
    global save_to_csv, save_to_frames, save_to_images, save_to_videos
    global csv_output_dir, frames_output_dir, images_output_dir, videos_output_dir
    global csv_filename, frames_filename, image_filename, video_filename, output_mp4_filename
    global cap, input_vid_out
    configs = get_configs()
    source, label, model, img_sz, conf, ed_alg_num, save_to_csv, save_to_frames, save_to_images, save_to_videos, debug_mode = [
        configs[key] for key in get_keys()
    ]
    # 0.1) INSTANTIATE LICENSE PLATE DETECTION PRE-TRAINED MODEL
    lpd_model = YOLO(model)
    # 0.2) CHECK IF IN DEBUG MODE
    if debug_mode == 1:
        lpd_model.predict(
            source=source, imgsz=img_sz,
            conf=0.01, classes=[class_id], show=True
        )
        # Don't continue with the program
        return False
    if debug_mode == 2:
        lpd_model.track(
            source=source, imgsz=img_sz,
            conf=0.01, classes=[class_id], show=True
        )
        # Don't continue with the program
        return False
    # 0.3) INITIALIZE EDGE DETECTION ALGORITHM TO BE USED
    if ed_alg_num == 0: detect_edge = detectEdgeMax
    elif ed_alg_num == 1: detect_edge = detectEdgeAbs
    else: pass # TODO: Add Canny edge detection here
    # 0.4) START-UP VIDEO CAPTURE DEVICE
    print("This is source:", source)
    cap = cv2.VideoCapture(source)
    # 0.5) CREATE OUTPUT DIRECTORY FOR RESULTS
    create_outputs_dir()
    # 0.6) RECORD START TIMESTAMP
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 0.7A) READY CSV OUTPUT FILE
    if save_to_csv: 
        csv_output_dir = create_outputs_csv_dir(label)
        csv_filename = csv_output_dir + label + '_' + timestamp + '.csv'
        write_csv({}, csv_filename, 'w')
    # 0.7B) READY FRAMES OUTPUT DIRECTORY
    if save_to_frames: 
        frames_output_dir = create_outputs_frames_dir(label)
        frames_filename = frames_output_dir + label + '_' + timestamp
    # 0.7C) READY IMAGES OUTPUT DIRECTORY
    if save_to_images: 
        images_output_dir = create_outputs_images_dir(label)
        image_filename = images_output_dir + label + '_' + timestamp
    # 0.7D) READY VIDEO OUTPUT FILE
    if save_to_videos: 
        videos_output_dir = create_outputs_videos_dir(label)
        video_filename = videos_output_dir + label + '_' + timestamp + '.mp4'
        output_mp4_filename = videos_output_dir + label + '_' + timestamp + '_output.mp4'
        # 0.8) SETUP TO SAVE A VIDEO OUTPUT WITH ANNOTATED RESULTS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_vid_out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    return True


def main():
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
                conf=0.1, classes=[568]
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
                if save_to_csv and license_plate_text is not None:
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
            if save_to_videos: input_vid_out.write(frame)
            results = {}
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    input_vid_out.release()
    cap.release()
    cv2.destroyAllWindows()
    if save_to_videos:
        interpolated_csv = perform_data_interpolation(csv_filename)
        visualize_data(interpolated_csv, video_filename, output_mp4_filename)

if __name__ == "__main__":
    if(setup()):
        main()