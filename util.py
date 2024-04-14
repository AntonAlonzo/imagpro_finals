import os
from configs import OUTPUTS_DIR, CSV_OUTPUTS_DIR, FRAMES_OUTPUTS_DIR, IMAGES_OUTPUTS_DIR, VIDEO_OUTPUTS_DIR, PERFORMANCE_OUTPUTS_DIR

def create_outputs_dir():
    if not os.path.exists(OUTPUTS_DIR):
        os.mkdir(OUTPUTS_DIR)
        print("`%s` directory created" % (OUTPUTS_DIR))

def create_outputs_csv_dir(label=None):
    if not os.path.exists(CSV_OUTPUTS_DIR):
        os.mkdir(CSV_OUTPUTS_DIR)
        print("`%s` directory created" % (CSV_OUTPUTS_DIR))
    if label is not None:
        labelled_csv_output_dir = CSV_OUTPUTS_DIR + label + '/'
        if not os.path.exists(labelled_csv_output_dir):
            os.mkdir(labelled_csv_output_dir)
        return labelled_csv_output_dir

def create_outputs_frames_dir(label=None):
    if not os.path.exists(FRAMES_OUTPUTS_DIR):
        os.mkdir(FRAMES_OUTPUTS_DIR)
        print("`%s` directory created" % (FRAMES_OUTPUTS_DIR))
    if label is not None:
        labelled_frames_output_dir = FRAMES_OUTPUTS_DIR + label + '/'
        if not os.path.exists(labelled_frames_output_dir):
            os.mkdir(labelled_frames_output_dir)
        return labelled_frames_output_dir

def create_outputs_images_dir(label=None):
    if not os.path.exists(IMAGES_OUTPUTS_DIR):
        os.mkdir(IMAGES_OUTPUTS_DIR)
        print("`%s` directory created" % (IMAGES_OUTPUTS_DIR))
    if label is not None:
        labelled_images_output_dir = IMAGES_OUTPUTS_DIR + label + '/'
        if not os.path.exists(labelled_images_output_dir):
            os.mkdir(labelled_images_output_dir)
        return labelled_images_output_dir

def create_outputs_videos_dir(label=None):
    if not os.path.exists(VIDEO_OUTPUTS_DIR):
        os.mkdir(VIDEO_OUTPUTS_DIR)
        print("`%s` directory created" % (VIDEO_OUTPUTS_DIR))
    if label is not None:
        labelled_videos_output_dir = VIDEO_OUTPUTS_DIR + label + '/'
        if not os.path.exists(labelled_videos_output_dir):
            os.mkdir(labelled_videos_output_dir)
        return labelled_videos_output_dir
    
def create_outputs_performance_dir(label=None):
    if not os.path.exists(PERFORMANCE_OUTPUTS_DIR):
        os.mkdir(PERFORMANCE_OUTPUTS_DIR)
        print("`%s` directory created" % (PERFORMANCE_OUTPUTS_DIR))
    if label is not None:
        labelled_performance_output_dir = PERFORMANCE_OUTPUTS_DIR + label + '/'
        if not os.path.exists(labelled_performance_output_dir):
            os.mkdir(labelled_performance_output_dir)
        return labelled_performance_output_dir


def write_csv(results, output_path, mode):
    with open(output_path, mode) as f:
        if mode == 'w':
            f.write('{},{},{},{},{},{}\n'.format(
                    'frame_num', 'lp_id',
                    'license_plate_bbox', 'license_plate_bbox_score', 
                    'license_number', 'license_number_score'
                )
            )
            f.close()
            return
        print("appending to file...")
        for frame_num in results.keys():
            for lp_id in results[frame_num].keys():
                print(results[frame_num][lp_id])
                if 'license_plate' in results[frame_num][lp_id].keys() and \
                   'text' in results[frame_num][lp_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{}\n'.format(
                        frame_num, lp_id, '[{} {} {} {}]'.format(
                            results[frame_num][lp_id]['license_plate']['bbox'][0],
                            results[frame_num][lp_id]['license_plate']['bbox'][1],
                            results[frame_num][lp_id]['license_plate']['bbox'][2],
                            results[frame_num][lp_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_num][lp_id]['license_plate']['bbox_score'],
                        results[frame_num][lp_id]['license_plate']['text'],
                        results[frame_num][lp_id]['license_plate']['text_score']
                    ))
        f.close()

def write_performance_csv(results, output_path, mode):
    with open(output_path, mode) as f:
        # headers
        if mode == 'w':
            f.write('{},{},{},{},{},{},{},{}\n'.format(
                    'image_filename', 'lp_id',
                    'license_plate_bbox', 'license_plate_bbox_score', 
                    'license_number', 'license_number_score', 
                    'edge_detection_type', 'edge_detection_time_us'
                )
            )
            f.close()
            return
        # data
        print("appending to file...")
        for img_fname in results.keys():
            for lp_id in results[img_fname].keys():
                print(results[img_fname][lp_id])
                if 'license_plate' in results[img_fname][lp_id].keys() and \
                   'text' in results[img_fname][lp_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},'.format(
                        img_fname, lp_id, '[{} {} {} {}]'.format(
                            results[img_fname][lp_id]['license_plate']['bbox'][0],
                            results[img_fname][lp_id]['license_plate']['bbox'][1],
                            results[img_fname][lp_id]['license_plate']['bbox'][2],
                            results[img_fname][lp_id]['license_plate']['bbox'][3]
                        ),
                        results[img_fname][lp_id]['license_plate']['bbox_score'],
                        results[img_fname][lp_id]['license_plate']['text'],
                        results[img_fname][lp_id]['license_plate']['text_score']
                    ))
                if 'edge_detection' in results[img_fname][lp_id].keys():
                    f.write('{},{}\n'.format(
                        results[img_fname][lp_id]['edge_detection']['algotrithm'],
                        results[img_fname][lp_id]['edge_detection']['time_us']
                    ))
        f.close()
