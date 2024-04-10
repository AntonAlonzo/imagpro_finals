import argparse

OPENIMAGESV7_VEHICLE_LICENSE_PLATE_CLASS_ID = 568

MODELS_DIR = './models/'

OUTPUTS_DIR = './outputs/'
CSV_OUTPUTS_DIR = OUTPUTS_DIR + 'csv/'
FRAMES_OUTPUTS_DIR = OUTPUTS_DIR + 'frames/'
IMAGES_OUTPUTS_DIR = OUTPUTS_DIR + 'images/'
VIDEO_OUTPUTS_DIR = OUTPUTS_DIR + 'videos/'

_KEYS = [
    'source',
    'label',
    'model',
    'img_sz',
    'conf',
    'edge_detec',
    'save_to_csv', 
    'save_to_frames',
    'save_to_images',
    'save_to_videos',
    'debug'
]

# Default configs
_CONFIGS = {
    'source': 1, # external web cam
    'label': 'test',
    'model': MODELS_DIR + 'yolov8n-oiv7.pt',
    'img_sz': (736, 1280),
    'conf': 0.1,
    'edge_detec': 0,        # 0 - Max RobComp, 1 - Abs RobComp, 3 - Canny
    'save_to_csv': True, 
    'save_to_frames': False,
    'save_to_images': False,
    'save_to_videos': False,
    'debug': 0              # 0 - No debug, 1 - Image debug, 2 - Video debug
}

def get_keys():
    return _KEYS

def get_configs():
    # Fetch arguments
    parser = argparse.ArgumentParser(description="License Plate Recognition Configuration")
    parser.add_argument("--source", help="Source of the input")
    parser.add_argument("--label", help="Label of the input and output")
    parser.add_argument("--model", help="Model to be used for plate detection and tracking")
    parser.add_argument("--img-sz", help="Input Frame/Image size")
    parser.add_argument("--conf", help="Confidence level of plate detection")
    parser.add_argument(
        "--edge-detec", 
        choices=['0', '1', '2'],
        help="Edge detection algorithm: 0 - Max RobComp, 1 - Abs RobComp, 3 - Canny"
    )
    parser.add_argument(
        "-c", "--save-to-csv",
        action="store_true",
        help="Save results to a CSV file under the `%s` directory" % (CSV_OUTPUTS_DIR)
    )
    parser.add_argument(
        "-f", "--save-to-frames",
        action="store_true",
        help="Save results per frame under the `%s` directory" % (FRAMES_OUTPUTS_DIR)
    )
    parser.add_argument(
        "-i", "--save-to-images",
        action="store_true",
        help="Save resulting image under the `%s` directory" % (IMAGES_OUTPUTS_DIR)
    )
    parser.add_argument(
        "-v", "--save-to-videos",
        action="store_true",
        help="Save resulting video under the `%s` directory" % (VIDEO_OUTPUTS_DIR)
    )
    parser.add_argument(
        "-d", "--debug",
        choices=['0', '1', '2'],
        help="Debug mode: 0 - No debug, 1 - Image debug, 2 - Video debug"
    )

    # Parse arguments
    args = parser.parse_args()
    if args.source:
        if is_int(args.source): _CONFIGS['source'] = int(args.source)
        if _CONFIGS['source'] == 0: _CONFIGS['img_sz'] = (480, 640) # default size
        if _CONFIGS['source'] == 1: _CONFIGS['img_sz'] = (736, 1280) # default size
    if args.label:
        _CONFIGS['label'] = args.label
    if args.model: _CONFIGS['model'] = MODELS_DIR + args.model
    if args.img_sz:
        h, w = args.img_sz
        if is_int(h) and is_int(w):
            _CONFIGS['img_sz'] = (int(h), int(w))
    if args.conf:
        if is_float(args.conf): _CONFIGS['conf'] = float(args.conf)
        else: raise Exception('Invalid confidence level argument value!')
        if _CONFIGS['conf'] < 0.01: _CONFIGS['conf'] = 0.01
        elif _CONFIGS['conf'] > 1.0: _CONFIGS['conf'] = 1.0
    if args.edge_detec:
        if is_int(args.edge_detec): _CONFIGS['edge_detec'] = int(args.edge_detec)
        if _CONFIGS['edge_detec'] < 0 or _CONFIGS['edge_detec'] > 2: raise Exception('Invalid Edge detection algorithm')
    
    if args.debug:
        if is_int(args.debug): _CONFIGS['debug'] = int(args.debug)
        if _CONFIGS['debug'] < 0 or _CONFIGS['debug'] > 2: raise Exception('Invalid debug mode')
    if args.save_to_csv:
        _CONFIGS['save_to_csv'] = args.save_to_csv
    if args.save_to_frames:
        _CONFIGS['save_to_frames'] = args.save_to_frames
    if args.save_to_images:
        _CONFIGS['save_to_images'] = args.save_to_images
    if args.save_to_videos:
        _CONFIGS['save_to_videos'] = args.save_to_videos
    return _CONFIGS


def is_int(val):
    try: 
        int(val)
    except ValueError:
        return False
    else:
        return True

def is_float(val):
    try: 
        float(val)
    except ValueError:
        return False
    else:
        return True