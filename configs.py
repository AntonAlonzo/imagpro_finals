import argparse

OPENIMAGESV7_VEHICLE_LICENSE_PLATE_CLASS_ID = 568

MODELS_DIR = './models/'

OUTPUTS_DIR = './outputs/'
CSV_OUTPUTS_DIR = OUTPUTS_DIR + 'csv/'
FRAMES_OUTPUTS_DIR = OUTPUTS_DIR + 'frames/'
IMAGES_OUTPUTS_DIR = OUTPUTS_DIR + 'images/'
VIDEO_OUTPUTS_DIR = OUTPUTS_DIR + 'videos/'
PERFORMANCE_OUTPUTS_DIR = OUTPUTS_DIR + 'performances/'

_KEYS = [
    'source',
    'fps',
    'label',
    'model',
    'img_sz',
    'conf',
    'edge_detec',
    'live',
    'execute',
    'record',
    'debug'
]

# Default configs
_CONFIGS = {
    'source': 1, # external web cam
    'fps': 30.0,
    'label': 'test',
    'model': MODELS_DIR + 'yolov8n-oiv7.pt',
    'img_sz': (736, 1280),
    'conf': 0.1,
    'edge_detec': 0,        # 0 - Max RobComp, 1 - Abs RobComp, 3 - Canny
    'live': False,
    'execute': False,
    'record': False,
    'debug': 0              # 0 - No debug, 1 - Image debug, 2 - Video debug, 3 - Verbose Image debug with Edge detection performance
}

def get_keys():
    return _KEYS

def get_configs():
    # Fetch arguments
    parser = argparse.ArgumentParser(description="License Plate Recognition Configuration")
    parser.add_argument("--source", help="Source of the input")
    parser.add_argument("--fps", help="FPS setting of camera")
    parser.add_argument("--label", help="Label of the input and output")
    parser.add_argument("--model", help="Model to be used for plate detection and tracking")
    parser.add_argument("--img-sz", help="Input Frame/Image size. (height, width)")
    parser.add_argument("--conf", help="Confidence level of plate detection")
    parser.add_argument(
        "--edge-detec", 
        choices=['0', '1', '2'],
        help="Edge detection algorithm: 0 - Max RobComp, 1 - Abs RobComp, 3 - Canny"
    )
    parser.add_argument(
        "-l", "--live",
        action="store_true",
        help="Perform License Plate Recognition on live webcam"
    )
    parser.add_argument(
        "-e", "--execute",
        action="store_true",
        help="Perform License Plate Recognition on a pre-recorded video"
    )
    parser.add_argument(
        "-r", "--record",
        action="store_true",
        help="Record or collect sample data"
    )
    parser.add_argument(
        "-d", "--debug",
        choices=['0', '1', '2', '3'],
        help="Debug mode: 0 - No debug, 1 - Image debug, 2 - Video debug"
    )

    # Parse arguments
    args = parser.parse_args()
    if args.source:
        if is_int(args.source):
            _CONFIGS['source'] = int(args.source)
            if _CONFIGS['source'] == 0: 
                _CONFIGS['img_sz'] = (480, 640) # default size
            elif _CONFIGS['source'] == 1: 
                _CONFIGS['img_sz'] = (736, 1280) # default size
        else: _CONFIGS['source'] = args.source
    if args.fps:
        _CONFIGS['fps'] = float(args.fps)
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
        if is_int(args.edge_detec): 
            _CONFIGS['edge_detec'] = int(args.edge_detec)
            if _CONFIGS['edge_detec'] < 0 or _CONFIGS['edge_detec'] > 2: 
                raise Exception('Invalid Edge detection algorithm')
    if args.live:
        _CONFIGS['live'] = args.live
    if args.execute:
        _CONFIGS['execute'] = args.execute
    if args.record:
        _CONFIGS['record'] = args.record
    if args.debug:
        if is_int(args.debug): _CONFIGS['debug'] = int(args.debug)
        if _CONFIGS['debug'] < 0 or _CONFIGS['debug'] > 3: raise Exception('Invalid debug mode')
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