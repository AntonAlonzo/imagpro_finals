import ast

import cv2
import numpy as np
import pandas as pd

def draw_car_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def draw_license_plate_boundary_box(frame, x1, y1, x2, y2):
    img = cv2.rectangle(
        frame, 
        (int(x1), int(y1)), 
        (int(x2), int(y2)), 
        (0, 0, 255), 
        12
    )
    return img

def display_label(frame, x1, y1, x2, y2=None):
    max_w = int(x2) if int(x2) > (int(x1) + 300) else int(x1) + 300
    frame[
        int(y1) - 72:int(y1),
        int(x1):max_w, 
        :    
    ] = (255, 255, 255)


def display_text(frame, text, x, y):
    img = cv2.putText(
        frame,
        text,
        (int(x) + 10, int(y) - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        12, cv2.LINE_AA
    )
    return img
    # cv2.putText(
    #     frame,
    #     license_plate[
    #         df.iloc[row_index]['car_id']
    #     ]['license_plate_number'],
    #     (50, 50),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 0, 0),
    #     2, cv2.LINE_AA
    # )
    # (text_width, text_height), _ = cv2.getTextSize(
    #     license_plate[
    #         df.iloc[row_index]['lp_id']
    #     ]['license_plate_number'],
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     2, 12
    # )


def visualize_data(interpolated_csv, input_mp4, output_mp4_filename):
    # 1.0) LOAD INTERPOLATED DATA
    results = pd.read_csv(interpolated_csv)

    # 2.0) LOAD INPUT VIDEO
    cap = cv2.VideoCapture(input_mp4)

    # 3.0) CONFIGURE VIDEO WRITIER FOR SAVING RESULTING VIDEO
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_output = cv2.VideoWriter(output_mp4_filename, fourcc, fps, (width, height))
    print('In visualization (fps):', fps)

    license_plate = {}
    for lp_id in np.unique(results['lp_id']):
        max = np.max(
            results[results['lp_id'] == lp_id]['license_number_score']
        )
        license_plate[lp_id] = {
            'license_crop': None,
            'license_plate_number': results[
                (results['lp_id'] == lp_id) 
                & (results['license_number_score'] == max)
            ]['license_number'].iloc[0]
        }
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[
            (results['lp_id'] == lp_id) 
            & (results['license_number_score'] == max)
        ]['frame_num'].iloc[0])
        ret, frame = cap.read()
        x1, y1, x2, y2 = ast.literal_eval(results[
            (results['lp_id'] == lp_id) 
            & (results['license_number_score'] == max)
        ]['license_plate_bbox'].iloc[0].replace(
            '[ ', '['
        ).replace(
            '   ', ' '
        ).replace(
            '  ', ' '
        ).replace(
            ' ', ','
        ))
        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
        license_plate[lp_id]['license_crop'] = license_crop
    
    frame_no = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_no += 1
        if ret:
            df = results[results['frame_num'] == frame_no]
            for row_index in range(len(df)):
                # Get boundary box data of detected license plate
                x1, y1, x2, y2 = ast.literal_eval(
                    df.iloc[row_index]['license_plate_bbox']
                        .replace('[ ', '[')
                        .replace('   ', ' ')
                        .replace('  ', ' ')
                        .replace(' ', ',')
                )
                # Draw boundary box of detected license plate
                draw_license_plate_boundary_box(frame, x1, y1, x2, y2)
                # crop license plate
                # license_crop = license_plate[
                #     df.iloc[row_index]['lp_id']
                # ]['license_crop']
                # H, W, _ = license_crop.shape
                # print(frame.shape)
                # print(license_crop.shape)

                try:
                    display_label(frame, x1, y1, x2)
                    display_text(
                        frame, 
                        license_plate[
                            df.iloc[row_index]['lp_id']
                        ]['license_plate_number'],
                        x1, y1
                    )
                except Exception as e:
                    print(e)
            vid_output.write(frame)
            frame = cv2.resize(frame, (1280, 720))
    vid_output.release()
    cap.release()
    cv2.destroyAllWindows()


# visualize_data(
#     './outputs/csv/test/test_2024-04-11_01-26-52_interpolated.csv',
#     './outputs/videos/test/test_2024-04-11_01-26-52.mp4',
#     './outputs/videos/test/fix.mp4'
# )
