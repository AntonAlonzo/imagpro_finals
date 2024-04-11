import csv
import numpy as np
from scipy.interpolate import interp1d

def interpolate_captured_data(data):
    # 1.0) EXTRACT NECESSARY DATA FROM EACH COLUMN OF EACH ROW
    frame_nums = np.array([int(row['frame_num']) for row in data])
    lp_ids = np.array([int(float(row['lp_id'])) for row in data])
    lp_bboxes = np.array([
        list(map(float, row['license_plate_bbox'][1:-1].split()))
        for row in data
    ])

    interpolated_data = []
    unique_lp_ids = np.unique(lp_ids)
    for lp_id in unique_lp_ids:
        frame_numbers = [
            p['frame_num'] for p in data
            if int(float(p['lp_id'])) == int(float(lp_id))
        ]
        print(frame_numbers, lp_id)

        # Filter data for a specific car ID
        lp_mask = lp_ids == lp_id
        lp_frame_numbers = frame_nums[lp_mask]
        lp_bboxes_interpolated = []

        first_frame_number = lp_frame_numbers[0]
        last_frame_number = lp_frame_numbers[-1]

        for i in range(len(lp_bboxes[lp_mask])):
            frame_number = lp_frame_numbers[i]
            lp_bbox = lp_bboxes[lp_mask][i]

            if i > 0:
                prev_frame_number = lp_frame_numbers[i-1]
                prev_lp_bbox = lp_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frame's bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_lp_bbox, lp_bbox)), axis=0, kind='linear')
                    interpolated_lp_bboxes = interp_func(x_new)

                    lp_bboxes_interpolated.extend(interpolated_lp_bboxes[1:])
                
            lp_bboxes_interpolated.append(lp_bbox)

        for i in range(len(lp_bboxes_interpolated)):
            frame_num = first_frame_number + i
            row = {}
            row['frame_num'] = str(frame_num)
            row['lp_id'] = str(lp_id)
            row['license_plate_bbox'] = ' '.join(map(str, lp_bboxes_interpolated[i]))
            

            if str(frame_num) not in frame_numbers:
                # Imputed row, set the following fields to 0
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [
                    p for p in data
                    if int(p['frame_num']) == frame_num
                    and int(float(p['lp_id'])) == int(float(lp_id))
                ][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'
            
            interpolated_data.append(row)

    return interpolated_data


def perform_data_interpolation(csv_file):
    # LOAD THE CSV FILE
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Interpolate missing data
    interpolated_data = interpolate_captured_data(data)
    print(interpolated_data)

    # WRITE UDPDATED DATA TO A NEW CSV FILE
    header = [
        'frame_num', 
        'lp_id',
        'license_plate_bbox', 
        'license_plate_bbox_score', 
        'license_number', 
        'license_number_score'
    ]
    interpolated_csv = csv_file[:-4] + '_interpolated' + csv_file[-4:]
    with open(interpolated_csv, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)
    
    return interpolated_csv
