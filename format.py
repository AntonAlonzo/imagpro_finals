# Plate Number Recognition System using Robinsons Compass Mask Edge Detection | RCM Header File
# IMAGPRO | Alonzo, Hernandez, Solis, Susada

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) == 6:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
           (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
           (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
           (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()):
            return True
    if len(text) == 7:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
           (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
           (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
           (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
           (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()):
            return True
    else:
        return False
    
def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    
    if len(text) == 6:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char,
                   3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        
    if len(text) == 7:
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char,
                   3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
    
    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)
    
    print(detections)

    for detection in detections:
        bbox, text, score = detection
        
#         if len(text) == 7:
#             text = text[:3] + text[4:]

        text = text.upper().replace(' ', '')
        
        print(text)

        if license_complies_format(text):
            return format_license(text), score

    return None, None