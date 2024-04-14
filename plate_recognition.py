import re
import itertools
import string
import easyocr

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False) # Change to true if PC has GPU

# Mapping dictionaries for character conversion
dict_char_to_int = {
    'O': '0',
    'I': '1',
    'J': '3',
    'Z': '3',
    'A': '4',
    'L': '4', # TODO: perform more test to confirm this
    'G': '6',
    'S': '5'
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',       
    '4': 'A',
    '6': 'G',
    '5': 'S',
    '@': 'D' # TODO: perform more test to confirm this
}


def remove_whitespaces(text):
    removed_whitespaces = re.sub(r'\s+', '', text)
    joined = ''.join(itertools.filterfalse(lambda x: x.isspace(), removed_whitespaces))
    return joined


def read_license_plate(license_plate_crop_gray):
    detections = reader.readtext(license_plate_crop_gray, decoder='wordbeamsearch')
    cleaned_text = ''
    for detection in detections:
        bbox, text, score = detection
        
        # text = text.upper().replace(' ', '')
        cleaned_text += remove_whitespaces(text.upper())
    print(len(cleaned_text))
    print(cleaned_text)
    # if is_4w_license_format_compliant(cleaned_text):
    #     return format_to_4w_license(cleaned_text), score
    # if is_2w_v1_license_format_compliant(cleaned_text):
    #     return format_to_2w_v1_license(cleaned_text), score
    # if is_2w_v2_license_format_compliant(cleaned_text):
    #     return format_to_2w_v2_license(cleaned_text), score
    # Check if 4-wheel license/registration plate compliant
    is_compliant, compliant_text = is_4w_license_format_compliant(cleaned_text)
    if is_compliant:
        return format_to_4w_license(compliant_text), score, cleaned_text
    # Check if 2-wheel license/registration plate compliant (v1)
    is_compliant, compliant_text = is_2w_v1_license_format_compliant(cleaned_text)
    if is_compliant:
        return format_to_2w_v1_license(compliant_text), score, cleaned_text
    # Check if 2-wheel license/registration plate compliant (v2)
    is_compliant, compliant_text = is_2w_v2_license_format_compliant(cleaned_text)
    if is_compliant:
        return format_to_2w_v2_license(compliant_text), score, cleaned_text
    return None, score, cleaned_text


def format_to_4w_license(text):
    # Four wheel vehicle format: LLL DDDD (L - letter, D - digit)
    formatted_license_plate = ''
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        2: dict_int_to_char,
        3: dict_char_to_int, 
        4: dict_char_to_int,
        5: dict_char_to_int, 
        6: dict_char_to_int
    }
    # text_len = len(text)
    # raw_text = text[:]
    # For removing unwanted characters
    # if text_len != 7:
    #     for i in range(text_len):
    #         start_ind = 0 + i
    #         end_ind = 7 + i
    #         substr = raw_text[start_ind:end_ind]
    #         if is_4w_license_format_compliant(substr):
    #             raw_text = substr
    for j in range(0,7):
        if text[j] in mapping[j].keys():
            formatted_license_plate += mapping[j][text[j]]
        else:
            formatted_license_plate += text[j]
    print(formatted_license_plate)
    return formatted_license_plate


def format_to_2w_v1_license(text):
    # Motorcycle vehicle format: DDD LLL (L - letter, D - digit)
    formatted_license_plate = ''
    mapping = {
        0: dict_char_to_int, 
        1: dict_char_to_int,
        2: dict_char_to_int,
        3: dict_int_to_char,
        4: dict_int_to_char,
        5: dict_int_to_char
    }
    for j in range(0,6):
        if text[j] in mapping[j].keys():
            formatted_license_plate += mapping[j][text[j]]
        else:
            formatted_license_plate += text[j]
    print(formatted_license_plate)
    return formatted_license_plate


def format_to_2w_v2_license(text):
    # Motorcycle vehicle format: L DDD LL (L - letter, D - digit)
    formatted_license_plate = ''
    mapping = {
        0: dict_int_to_char,
        1: dict_char_to_int, 
        2: dict_char_to_int,
        3: dict_char_to_int,
        4: dict_int_to_char,
        5: dict_int_to_char
    }
    for j in range(0,6):
        if text[j] in mapping[j].keys():
            formatted_license_plate += mapping[j][text[j]]
        else:
            formatted_license_plate += text[j]
    print(formatted_license_plate)
    return formatted_license_plate


def is_4w_license_format_compliant(text):
    # Four wheel vehicle format: LLL DDDD (L - letter, D - digit)
    text_len = len(text)
    if text_len < 7: return False, text
    if len(text) != 7:
        # Double check if format exists in string
        for i in range(text_len):
            if (6 + i) < text_len and (
                (text[0 + i] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and 
                (text[1 + i] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
                (text[2 + i] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and
                (text[3 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3 + i] in dict_char_to_int.keys()) and
                (text[4 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4 + i] in dict_char_to_int.keys()) and
                (text[5 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5 + i] in dict_char_to_int.keys()) and
                (text[6 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6 + i] in dict_char_to_int.keys())
            ):
                return True, text[0+i:7+i]
        return False, text
    if (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and 
        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
        (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and
        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and
        (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and
        (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and
        (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys())
    ):
        return True, text
    return False, text


def is_2w_v1_license_format_compliant(text):
    # Motorcycle vehicle format: DDD LLL (L - letter, D - digit)
    text_len = len(text)
    if text_len < 6: return False, text
    if text_len != 6:
        # Double check if format exists in string
        for i in range(text_len):
            if (5 + i) < text_len and (
                (text[0 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0 + i] in dict_char_to_int.keys()) and
                (text[1 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1 + i] in dict_char_to_int.keys()) and
                (text[2 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2 + i] in dict_char_to_int.keys()) and
                (text[3 + i] in string.ascii_uppercase or text[3 + i] in dict_int_to_char.keys()) and 
                (text[4 + i] in string.ascii_uppercase or text[4 + i] in dict_int_to_char.keys()) and
                (text[5 + i] in string.ascii_uppercase or text[5 + i] in dict_int_to_char.keys())
            ):
                return True, text[0+i:6+i]
        return False, text
    if (
        (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and
        (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and
        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and
        (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and 
        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
    ):
        return True, text
    return False, text


def is_2w_v2_license_format_compliant(text):
    # Motorcycle vehicle format: L DDD LL (L - letter, D - digit)
    text_len = len(text)
    if text_len < 6: return False, text
    if text_len != 6:
        # Double check if format exists in string
        text_len = len(text)
        for i in range(text_len):
            if (5 + i) < text_len and (
                (text[0 + i] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and 
                (text[1 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1 + i] in dict_char_to_int.keys()) and
                (text[2 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2 + i] in dict_char_to_int.keys()) and
                (text[3 + i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3 + i] in dict_char_to_int.keys()) and
                (text[4 + i] in string.ascii_uppercase or text[4 + i] in dict_int_to_char.keys()) and
                (text[5 + i] in string.ascii_uppercase or text[5 + i] in dict_int_to_char.keys())
            ):
                return True, text[0+i:6+i]
        return False, text
    if (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and 
        (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and
        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and
        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and
        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
    ):
        return True, text
    return False, text
