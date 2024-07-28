import hashlib
from urllib.parse import urlparse, parse_qs


def calculate_checksum(file_path, chunk_size):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_md5.update(chunk)
    except IOError as e:
        print(f"Error opening or reading file: {e}")
        raise
    return hash_md5.hexdigest()


def extract_video_id(url):
    parsed_url = urlparse(url)
    if "youtu.be" in parsed_url.netloc:
        # For shortened URLs like https://youtu.be/kF1nTG8l7B0?si=xWKIPMB-EI81R9fp
        video_id = parsed_url.path.lstrip("/")
    elif "youtube.com" in parsed_url.netloc:
        # For full URLs like https://www.youtube.com/watch?v=kF1nTG8l7B0&ab_channel=PSMH
        video_id = parse_qs(parsed_url.query).get('v', [None])[0]
    else:
        # Not a valid YouTube URL
        video_id = None
    return video_id


def read_image_to_binary(image_path):
    with open(image_path, "rb") as image_file:
        binary_image = image_file.read()
    return binary_image
