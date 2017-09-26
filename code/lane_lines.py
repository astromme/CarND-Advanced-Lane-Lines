import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from calibrate_camera import get_undistort
from thresholds import abs_sobel_thresh, mag_thresh, dir_threshold
from birdseye import birdseye, M
from logger import Logger
# from find_lines import find_lines, find_lines_shortcut, ShortcutFailedError, FindLinesFailedError
from overlay import drawOverlay, writeOverlayText
from Lines import Lines

undistort = get_undistort()



def arrange_in_grid(array, ncols=2):
    """
    from https://stackoverflow.com/questions/42040747/more-idomatic-way-to-display-images-in-a-grid-with-numpy
    arranges images in a grid
    """
    array = np.asarray(array)
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def pipeline(image, write_images=False, prefix='', frame=None):
    logger = Logger(enabled=write_images, prefix=prefix)
    try:
        pipeline.lines.logger = logger
    except AttributeError:
        pipeline.lines = Lines(height=image.shape[0], logger=logger)

    logger.image('image', image)

    image = undistort(image)
    logger.image('undistorted', image)

    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    H = hls[:,:,0]
    logger.image('hue', H)
    L = hls[:,:,1]
    logger.image('lightness', L)
    S = hls[:,:,2]
    logger.image('saturation', S)

    SL = cv2.addWeighted(S, 1, L, 0.5, 0)
    logger.image('saturation+0.3lightness', SL)

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(SL, orient='x', sobel_kernel=5, thresh=(40, 100))
    logger.image('gradx', gradx*255)
    grady = abs_sobel_thresh(SL, orient='y', sobel_kernel=5, thresh=(40, 100))
    logger.image('grady', grady*255)
    mag_binary = mag_thresh(SL, sobel_kernel=3, mag_thresh=(50, 100))
    logger.image('mag_binary', mag_binary*255)
    dir_binary = dir_threshold(SL, sobel_kernel=3, thresh=(0.7, 1.3))
    logger.image('dir_binary', dir_binary*255)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    logger.image('combined_binary', combined*255)

    birdseye_image = birdseye(combined)
    logger.image('birdseye', birdseye_image*255)

    pipeline.lines.find(birdseye_image)

    result = drawOverlay(image, birdseye_image, pipeline.lines, M)
    writeOverlayText(result, pipeline.lines, frame,
        ['mean: {:0.1f} sum: {:0.1f}'.format(np.mean(L), np.sum(birdseye_image))])
    logger.image('result', result)

    arranged = arrange_in_grid([
        result,
        cv2.cvtColor((birdseye_image*255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(SL, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor((combined*255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        ])

    logger.image('grid', arranged)
    return arranged


def process_test_images():
    image_filenames = glob.glob('test_images/*.jpg')

    for i, filename in enumerate(image_filenames):
        image = cv2.imread(filename)
        result = pipeline(image, write_images=True, prefix='test_images/output/image{}'.format(i))

def process_movie(movie):
    from moviepy.editor import VideoFileClip

    def movie_pipeline(image):
        if movie['entire_clip'] or (movie_pipeline.frame >= movie['start_frame'] and movie_pipeline.frame <= movie['end_frame']):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            debug = movie_pipeline.debug_all or movie_pipeline.frame in movie_pipeline.debug_frames
            result = pipeline(
                image,
                write_images=debug,
                prefix='{}/image{}'.format(movie['debug_folder'], movie_pipeline.frame),
                frame=movie_pipeline.frame)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            movie_pipeline.frame += 1
            return result
        else:
            movie_pipeline.frame += 1
            return image

    movie_pipeline.frame = 1
    movie_pipeline.debug_frames = movie['debug_frames']
    movie_pipeline.debug_all = False

    clip = VideoFileClip(movie['input'])

    video_with_overlay = clip.fl_image(movie_pipeline)
    video_with_overlay.write_videofile(movie['output'], audio=False)


def main():
    movies = {
        'project': {
            'input': 'project_video.mp4',
            'output': 'project_video_output.mp4',
            'debug_folder': 'project_video_debug',
            'start_frame': 0,
            'end_frame': 187,
            'entire_clip': True,
            'debug_frames': [1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301],
        },
        'challenge': {
            'input': 'challenge_video.mp4',
            'output': 'challenge_video_output.mp4',
            'debug_folder': 'challenge_video_debug',
            'start': 0,
            'end': 0.5,
            'start_frame': 0,
            'end_frame': 30,
            'entire_clip': True,
            'debug_frames': [],#[128, 131, 134, 135, 136, 140, 142, 144, 146, 150, 153, 156, 163, 183],
        },
        'harder_challenge': {
            'input': 'harder_challenge_video.mp4',
            'output': 'harder_challenge_video_output.mp4',
            'debug_folder': 'harder_challenge_video_debug',
            'start_frame': 120,
            'end_frame': 187,
            'entire_clip': True,
            'debug_frames': [1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301],
        },
    }

    #process_movie(movies['project'])
    #process_movie(movies['challenge'])
    process_movie(movies['harder_challenge'])
    #process_test_images()

if __name__ == '__main__':
    main()
