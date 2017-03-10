#!/usr/bin/python
## Author: sumanth
## Date: March 08,2017
# main file running the pipline for vehicle detection and tracking
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import pipeLine

# file names for both input and output
output_file = 'project_output.mp4'
input_file = 'project_video.mp4'


# run the pipeline and generate the ouput video
video = VideoFileClip(input_file)
annotated_video = video.fl_image(pipeLine.processImage)
annotated_video.write_videofile(output_file, audio=False)
