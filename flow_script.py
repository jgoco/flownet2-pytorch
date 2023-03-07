import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
from PIL import Image
import scipy.io
import cv2
import flowiz as fz
import main
import pdb

def echo_to_png(echo, frame_directory):
    for i in range(0, echo.shape[2]):
        frame = echo[:,:,i]
        image = Image.fromarray(frame)
        fn = '{:06d}'.format(i) + '_frame.png'
        filename = frame_directory + fn
        image.save(filename)
        print('converted frame', i)
    print('COMPLETED: echo_to_png')
    
def flo_to_png(echo, flo_directory, overlay_directory):
    for i in range(0, echo.shape[2]-1):
        frame = echo[:,:,i]
        fn = '{:06d}'.format(i) +'.flo'
        filename = flo_directory + fn
        flo = fz.convert_from_file(filename)

        # [OPTIONAL] invert colour
        black = np.zeros(flo.shape, np.uint8)
        flo = black - flo   
    
        # recenter flo
        lr = round((frame.shape[1] - flo.shape[1]) / 2)
        ud = round((frame.shape[0] - flo.shape[0]) / 2)
        recentered_flo = frame.copy()
        recentered_flo[ud:ud+flo.shape[0], lr:lr+flo.shape[1], :] = flo
    
        # plot echo and flo video
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(frame, interpolation='nearest')
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.imshow(frame, interpolation='nearest')
        plt.imshow(recentered_flo, alpha=0.7, interpolation='bilinear')
    
        # save figure
        save_name ='{:06d}'.format(i) +'_frame.png'
        save_path = overlay_directory + save_name
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()   
        print('saved frame', i)       
    print('COMPLETED: flo_to_png')

def png_to_avi(overlay_directory, video_name, num_frames):
    fps = 15
    sample_path = overlay_directory + '000000_frame.png'
    sample = cv2.imread(sample_path)
    height, width, layers = sample.shape
    
    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height), True)
    for i in range(0, num_frames):
        file = '{:06d}'.format(i) +'_frame.png'
        path = overlay_directory + file
        image = cv2.imread(path)
        writer.write(image)
        print('wrote frame', i)
    writer.release()
    print('COMPLETED: png_to_avi')
    
if __name__ == '__main__':
    print('start flow_script')
    
    # ----------Parse Inputs----------
    import sys
    matfile_path = sys.argv[1]
    #matfile_path = 'dataset/flow_echo_sample.mat'
    result_name = sys.argv[2]
    #result_name = 'results/trial1'
    
    # ----------Create Directory and File Names----------
    result_directory = result_name
    frame_directory = 'dataset/echo_frames/'
    flo_directory = result_directory + '/inference/run.epoch-0-flow-field/'
    overlay_directory = result_directory + '/overlay_frames/'
    video_name = result_directory + '/output.avi'
    
    # ----------Load Echo from .mat File----------
    echo = scipy.io.loadmat(matfile_path)['cine'] # HxWxFxC
    
    # ----------Convert Echo to .png Frames----------
    os.makedirs(frame_directory, exist_ok=True)
    echo_to_png(echo, frame_directory)
    
    # ----------Run FlowNet2----------
    main.main()
    
    # ----------Visualize Flow----------
    os.makedirs(overlay_directory, exist_ok=True)
    flo_to_png(echo, flo_directory, overlay_directory)
    
    # ----------Turn .png Frames into .avi Video----------
    png_to_avi(overlay_directory, video_name, echo.shape[2]-1)
    
    print('END: flow_script')