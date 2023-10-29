import cv2
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from matplotlib import cm

config = {}
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

minYPoint = min(config['y_coordinate_points_for_analysis'])
yPointsMapping = {yC: (yC - minYPoint) * config['1_pixel_in_mm_y_axis']
                  for yC in config['y_coordinate_points_for_analysis']}
print(yPointsMapping)


# Parsing the arguments
def ArgParse():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-fn", "--filename", default='cropped.mp4', type=str,
                    help="Path of the input video.")

    args = vars(ap.parse_args())  # Converting it to dictionary.

    return args


def generate_roi_mask_image(h, w):
    # Generating empty image first
    roiMaskImage = np.zeros((h, w), dtype=np.uint8)

    # ROI Contour
    roiContour = np.array(
        [[config['roi_mask_p1']], [config['roi_mask_p2']], [config['roi_mask_p3']], [config['roi_mask_p4']]])

    # Drawing the roi contour
    roiMaskImage = cv2.drawContours(roiMaskImage, [roiContour], -1, 255, -1)

    # Saving the roiMaskImage
    cv2.imwrite(config['roi_mask_image_name'], roiMaskImage)

    return roiMaskImage


def crop_roi(img):
    return img[config['roi_y1']:config['roi_y2'] + 1, config['roi_x1']:config['roi_x2'] + 1, :].copy()


def get_mask(img, roiMask, colorScale, colorLb, colorUb):
    # Changing color scale of the image if required
    if colorScale == 'BGR':
        pass
    elif colorScale == 'Lab':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif colorScale == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError(f"Color Scale can only take values 'BGR', 'Lab', or 'HSV'. Given value is '{colorScale}'")

    # Extracting white area by inRange operation
    areaMask = cv2.inRange(img, np.array(colorLb), np.array(colorUb))

    # Extracting only the required area.
    areaMask = cv2.bitwise_and(areaMask, roiMask)

    return areaMask


def get_biggest_2_contours(maskImg):
    # Getting contours
    contours = cv2.findContours(maskImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Getting the biggest 2 contours
    biggest2Contours = sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[:2]

    # Creating new mask image with these contours
    newMask = np.zeros(maskImg.shape, dtype=np.uint8)
    newMask = cv2.drawContours(newMask, biggest2Contours, -1, 255, -1)

    return newMask


def get_white_area_mask(roiImg, roiMaskImage):
    # Getting initial white mask
    whiteAreaMaskImg = get_mask(roiImg, roiMaskImage, config['white_area_color_scale'], config['white_area_color_lb'],
                                config['white_area_color_ub'])

    # Getting blue flame mask
    blueAreaMaskImg = get_mask(roiImg, roiMaskImage, config['blue_area_color_scale'], config['blue_flame_color_lb'],
                               config['blue_flame_color_ub'])

    # Removing blue area mask from white area mask
    whiteAreaMaskImg = cv2.bitwise_and(whiteAreaMaskImg, cv2.bitwise_not(blueAreaMaskImg))

    # Getting two biggest contours' image
    finalWhiteAreaMaskImg = get_biggest_2_contours(whiteAreaMaskImg)

    return finalWhiteAreaMaskImg


def get_left_right_distances(whiteAreaMaskImg, yCoordinatePoints):
    frameToShow = whiteAreaMaskImg.copy()
    frameToShow = cv2.merge((frameToShow, frameToShow, frameToShow))

    # Middle line X coordinate
    midX = whiteAreaMaskImg.shape[1] // 2

    distances = {}
    for y in yCoordinatePoints:
        dist = {
            'left': None,
            'right': None,
        }

        # Left dist
        for x in range(midX, -1, -1):
            if whiteAreaMaskImg[y, x] == 255:
                dist['left'] = midX - x
                # frameToShow = cv2.line(frameToShow, (x, y), (midX, y), (0, 255, 0), 2)
                break

        # Right dist
        for x in range(midX, whiteAreaMaskImg.shape[1], 1):
            if whiteAreaMaskImg[y, x] == 255:
                dist['right'] = x - midX
                # frameToShow = cv2.line(frameToShow, (midX, y), (x, y), (0, 0, 255), 2)
                break

        distances[y] = dist

    return distances, frameToShow


def process_frame(frame, roiMaskImage):
    # Getting Mask of the white area inside ROI
    whiteAreaMask = get_white_area_mask(frame, roiMaskImage)

    # Get left and right distances from middle line
    leftRightDistances, frameToShow = get_left_right_distances(whiteAreaMask,
                                                               config['y_coordinate_points_for_analysis'])

    return whiteAreaMask, leftRightDistances, frameToShow


def perform_analysis(resDf: pd.DataFrame, FPS, framesToSkip, yCoordinatePoints):
    # Setting the frame number col as the index column
    resDf.set_index('FrameNumber', inplace=True)

    # First graph line points {frame number : [(x-distance, yCoordinate)]}
    linePointsLeft = defaultdict(list)
    linePointsRight = defaultdict(list)
    frameNums = list(range(max(framesToSkip, 1), len(resDf.index), int(FPS * config['time_interval_for_observation'])))
    for i, frameNum in enumerate(frameNums):
        # For each y coordinate
        for yC in yCoordinatePoints:
            # Adding left side distance
            if not np.isnan(resDf.at[frameNum, f'Y_{yC}_leftDist']):
                linePointsLeft[frameNum].append([resDf.at[frameNum, f'Y_{yC}_leftDist'], yC])

            # Adding right side distance
            if not np.isnan(resDf.at[frameNum, f'Y_{yC}_rightDist']):
                linePointsRight[frameNum].append([resDf.at[frameNum, f'Y_{yC}_rightDist'], yC])

    return linePointsLeft, linePointsRight, frameNums


def display_graph_1(linePoints, frameNums, title):
    # Define the colormap here
    cmap = cm.get_cmap('viridis', len(frameNums))
    norm = Normalize(0, len(frameNums) - 1)

    for t, frameNum in enumerate(frameNums):
        if frameNum in linePoints:
            color = cmap(norm(t))  # Assign a color based on the colormap and normalization
            plt.plot(
                [yPointsMapping[pt[1]] for pt in linePoints[frameNum]],
                [config['1_pixel_in_mm_x_axis'] * pt[0] for pt in linePoints[frameNum]],
                label=f"T = {t * config['time_interval_for_observation']} sec",
                color=color  # Set the color based on the colormap
            )

    plt.title(title)
    plt.xlabel('yCoordinate')
    plt.ylabel('X-Distance')
    # plt.legend()

    # Save the graph to a file
    plt.savefig(f'{title}_graph.png')  # Replace 'graph1.png' with an appropriate filename
    plt.close()  # Close the figure to clear the plot

def display_graph_2(linePoints, frameNums, title, yCoordinatePoints, FPS):
    plt.figure(figsize=(12, 8))

    # Create a color map based on the frame numbers
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(frameNums), vmax=max(frameNums))

    for t, frameNum in enumerate(frameNums):
        if frameNum in linePoints:
            data = linePoints[frameNum]  # Extract the data for this frame
            x_data = [pt[0] for pt in data]  # Extract x-values
            y_data = [pt[1] for pt in data]  # Extract y-values
            colors = [frameNum] * len(x_data)  # Create a list of colors for each point
            plt.scatter(
                [yPointsMapping[y] for y in y_data],  # Convert y-values
                [x * config['1_pixel_in_mm_x_axis'] for x in x_data],  # Convert x-values
                c=colors, cmap=cmap, norm=norm, label=f"T = {t * config['time_interval_for_observation']} sec"
            )

    plt.title(title)
    plt.xlabel('yCoordinate')
    plt.ylabel('Rate of Change')

    # Add a color map legend
    cbar = plt.colorbar(orientation='vertical')
    cbar.set_label('Time (seconds)')

    # Save the graph to a file
    plt.savefig(f'{title}_graphssss.png')  # Replace 'graph2.png' with an appropriate filename
    plt.close()  # Close the figure to clear the plot





def display_graph_util(linePoints, frameNums, title, yCoordinatePoints, FPS, subplot1, subplot2):
    # # Displaying graph 1
    # plt.subplot(subplot1)
    # display_graph_1(linePoints, frameNums, title + '1')

    # Displaying graph 2
    plt.subplot(subplot2)
    display_graph_2(linePoints, frameNums, title + '2', yCoordinatePoints, FPS)


def display_graphs(linePointsLeft, linePointsRight, frameNums, yCoordinatePoints, FPS):
    # Displaying graph of left side
    # Inside your main function
    display_graph_2(linePointsLeft, frameNums, 'Left Side Graph', config['y_coordinate_points_for_analysis'], FPS)
    display_graph_2(linePointsRight, frameNums, 'Right Side Graph', config['y_coordinate_points_for_analysis'], FPS)

    plt.show()


def main(video_path):
    # Dataframe to store results
    resDf = pd.DataFrame(columns=['FrameNumber'] +
                                 [f'Y_{y}_leftDist' for y in config['y_coordinate_points_for_analysis']] +
                                 [f'Y_{y}_rightDist' for y in config['y_coordinate_points_for_analysis']])

    # Capturing the video
    cap = cv2.VideoCapture(video_path)

    # Checking if video is opened properly
    if not cap.isOpened():
        print("Error: Video file not found or cannot be opened.")
        return

    # Extracting FPS info
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    # Looping over each frame of the video
    frameNum = 0
    roiMaskImage = None
    while True:
        frameNum += 1

        # Reading the video frame
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Generating ROI Mask Image if not created
        if roiMaskImage is None:
            roiMaskImage = generate_roi_mask_image(frame.shape[0], frame.shape[1])

        # Processing the frame
        roiMaskImage, leftRightDistances, frameToShow = process_frame(frame, roiMaskImage)
        cv2.imshow("Video Frame", frame)
        cv2.imshow("Processed Frame", frameToShow)

        # Press 'q' to exit the video display
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            cv2.imwrite("temp.jpg", frame)

        # Storing the results
        data = {'FrameNumber': frameNum}
        for yCoordinate in config['y_coordinate_points_for_analysis']:
            data[f'Y_{yCoordinate}_leftDist'] = leftRightDistances[yCoordinate]['left']
            data[f'Y_{yCoordinate}_rightDist'] = leftRightDistances[yCoordinate]['right']
        resDf = resDf.append(data, ignore_index=True)

    # Releasing video and closing all screens
    cap.release()
    cv2.destroyAllWindows()

    # Storing the results
    resDf.to_csv("Results.csv", index=False)

    # Performing analysis
    linePointsLeft, linePointsRight, frameNums = perform_analysis(resDf, FPS,
                                                                  config['num_of_frames_to_skip_from_beginning'],
                                                                  config['y_coordinate_points_for_analysis'])

    # Displaying graphs
    display_graphs(linePointsLeft, linePointsRight, frameNums, config['y_coordinate_points_for_analysis'], FPS)


if __name__ == "__main__":
    # Getting the arguments
    args = ArgParse()

    main(args['filename'])


def display_graph_2(linePoints, frameNums, title, yCoordinatePoints, FPS):
    plt.figure(figsize=(12, 8))

    # Create a color map based on the frame numbers
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(frameNums), vmax=max(frameNums))

    for t, frameNum in enumerate(frameNums):
        if frameNum in linePoints:
            data = linePoints[frameNum]  # Extract the data for this frame
            x_data = [pt[0] for pt in data]  # Extract x-values
            y_data = [pt[1] for pt in data]  # Extract y-values
            colors = [frameNum] * len(x_data)  # Create a list of colors for each point
            plt.scatter(
                [yPointsMapping[y] for y in y_data],  # Convert y-values
                [x * config['1_pixel_in_mm_x_axis'] for x in x_data],  # Convert x-values
                c=colors, cmap=cmap, norm=norm, label=f"T = {t * config['time_interval_for_observation']} sec"
            )

    plt.title(title)
    plt.xlabel('yCoordinate')
    plt.ylabel('Rate of Change')

    # Add a color map legend
    cbar = plt.colorbar(orientation='vertical')
    cbar.set_label('Time (seconds)')

    # Save the graph to a file
    plt.savefig(f'{title}_graphssss.png')  # Replace 'graph2.png' with an appropriate filename
    plt.close()  # Close the figure to clear the plot






def display_graph_util(linePoints, frameNums, title, yCoordinatePoints, FPS, subplot1, subplot2):
    # # Displaying graph 1
    # plt.subplot(subplot1)
    # display_graph_1(linePoints, frameNums, title + '1')

    # Displaying graph 2
    plt.subplot(subplot2)
    display_graph_2(linePoints, frameNums, title + '2', yCoordinatePoints, FPS)


def display_graphs(linePoints, frameNums, yCoordinatePoints, FPS, side):
    for t, frameNum in enumerate(frameNums):
        if frameNum in linePoints:
            plt.figure(figsize=(12, 8))
            data = linePoints[frameNum]  # Extract the data for this frame
            x_data = [pt[0] for pt in data]  # Extract x-values
            y_data = [pt[1] for pt in data]  # Extract y-values
            plt.plot(
                [yPointsMapping[y] for y in y_data],  # Convert y-values
                [x * config['1_pixel_in_mm_x_axis'] for x in x_data],  # Convert x-values
                label=f"{side} Side - T = {frameNum * config['time_interval_for_observation']} sec"
            )
            plt.title(f'{side} Side Graph')
            plt.xlabel('yCoordinate')
            plt.ylabel('Rate of Change')

            # Save the graph to a file in the current working directory
            chart_filename = f'{side}_Graph_{t}.png'
            plt.savefig(chart_filename)
            plt.close()  # Close the figure to clear the plot

def main(video_path):
    # Dataframe to store results
    resDf = pd.DataFrame(columns=['FrameNumber'] +
                                 [f'Y_{y}_leftDist' for y in config['y_coordinate_points_for_analysis']] +
                                 [f'Y_{y}_rightDist' for y in config['y_coordinate_points_for_analysis']])

    # Capturing the video
    cap = cv2.VideoCapture(video_path)

    # Checking if the video is opened properly
    if not cap.isOpened():
        print("Error: Video file not found or cannot be opened.")
        return

    # Extracting FPS info
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    # Looping over each frame of the video
    frameNum = 0
    roiMaskImage = None
    while True:
        frameNum += 1

        # Reading the video frame
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Generating ROI Mask Image if not created
        if roiMaskImage is None:
            roiMaskImage = generate_roi_mask_image(frame.shape[0], frame.shape[1])

        # Processing the frame
        roiMaskImage, leftRightDistances, frameToShow = process_frame(frame, roiMaskImage)
        cv2.imshow("Video Frame", frame)
        cv2.imshow("Processed Frame", frameToShow)

        # Press 'q' to exit the video display
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            cv2.imwrite("temp.jpg", frame)

        # Storing the results
        data = {'FrameNumber': frameNum}
        for yCoordinate in config['y_coordinate_points_for_analysis']:
            data[f'Y_{yCoordinate}_leftDist'] = leftRightDistances[yCoordinate]['left']
            data[f'Y_{yCoordinate}_rightDist'] = leftRightDistances[yCoordinate]['right']
        resDf = resDf.append(data, ignore_index=True)

        # Check if it's time to generate and save charts
        if frameNum % int(FPS * config['time_interval_for_observation']) == 0:
            # Performing analysis for the current time interval
            linePointsLeft, linePointsRight, frameNums = perform_analysis(resDf, FPS,
                                                                          config['num_of_frames_to_skip_from_beginning'],
                                                                          config['y_coordinate_points_for_analysis'])



            # Displaying and saving graphs for left side
            display_graphs(linePointsLeft, frameNums, config['y_coordinate_points_for_analysis'], FPS, 'Left_Side')

            # Displaying and saving graphs for right side
            display_graphs(linePointsRight, frameNums, config['y_coordinate_points_for_analysis'], FPS, 'Right_Side')

    # Releasing video and closing all screens
    cap.release()
    cv2.destroyAllWindows()

    # Storing the results
    resDf.to_csv("Results.csv", index=False)


if __name__ == "__main__":
    # Getting the arguments
    args = ArgParse()

    main(args['filename'])
