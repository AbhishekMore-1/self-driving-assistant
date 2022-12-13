# -*- coding: utf-8 -*-
# Import required libraries

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):  
    x_left = []
    y_left = []
    
    x_right = []
    y_right = []
    
    shape = img.shape

    for line in lines:
        for x1,y1,x2,y2 in line:
            
            slope = -(y2 - y1) / (x2 - x1)
            
            # Collecting all x and y values from left part of image 
            if x1 < shape[1]/2 and x2 < shape[1]/2:
                if 0.25 < slope < 0.8:
                    x_left += [x1, x2]
                    y_left += [y1, y2]
            
            # Collecting all x and y values from right part of image
            else:
                if -0.8 < slope < -0.25:
                    x_right += [x1, x2]
                    y_right += [y1, y2]
            
    if len(x_left) != 0 and len(y_left) != 0:
        # Calculating average slope and b from all the points on left - here slope will be negative
        m,b = np.polyfit(x_left, y_left, 1)
        min_x = min(x_left)
        max_x = max(x_left)
        # Plotting line from x1=(y_max-b)/m, y1=y_max to x2=x_max, y2=m*x_max + b
        cv2.line(img, (int((shape[0]-b)/m), shape[0]), (max_x, int(max_x * m + b)), color, thickness)

    if len(x_right) != 0 and len(y_right) != 0:
        # Calculating average slope and b from all the points on right - here slope will be positive
        m,b = np.polyfit(x_right, y_right, 1)
        min_x = min(x_right)
        max_x = max(x_right)
        # Plotting line from x1=x_min, y1=(x_min*m)+b to x2=(y_max-b)/m, y2=y_max
        cv2.line(img, (min_x, int(min_x * m + b)), (int((shape[0]-b)/m), shape[0]), color, thickness)

        
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image):
    color_select= np.copy(image)
    line_image= np.copy(image)
    
    gray = grayscale(color_select)
    
    ysize = image.shape[0]
    xsize = image.shape[1]
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray,kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150

    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = color_select.shape
    vertices = np.array([[(50,imshape[0]),(450, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    #draw_lines(line_image, lines, color=[255, 0, 0], thickness=2)
    
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    return weighted_img(line_image,image)

# Pre-process Image
def preProcessImg(img):
    # Pre-processing image: resize image
    height, width, _ = img.shape
    width = int(480/height*width)
    height = 480
    img = cv2.resize(img,(width,height))
    return img

# Upload Image
def uploadImage(key):

    uploaded_file = st.file_uploader("Choose a Image file",key=key)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # Pre-processing image: resize image
        return preProcessImg(img)
    
    return cv2.cvtColor(preProcessImg(cv2.imread('img.jpg')),cv2.COLOR_BGR2RGB)

def lane_detection():
    image = uploadImage(1)
    h = image.shape[0]
    w = image.shape[1]
    image = cv2.resize(process_image(cv2.resize(image,(960,540))),(w,h))
    st.image(image, use_column_width=True)

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Set wide view for page
    st.set_page_config(layout="wide")
    
    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Select Dropbox
    app_mode = st.sidebar.selectbox("Select section",
        ['About Team','Object Detection','Lane Detection','Final Detection'])
    if app_mode == "About Team":
        aboutTeam()
    elif app_mode == "Object Detection":
        objDetection()
    elif app_mode == "Lane Detection":
        lane_detection()
    elif app_mode == "Final Detection":
        final_detection()

def final_detection():
    video_file = open('project_2.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# About Team UI 
def aboutTeam():
    st.sidebar.success('To continue select "Object/Lane Detection".')
    # Render the intro as markdown using st.markdown.
    st.markdown(open("intro.md").read(),unsafe_allow_html=True)

# This is the main app app itself, which appears when the user selects "Run the app".
def objDetection():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.experimental_memo
    def load_metadata(url):
        return pd.read_csv(url)

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.experimental_memo
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
            "label_biker": "biker",
            "label_car": "car",
            "label_pedestrian": "pedestrian",
            "label_trafficLight": "traffic light",
            "label_truck": "truck"
        })
        return summary

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
    summary = create_summary(metadata)

    # Uncomment these lines to peek at these DataFrames.
    # st.write('## Metadata', metadata[:1000], '## Summary', summary[:1000])

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load the image from S3.
    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    image = load_image(image_url)

    original_img, _, processed_img = st.columns([1,0.01,1])

    with original_img:
        # Add boxes for objects on the image. These are the boxes for the ground image.
        boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
        draw_image_with_boxes(image, boxes, "Ground Truth",
            "**Human-annotated data** (frame `%i`)" % selected_frame_index)

    with processed_img:
        # Get the boxes for the objects detected by YOLO by running the YOLO model.
        yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)
        draw_image_with_boxes(image, yolo_boxes, "Real-time Computer Vision",
            "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)" % (overlap_threshold, confidence_threshold))

# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(summary):
    st.sidebar.markdown("# Frame")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)

    # The user can select a range for how many of the selected objecgt should be present.
    min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [10, 20])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
        return None, None

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)

    # Draw an altair chart in the sidebar with information on the frame.
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % object_type))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x = "selected_frame")
    st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame

# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = {
        "car": [255, 0, 0],
        "pedestrian": [0, 255, 0],
        "truck": [0, 0, 255],
        "trafficLight": [255, 255, 0],
        "biker": [255, 0, 255],
    }
    image_with_boxes = image
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes = cv2.rectangle(image_with_boxes,(xmin,ymax),(xmax,ymin), LABEL_COLORS[label],1)
        # image_with_boxes = cv2.putText(image_with_boxes, label, (ymin,xmin), cv2.FONT_HERSHEY_SIMPLEX, 1, LABEL_COLORS[label], 1, cv2.LINE_AA)

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.experimental_memo(show_spinner=False)
def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

# Run the YOLO model to detect objects.
def yolo_v3(image, confidence_threshold, overlap_threshold):
    # Load the network. Because this is cached it will only happen once.
    @st.cache(allow_output_mutation=True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

    # Run the YOLO neural net.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # Supress detections in case of too low confidence or too much overlap.
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

    # Map from YOLO labels to Udacity labels.
    UDACITY_LABELS = {
        0: 'pedestrian',
        1: 'biker',
        2: 'car',
        3: 'biker',
        5: 'truck',
        7: 'truck',
        9: 'trafficLight'
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = UDACITY_LABELS.get(class_IDs[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]

# Path to the Streamlit public S3 bucket
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

if __name__ == "__main__":
    main()
