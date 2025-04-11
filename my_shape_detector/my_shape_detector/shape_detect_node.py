import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from shape_info.msg import ShapeInfo
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# Image properties
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# ROS 2 Node Class
class ShapeDetectNode(Node):
    def __init__(self):
        super().__init__('shape_detect_node')
        
        # Initialize variables
        self.fps = 0
        self.triangle_area_threshold = 500  # Added missing threshold for triangles

        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Create publishers
        self.image_pub = self.create_publisher(Image, 'processed_image', 10)
        self.shape_pub = self.create_publisher(ShapeInfo, 'shape_info', 10)
        self.fps_pub = self.create_publisher(Float32, 'fps', 10)

        # Create a timer to call the processing function periodically
        self.timer = self.create_timer(0.03, self.process_and_publish)
        
        # OpenCV Video Capture
        self.camera = cv2.VideoCapture(2)
        self.camera.set(cv2.CAP_PROP_FPS, 60)  # Set FPS
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE,1)  # Set buffer size
        self.set_camera_properties(self.camera)

    def set_camera_properties(self, camera):
        """ Set resolution properties for the webcam """
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    def capture_frame(self, camera):
        """ Capture a frame from the video source and convert it to BGR """
        ret, frame = camera.read()
        if not ret:
            self.get_logger().error("Error: Failed to capture image.")
            return None
        return frame


    def calculate_distance(self,pointcenter, point1):
        """ Calculate the distance between two points """
        x = point1[1] - pointcenter[1]
        y = -point1[0] + pointcenter[0]
        return x, y

    def detect_blue_regions(self, frame):
        """
        Detect blue regions in the image using HSV color space,
        then calculate the min enclosing circle for those regions.
        """
        # Convert from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define thresholds for blue color
        lower_blue = np.array([95, 80, 60])
        upper_blue = np.array([125, 255, 255])
        
        # Create mask for blue regions
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours from the mask
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blue_circles = []  # List to store blue circles

        for cnt in contours:
            # Only process regions with sufficient area
            if cv2.contourArea(cnt) > 1000:
                (x, y), radius = cv2.minEnclosingCircle(cnt)  # Calculate enclosing circle
                blue_circles.append([x, y, radius])  # Add circle info to the list

        if len(blue_circles) > 0:
            blue_circles = np.array(blue_circles, dtype=np.float32)  # Convert list to numpy array
        else:
            blue_circles = None  # No circles detected

        return blue_circles, mask_blue

    def detect_red_regions(self, frame):
        """
        Detect red regions in the image using HSV color space,
        then calculate the min enclosing circle for those regions.
        """
        # Convert from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define thresholds for red color (red wraps around the hue spectrum)
        lower_red1 = np.array([0, 100, 100])    # Lower threshold for red
        upper_red1 = np.array([10, 255, 255])   # Upper threshold for red
        lower_red2 = np.array([160, 100, 100])  # Lower threshold for red (end of color wheel)
        upper_red2 = np.array([180, 255, 255])  # Upper threshold for red (end of color wheel)
        
        # Create mask for red regions
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # Combine both masks

        # Find contours from the mask
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        red_circles = []  # List to store red circles

        for cnt in contours:
            # Only process regions with sufficient area
            if cv2.contourArea(cnt) > 1000:
                (x, y), radius = cv2.minEnclosingCircle(cnt)  # Calculate enclosing circle
                red_circles.append([x, y, radius])  # Add circle info to the list

        if len(red_circles) > 0:
            red_circles = np.array(red_circles, dtype=np.float32)  # Convert list to numpy array
        else:
            red_circles = None  # No circles detected

        return red_circles, mask_red

    def detect_yellow_regions(self, frame):
        """
        Detect yellow regions in the image using HSV color space,
        then calculate the min enclosing circle for those regions.
        """
        # Convert from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define thresholds for yellow color
        lower_yellow = np.array([20, 100, 100])  # Lower threshold for yellow
        upper_yellow = np.array([30, 255, 255])  # Upper threshold for yellow
        
        # Create mask for yellow regions
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yellow_circles = []  # List to store yellow circles

        for cnt in contours:
            # Only process regions with sufficient area
            if cv2.contourArea(cnt) > 1000:
                (x, y), radius = cv2.minEnclosingCircle(cnt)  # Calculate enclosing circle
                yellow_circles.append([x, y, radius])  # Add circle info to the list

        if len(yellow_circles) > 0:
            yellow_circles = np.array(yellow_circles, dtype=np.float32)  # Convert list to numpy array
        else:
            yellow_circles = None  # No circles detected

        return yellow_circles,mask_yellow

    def detect_white_regions(self,frame):
        """
        Phát hiện vùng màu trắng trong hình ảnh sử dụng không gian màu HSV,
        sau đó tính toán hình tròn bao quanh (min enclosing circle) cho các vùng đó.

        Parameters:
            frame (ndarray): Hình ảnh gốc từ video (BGR).

        Returns:
            white_circles (ndarray or None): Một mảng numpy với kích thước (N, 3),
                                            trong đó mỗi hàng chứa [x, y, r] của vòng tròn bao quanh vùng màu trắng.
                                            Nếu không phát hiện được vùng nào, trả về None.
            mask_white (ndarray): Mặt nạ nhị phân biểu diễn vùng trắng trong ảnh.
        """
        # Chuyển đổi từ BGR sang HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Định nghĩa ngưỡng cho màu trắng (sáng cao, bão hòa thấp)
        lower_white = np.array([0, 0, 200])     # Giới hạn thấp cho màu trắng
        upper_white = np.array([180, 40, 255])  # Giới hạn cao cho màu trắng

        # Tạo mặt nạ cho vùng màu trắng
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Tìm các contour từ mặt nạ
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        white_circles = []  # Danh sách lưu trữ các vòng tròn màu trắng

        for cnt in contours:
            # Chỉ xử lý những vùng có diện tích đủ lớn
            if cv2.contourArea(cnt) > 1000:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                white_circles.append([x, y, radius])

        if len(white_circles) > 0:
            white_circles = np.array(white_circles, dtype=np.float32)
        else:
            white_circles = None

        return white_circles, mask_white

    def process_frame(self, frame):
        """ Blur, convert to grayscale and detect circles """
        frame_blur = cv2.blur(frame, (3, 3))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blue_circles, mask_blue = self.detect_blue_regions(frame)
        red_circles, mask_red = self.detect_red_regions(frame)
        yellow_circles, mask_yellow = self.detect_yellow_regions(frame)
        white_circles, mask_white = self.detect_white_regions(frame)
        return frame_blur, gray, mask_blue, mask_red, mask_yellow, mask_white, blue_circles, red_circles, yellow_circles, white_circles

    def visualize(self, image, text, unit, row_size, color, numberdis):
        """ Overlay the number_dis value onto the given image. """
        if color == 'green':
            color = (0, 255, 0)
        elif color == 'brown':
            color = (42, 42, 165)
        elif color == 'pink':
            color = (255, 25, 255)
        elif color == 'blue':
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        
        left_margin = 24  # pixels
        font_size = 1
        font_thickness = 1
        
        numberdis_text = text + ': {:.1f}'.format(numberdis) + unit
        text_location = (left_margin, row_size)
        cv2.putText(image, numberdis_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, color, font_thickness)

        return image

    def get_color_name(self, r, g, b):
        """ Returns the accurate color name based on RGB values by converting to HSV """
        color_bgr = np.uint8([[[b, g, r]]])  # OpenCV uses BGR
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv  # Hue, Saturation, Value

        # Determine color by brightness first
        if v < 50:
            return "Black"
        elif v > 200 and s < 50:
            return "White"
        elif s < 30:
            return "Gray"

        # Determine color based on Hue
        if (0 <= h < 5) or (h >= 175): 
            return "Red"
        elif 5 <= h < 20:  
            return "Orange"
        elif 20 <= h < 40:  
            return "Yellow"
        elif 40 <= h < 85:  
            return "Green"
        elif 85 <= h < 120:  
            return "Cyan"
        elif 120 <= h < 160:  
            return "Blue"
        elif 160 <= h < 175:  
            return "Magenta"

        return "Unknown"

    def get_dominant_color_name(self, roi):
        """ Return the most dominant color name in the ROI area """
        if roi.size == 0:
            return "Unknown", (0, 0, 0)
            
        roi_small = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
        pixels = roi_small.reshape(-1, 3)
        avg_color = np.mean(pixels, axis=0)
        r, g, b = int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
        color_name = self.get_color_name(r, g, b)  # Now get_color_name returns just the color name
        return color_name, (b, g, r)  # Return color name and BGR values

    def get_circle_shape_id(self, color_name):
        """Assign shape_id for circles based on color"""
        color_name = str(color_name).lower()  # Ensure it's a string and convert to lowercase
        if "yellow" in color_name or "green" in color_name:
            return 1
        elif "red" in color_name or "orange" in color_name:
            return 2
        elif "blue" in color_name or "magenta" in color_name or "cyan" in color_name:
            return 3
        else:
            return -1  # Default ID for unrecognized colors


    def draw_circles_on_frame(self, frame, circles, blue_circles=None, red_circles=None, yellow_circles=None):
        """ Draw the largest circle and display the dominant color """
        output = frame.copy()
        center_x = output.shape[1] // 2
        center_y = output.shape[0] // 2
        cv2.circle(output, (center_x, center_y), 4, (255, 255, 200), -1)  # Frame center

        all_circles = []  # List to store all circles
        shape_info = None

        # Add circles from Hough Circle to the list
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            all_circles.extend(circles)

        # # Add circles from blue regions to the list
        # if blue_circles is not None:
        #     blue_circles = np.round(blue_circles).astype("int")
        #     all_circles.extend(blue_circles)

        # Add circles from red regions to the list
        if red_circles is not None:
            red_circles = np.round(red_circles).astype("int")
            all_circles.extend(red_circles)

        # Add circles from yellow regions to the list
        if yellow_circles is not None:
            yellow_circles = np.round(yellow_circles).astype("int")
            all_circles.extend(yellow_circles)

        # Find the largest circle
        largest_circle = None
        largest_radius = 0

        for (x, y, r) in all_circles:
            if r > largest_radius:
                largest_radius = r
                largest_circle = (x, y, r)

        # Draw the largest circle if found
        if largest_circle is not None:
            x, y, r = largest_circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)  # Draw largest circle in green
            cv2.circle(output, (x, y), 4, (0, 0, 255), -1)  # Draw center of the circle in red

            # Draw direction lines
            cv2.line(output, (center_x, center_y), (x, y), (0, 0, 0), 2)  # Black line
            cv2.line(output, (center_x, center_y), (center_x, y), (42, 42, 165), 2)  # Brown vertical line
            cv2.line(output, (x, y), (center_x, y), (255, 0, 0), 2)  # Blue horizontal line
            cv2.circle(output, (center_x, y), 4, (0, 0, 0), -1)  # Draw intersection point in black

            # Calculate distances
            x11, y11 = self.calculate_distance((center_x, center_y), (x, y))
            x22, y22 = self.calculate_distance((center_x, center_y), (center_x, y))

            # Hiển thị khoảng cách
            self.visualize(output, 'x1', 'px', 40, 'pink', x11)
            self.visualize(output, 'y1', 'px', 70, 'pink', y11)

            self.visualize(output, 'x2', 'px', 100, 'brown', x22)
            self.visualize(output, 'y2', 'px', 120, 'brown', y22)


            # Create mask and get ROI for color analysis
            mask = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            masked_frame = cv2.bitwise_and(output, output, mask=mask)
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(output.shape[1], x + r), min(output.shape[0], y + r)
            roi = masked_frame[y1:y2, x1:x2]

            color_name, dominant_bgr = self.get_dominant_color_name(roi)  # Now get_dominant_color_name returns 2 values

            # Assign shape_id based on color    
            shape_id = self.get_circle_shape_id(color_name)
            self.get_logger().info(f'Detected circle color: {color_name}')
            # Display color name
            cv2.putText(output, f"Circle - {color_name}", (x - 40, y - r - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, dominant_bgr, 2)
                        
            # Create ShapeInfo message using the correct message definition
            shape_info = ShapeInfo()
            shape_info.has_shape = True
            shape_info.shape_id = shape_id
            shape_info.x = float(x11)
            shape_info.y = float(y11)
            self.get_logger().info('Shape detected: circle')
        return output, shape_info



    def detect_and_annotate_triangles(self, frame):
        """ Draw the triangles and display the dominant color """
        output = frame.copy()
        # imgblur = cv2.blur(frame, (3, 3))
        # imggray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(imggray, 50, 150)
        # edges = cv2.dilate(edges, np.ones((5, 5)), iterations=1)

        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Change this line:
        _, mask_white = self.detect_white_regions(frame)  # Added self.
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = output.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        triangles = []  # Store triangle data for publishing
        shape_info = None
        
        # Find the largest triangle
        largest_triangle = None
        largest_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.triangle_area_threshold:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) == 3:
                    x,y,w_rect,h_rect = cv2.boundingRect(approx)


                    # Calculate triangle area
                    M = np.mean(approx.reshape(-1, 2), axis=0).astype(int)
                    cx, cy = M[0], M[1]
                    cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)  # Draw center
                    cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

                    # Draw direction indicators

                    cv2.line(output, (center_x, center_y), (cx, cy), (0, 0, 0), 2)
                    cv2.line(output, (center_x, center_y), (center_x, cy), (42, 42, 165), 2)
                    cv2.line(output, (cx, cy), (center_x, cy), (255, 0, 0), 2)
                    cv2.circle(output, (center_x, cy), 4, (0, 0, 0), -1)

                    # Calculate distances
                    x11, y11 = self.calculate_distance((center_x, center_y), (x, y))
                    x22, y22 = self.calculate_distance((center_x, center_y), (center_x, y))

                    # Hiển thị khoảng cách
                    self.visualize(output, 'D1', 'px', 40, 'pink', x11)
                    self.visualize(output, 'D1', 'px', 50, 'pink', y11)

                    self.visualize(output, 'D2', 'px', 60, 'brown', x22)
                    self.visualize(output, 'D2', 'px', 70, 'brown', y22)
    

                    # Create mask and get ROI
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [approx], -1, 255, -1)
                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                    roi = masked_frame[y:y + h_rect, x:x + w_rect]
                    color_name, dominant_bgr = self.get_dominant_color_name(roi)  # Changed this line
                    # Display color name
                    cv2.putText(output, f"Triangle - {color_name}", (x, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, dominant_bgr, 2)
                    shape_info = ShapeInfo()
                    shape_info.has_shape = True
                    shape_info.shape_id = 0
                    shape_info.x = float(x11)
                    shape_info.y = float(y11)
                    self.get_logger().info('Shape detected: triangle')
                    # Store triangle data
        return output, triangles, shape_info

    def anisotropic_diffusion(self, img, num_iter=3, kappa=30, gamma=0.1):
        """
        Function to perform Anisotropic Diffusion on an image.
        """
        img = img.astype(np.float32) / 255.0  # Convert to float
        for _ in range(num_iter):
            # Calculate gradient
            dx = np.roll(img, -1, axis=1) - img
            dy = np.roll(img, -1, axis=0) - img
            # Calculate coefficients
            c = np.exp(-(dx ** 2 + dy ** 2) / kappa)
            # Update image
            img += gamma * (c * (dx + dy))

        return (img * 255).astype(np.uint8)

    def combine_images_horizontally(self, img_list):
        """
        Combine images horizontally.
        Convert all images to the same size, 3 channels, and same dtype to avoid hconcat errors.
        """
        resized_imgs = []
        h, w = img_list[0].shape[:2]
        dtype = img_list[0].dtype

        # Changed this line:
        for img in img_list:  # Removed 'i,' from the loop
            # If it's a grayscale image (1 channel), convert to 3 channels
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Resize and convert dtype if needed
            resized = cv2.resize(img, (w, h))
            if resized.dtype != dtype:
                resized = resized.astype(dtype)
            resized_imgs.append(resized)

        combined = cv2.hconcat(resized_imgs)
        return combined

    def process_and_publish(self):
        try:
            start_time = time.time()
            """ Capture a frame, process it, and publish the result """
            frame = self.capture_frame(self.camera)
            if frame is None:
                return
                    
            # Perform anisotropic diffusion to reduce glare
            processed_color_frame = self.anisotropic_diffusion(frame)
            
            # Process frame to detect shapes (circles, triangles)
            blur_frame, gray, mask_blue, mask_red, mask_yellow, mask_white, _, red_circles, yellow_circles, white_circles = self.process_frame(processed_color_frame)

            # Detect and annotate circles
            output, circle_info = self.draw_circles_on_frame(blur_frame, None, _, red_circles, yellow_circles)

            # Detect and annotate triangles
            # final_output, triangles, triangle_info = self.detect_and_annotate_triangles(output)

            # Calculate FPS
            end_time = time.time()
            seconds = end_time - start_time
            self.fps = 1.0 / seconds if seconds > 0 else 0

            # Display FPS on the image
            output = self.visualize(output, 'FPS', 'hz', 20, 'green', self.fps)
            
            # Convert to ROS Image message and publish
            image_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
            self.image_pub.publish(image_msg)
            
            # Publish ShapeInfo message (prioritize triangle over circle if both exist)
            if circle_info is not None:
                self.shape_pub.publish(circle_info)  # circle
                self.get_logger().info('Publishing circle info')
            else:
                # Create an empty shape info message
                shape_info = ShapeInfo()
                shape_info.has_shape = False
                shape_info.shape_id = -1
                shape_info.x = 0.0
                shape_info.y = 0.0
                self.shape_pub.publish(shape_info)
                self.get_logger().info('No shape detected')  
        except Exception as e:
            self.get_logger().error(f"Error in processing: {e}")

def main(args=None):
    rclpy.init(args=args)

    # Create the node
    shape_detect_node = ShapeDetectNode()

    # Spin the node to keep it alive and processing
    rclpy.spin(shape_detect_node)

    # Shutdown ROS 2 when finished
    shape_detect_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()