from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Initialize YOLO models
model1 = YOLO('yolo11l.pt')
model2 = YOLO('best.pt')

# Video capture source (use a video file path or webcam index)
video_source = "TV.mp4"  # 0 for webcam, or replace with video file path

# Video capture
cap = cv2.VideoCapture(video_source)

# Video streaming generator
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run YOLOv8 inference on the frame
            results1 = model1(frame)  # Use model1 for inference

            # Access detection results
            boxes = results1[0].boxes.xyxy  # [x1, y1, x2, y2]
            confidences = results1[0].boxes.conf  # Confidence scores
            classes = results1[0].boxes.cls  # Class labels

            # Annotate the frame with detections
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                confidence = confidences[i]
                class_id = int(classes[i])
                label = f"{class_id}: {confidence:.2f}"
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Render HTML template for video display

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')

