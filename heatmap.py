import cv2
import time
import numpy as np
import os
from ultralytics import YOLO

# Load YOLO and gender models
gender_net = cv2.dnn.readNetFromCaffe('E:/ML project/deploy_gender.prototxt', 'E:/ML project/gender_net.caffemodel')
model = YOLO('best.pt')

# Create directory to save exit images
if not os.path.exists('person_images'):
    os.makedirs('person_images')

# Timer and identity tracking
person_timers = []
min_distance = 50
movement_threshold = 20

def calculate_distance(box1, box2):
    x1, y1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    x2, y2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def log_person_exit(person_id, duration, exit_time, person_image):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    image_filename = f"person_{person_id}_{timestamp}.jpg"
    image_path = os.path.join('person_images', image_filename)
    cv2.imwrite(image_path, person_image)
    with open('person_exit_log.txt', 'a') as log_file:
        log_file.write(f"Person {person_id} left after {duration:.2f} seconds at {exit_time:.2f}. Saved as {image_filename}\n")

def process_frame(frame):
    global person_timers
    current_time = time.time()
    results = model(frame)
    person_results = [r for r in results[0].boxes.data.tolist() if r[5] == 0]
    people_coords = []

    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    currently_in_frame = []
    matched_ids = set()

    for person in person_results:
         xmin, ymin, xmax, ymax, confidence, class_id = person
         xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
         if confidence > 0.5:
             new_bbox = [xmin, ymin, xmax, ymax]
             matched = False
        
             for pt in person_timers:
                 if calculate_distance(pt['bbox'], new_bbox) < min_distance:
                     if calculate_distance(pt['bbox'], new_bbox) >= movement_threshold:
                         pt['bbox'] = new_bbox
                     pt['last_seen_image'] = frame[ymin:ymax, xmin:xmax]  # 🔥 Update image every frame
                     currently_in_frame.append(pt)
                     matched_ids.add(pt['id'])
                     matched = True
                     break
                    
             if not matched:
                 new_id = max([pt['id'] for pt in person_timers], default=0) + 1
                 person_crop = frame[ymin:ymax, xmin:xmax]
                 new_timer = {
                     'bbox': new_bbox,
                     'start_time': current_time,
                     'id': new_id,
                     'last_seen_image': person_crop
                 }
                 person_timers.append(new_timer)
                 currently_in_frame.append(new_timer)
                 matched_ids.add(new_id)
        
             people_coords.append((xmin, ymin, xmax, ymax))
        
             face = frame[ymin:ymax, xmin:xmax]
             if face.size > 0:
                 try:
                     face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104, 117, 123))
                     gender_net.setInput(face_blob)
                     gender_predictions = gender_net.forward()
                     male_confidence = gender_predictions[0][0]
                     female_confidence = gender_predictions[0][1]
                     gender = "Male" if male_confidence > female_confidence else "Female"
                     color = (255, 0, 0) if gender == "Male" else (203, 192, 255)
                     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                     cv2.putText(frame, gender, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                 except:
                     pass
                    
             heatmap[ymin:ymax, xmin:xmax] += 1

    for pt in person_timers[:]:
        if pt['id'] not in matched_ids:
            xmin, ymin, xmax, ymax = pt['bbox']
            person_image = frame[ymin:ymax, xmin:xmax]
            duration = current_time - pt['start_time']
            exit_time = current_time
            log_person_exit(pt['id'], duration, exit_time, person_image)
            person_timers.remove(pt)

    for pt in person_timers:
        xmin, ymin, xmax, ymax = pt['bbox']
        duration = current_time - pt['start_time']
        label = f"{duration:.1f} sec"

        # Determine position for timer 
        text_x = xmax - 30   
        text_y = ymin - 10

        # Draw timer text
        cv2.putText(frame, label, (text_x + 5, text_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)



    total_people = len(people_coords)
    cv2.putText(frame, f"Total Persons: {total_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if total_people >= 5:
        x_min = min([coord[0] for coord in people_coords])
        y_min = min([coord[1] for coord in people_coords])
        x_max = max([coord[2] for coord in people_coords])
        y_max = max([coord[3] for coord in people_coords])
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, "Crowd Detected", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

    return blended

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Real-time Detection (Timers + Gender + Heatmap)', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
