import cv2
import time
from ultralytics import YOLO
import os

# Create a directory to save images if it doesn't exist
if not os.path.exists('person_images'):
    os.makedirs('person_images')

model = YOLO('best.pt')

# Variable to track timers for each person (using their bounding box positions)
person_timers = []  # List of dictionaries with 'bbox', 'start_time', and 'id'
min_distance = 50  # Minimum distance to consider the same person
movement_threshold = 20  # Distance threshold to ignore slight movements

def calculate_distance(box1, box2):
    """Calculate Euclidean distance between two bounding boxes."""
    x1, y1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2  # Center of box1
    x2, y2 = (box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2  # Center of box2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def log_person_exit(person_id, duration, exit_time, person_image):
    """Log the exit of a person to a text file and save their image."""
    timestamp = time.strftime('%Y%m%d-%H%M%S')  # Create a timestamp for the image filename
    image_filename = f"person_{person_id}_{timestamp}.jpg"  # Create a unique filename
    image_path = os.path.join('person_images', image_filename)  # Full path to save the image

    # Save the person's image
    cv2.imwrite(image_path, person_image)

    # Log the person's exit details and the image filename
    with open('person_exit_log.txt', 'a') as log_file:
        log_file.write(f"Person {person_id} left the screen after {duration:.2f} seconds at {exit_time:.2f} seconds. Image saved as {image_filename}\n")

def process_frame(frame):
    global person_timers  # Ensure we're working with the global person_timers
    
    current_time = time.time()
    
    # Perform person detection using YOLO
    results = model(frame)
    person_results = [r for r in results[0].boxes.data.tolist() if r[5] == 0]

    # List to keep track of which persons are currently in the frame
    currently_in_frame = []

    # For each detected person, check if it's a new person or an existing one
    for person in person_results:
        xmin, ymin, xmax, ymax, confidence, class_id = person
        if confidence > 0.5:
            new_bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            is_new = True

            # Check if the person already exists by comparing bounding boxes
            for pt in person_timers:
                # If they are close enough, treat them as the same person
                if calculate_distance(pt['bbox'], new_bbox) < min_distance:
                    # Update bounding box only if the movement is significant
                    if calculate_distance(pt['bbox'], new_bbox) >= movement_threshold:
                        pt['bbox'] = new_bbox  # Update bounding box for this person
                    currently_in_frame.append(pt)  # Keep track of existing persons in frame
                    is_new = False
                    break
            
            if is_new:
                # Add new person with their bounding box and timer
                person_timers.append({'bbox': new_bbox, 'start_time': current_time, 'id': len(person_timers) + 1})
                currently_in_frame.append(person_timers[-1])  # Keep track of the newly added person

    # Filter out people who are no longer in the frame
    for pt in person_timers[:]:  # Iterate over a copy of the list
        if pt not in currently_in_frame:
            # Capture the person's image before they leave
            xmin, ymin, xmax, ymax = pt['bbox']
            person_image = frame[ymin:ymax, xmin:xmax]  # Extract the bounding box region from the frame
            
            # Log the exit time and save the person's image
            duration = current_time - pt['start_time']
            exit_time = current_time
            log_person_exit(pt['id'], duration, exit_time, person_image)  # Log the exit and save the image
            person_timers.remove(pt)  # Remove from the timer list

    # Draw bounding boxes and display labels and timers
    for pt in person_timers:
        xmin, ymin, xmax, ymax = pt['bbox']
        duration = current_time - pt['start_time']
        person_label = f"Person: {duration:.2f} sec"  # Generic label for all persons
        
        # Draw the bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Display the person label and timer
        cv2.putText(frame, person_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)

    cv2.imshow('Real-time Person Detection with Timers', processed_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
