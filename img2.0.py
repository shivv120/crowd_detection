import cv2
from ultralytics import YOLO
from tkinter import Tk, Button, Label, filedialog, font
from PIL import Image, ImageTk

# Load the YOLOv8n model
model = YOLO('best.pt')

# Process the image using YOLO model
def process_image(image):
    results = model(image)
    person_results = [r for r in results[0].boxes.data.tolist() if r[5] == 0]
    people_coords = []

    for person in person_results:
        xmin, ymin, xmax, ymax, confidence, class_id = person
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        if confidence > 0.3:
            people_coords.append((xmin, ymin, xmax, ymax))
            # Draw rectangle around the person
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    total_people = len(people_coords)
    # Adjust the position and font size of the text
    cv2.putText(image, f"Total Persons: {total_people}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Detect crowd if total people are 10 or more
    if total_people >= 10:
        x_min = min([coord[0] for coord in people_coords])
        y_min = min([coord[1] for coord in people_coords])
        x_max = max([coord[2] for coord in people_coords])
        y_max = max([coord[3] for coord in people_coords])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(image, "Crowd Detected", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return image


# Function to open and process the image
def open_image():
    file_path = filedialog.askopenfilename(title="Upload an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image = cv2.imread(file_path)
        processed_image = process_image(image)

        # Convert image to RGB format for displaying in Tkinter
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(processed_image)

        # Resize the image to fit the window while maintaining aspect ratio
        screen_width = root.winfo_width()
        screen_height = root.winfo_height() - 150  # Adjusting for header and button space
        im_width, im_height = im_pil.size
        scale = min(screen_width / im_width, screen_height / im_height)
        new_width = int(im_width * scale)
        new_height = int(im_height * scale)
        resized_image = im_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Updated resizing method

        imgtk = ImageTk.PhotoImage(image=resized_image)

        # Update label with the new image
        img_label.config(image=imgtk)
        img_label.image = imgtk

# Set up the main application window
root = Tk()
root.title("YOLO Crowd Detection")
root.state('zoomed')  # Maximize window without full-screen, keeping window controls
root.config(bg='#ffcb96')  # Set background color to #ffcb96

# Custom font for the button and heading
button_font = font.Font(family='Helvetica', size=16, weight='bold')
heading_font = font.Font(family='Helvetica', size=24, weight='bold')

# Heading for the application
heading_label = Label(root, text="Detect a Crowd", font=heading_font, bg='#ffcb96', fg='#333333')
heading_label.pack(pady=20)

# Button to upload image
upload_btn = Button(root, text="Upload an Image", command=open_image, font=button_font, bg="#FFA07A", fg="white", padx=20, pady=10, relief="flat", borderwidth=0)
upload_btn.pack(pady=20)

# Label to display the image
img_label = Label(root, bg='#ffcb96')
img_label.pack()

# Run the Tkinter event loop
root.mainloop()
