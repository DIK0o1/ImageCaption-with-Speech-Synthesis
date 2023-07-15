import pickle
import cv2
import pyttsx3

model = pickle.load(open("imageCaption/model.bin", 'rb'))
tokenizer = pickle.load(open("imageCaption/tokenizer.bin", 'rb'))
image_processor = pickle.load(open("imageCaption/image_processor.bin", 'rb'))


# A function to perform inference
def get_caption(model, image_processor, tokenizer):
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    image = cv2.flip(image, 0)
    img = image_processor(image, return_tensors="pt")

    # Generate the caption (using greedy decoding by default)
    output = model.generate(**img)

    # Decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    cap.release()
    return caption


# Function to convert caption to speech
def caption_to_speech(caption):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust the speech rate (optional)
    engine.say(caption)
    engine.runAndWait()


# Get the caption
caption = get_caption(model, image_processor, tokenizer)
print("Caption:", caption)

# Convert the caption to speech
caption_to_speech(caption)
