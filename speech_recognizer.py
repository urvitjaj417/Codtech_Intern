import speech_recognition as sr
import logging
import argparse
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def recognize_audio(file_path=None, use_cloud=False, cloud_key=None):
    recognizer = sr.Recognizer()

    # Use microphone if no file provided
    if file_path is None:
        with sr.Microphone() as source:
            print("Please speak clearly into the microphone...")
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased duration
            logging.info("Listening for audio input from microphone")
            audio = recognizer.listen(source, phrase_time_limit=20)
    else:
        with sr.AudioFile(file_path) as source:
            logging.info(f"Loading audio file: {file_path}")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)

    try:
        logging.info("Transcribing audio...")
        if use_cloud and cloud_key:
            text = recognizer.recognize_google_cloud(audio, credentials_json=cloud_key)
        else:
            text = recognizer.recognize_google(audio, show_all=False)
        logging.info("Transcription successful")
        print("\n📝 Transcribed Text:\n")
        print(text)
    except sr.UnknownValueError:
        logging.error("Speech was unintelligible")
        print("Could not understand audio. Please speak clearly and ensure your microphone is working.")
    except sr.RequestError as e:
        logging.error(f"API error: {e}")
        print(f"Could not request results; {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Recognition System")
    parser.add_argument('--file', type=str, help='Path to audio file (.wav)')
    parser.add_argument('--cloud', action='store_true', help='Use Google Cloud Speech API (requires credentials)')
    parser.add_argument('--cloud_key', type=str, help='Path to Google Cloud credentials JSON')
    args = parser.parse_args()

    cloud_key = None
    if args.cloud and args.cloud_key:
        with open(args.cloud_key, 'r') as f:
            cloud_key = f.read()

    recognize_audio(file_path=args.file, use_cloud=args.cloud, cloud_key=cloud_key)
