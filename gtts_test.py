import os
from gtts import gTTS

# Malayalam text to convert to speech
# text = "എറണാകുളത്തേക്കുള്ള ബസ് എത്തി"
# text = "കൊട്ടാരക്കര ഭാഗത്തേക്കുള്ള ബസ് എത്തിച്ചേർന്നിരിക്കുന്നു"
# text = "തൃശൂർ കൊച്ചി അങ്കമാലി കൊല്ലം"
text = "തൊടുപുഴ ഭാഗത്തേക്കുള്ള ബസ്സ് എത്തിച്ചേർന്നിരിക്കുന്നു"

# Create a gTTS object with the Malayalam language code ('ml')
tts = gTTS(text=text, lang="ml")

# Specify the output directory where you want to save the audio file
output_dir = "C:/PROJECT/audio/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the output file path
output_file = os.path.join(output_dir, "output.mp3")

# Save the generated speech to the output file
tts.save(output_file)
print(f"Audio saved to: {output_file}")

# Play the audio file (for Windows; use an alternative command on Linux/macOS)
os.system(f"start {output_file}")
