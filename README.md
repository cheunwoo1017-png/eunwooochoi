✊✌️🖐️ Rock Paper Scissors — Gesture Recognition Game
A real-time hand gesture recognition game where you play Rock Paper Scissors against the computer using your webcam. Inspired by non-traditional human-computer interaction (like voice-controlled games such as Mage Arena, where saying "fireball" casts a spell), this project explores how we can interact with computers beyond keyboard and mouse — using our body as input.

🎯 Motivation
Games like Mage Arena let players cast spells by speaking out loud — "fireball" triggers a fireball in-game. This kind of natural, body-based interaction is what inspired this project. Instead of pressing a key to choose rock, paper, or scissors, you simply show your hand to the webcam. The AI reads your gesture and plays against you.
This is one step toward richer, more intuitive human-computer interfaces.

🕹️ How It Works

Your webcam captures a live video feed
Each frame is resized to 224×224 and normalized
A Teachable Machine model (Keras .h5) predicts your gesture
The computer randomly picks rock, paper, or scissors
The winner of each round is determined and scores are tracked over 5 rounds


📁 File Structure
rps-game/

├── main.py           # Main game loop

├── keras_model.h5    # Trained Teachable Machine model

└── labels.txt        # Class labels (scissors / rock / paper)

🧠 Model & Dataset
ItemDetailModelKeras .h5 (trained via Google Teachable Machine)Input224×224 RGB image from webcamClassesScissors, Rock, PaperDataset size~1,000 images total
Data distribution:

✌️ Scissors: ~600 images
✊ Rock: ~200 images
🖐️ Paper: ~200 images


Note: The dataset is imbalanced. Scissors has 3× more training data, which may affect prediction confidence across classes.


▶️ How to Run
1. Install dependencies
bashpip install tensorflow opencv-python numpy
2. Prepare model files
Place keras_model.h5 and labels.txt (exported from Teachable Machine) in the same directory as main.py.
3. Run the game
bashpython main.py
4. Controls
KeyActionSPACELock in your move and judge the roundQQuit the game

🔬 Significance
This project sits at the intersection of computer vision and natural interaction design. Rather than mapping physical buttons to game actions, the system reads the user's body language directly — a gesture becomes a command.
Key takeaways:

Accessible input: Gesture-based control can benefit users who find traditional input devices difficult
Teachable Machine as a prototyping tool: Enables rapid training of custom vision models without deep ML expertise
Imbalanced data matters: The uneven dataset (600/200/200) is a concrete lesson in how data distribution affects model behavior
Edge toward embodied AI: This is a small but real step toward AI systems that understand human physical expression


🛠️ Tech Stack

-Language: Python

-Model: TensorFlow / Keras (.h5)

-Vision: OpenCV

-Training: Google Teachable Machine

-Input: Laptop webcam


👥 Authors

Eunwoo Choi (20260464)
Jiwon kim (20260957)

🔄 Revision (v1.1)

Applied feedback: win/loss result and score now displayed directly on the webcam window in real time
Future improvement: apply object segmentation to remove background noise and improve gesture recognition accuracy
