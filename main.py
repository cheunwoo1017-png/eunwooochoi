# Rock Paper Scissors Game - AI Model
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import random

# 1. Load Model
def load_rps_model():
    model = load_model("keras_model.h5", compile=False)
    labels = open("labels.txt", "r").readlines()
    return model, labels

# 2. Preprocess Webcam Image
def preprocess_frame(frame):
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

# 3. Judge Winner
def judge(player, computer):
    if player == computer:
        return "draw"
    win_cases = {"scissors": "paper", "rock": "scissors", "paper": "rock"}
    if win_cases[player] == computer:
        return "player wins!"
    return "computer wins!"

# Main Game Loop
def main():
    model, labels = load_rps_model()
    choices = ["scissors", "rock", "paper"]
    
    cap = cv2.VideoCapture(0)
    score = {"player": 0, "computer": 0}
    total_rounds = 5

    print("=== Rock Paper Scissors Game Start! ===")
    print("SPACE: judge | q: quit\n")

    for round_num in range(1, total_rounds + 1):
        print(f"--- Round {round_num} ---")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # AI prediction
            processed = preprocess_frame(frame)
            prediction = model.predict(processed, verbose=0)
            idx = np.argmax(prediction)
            player_move = labels[idx][2:].strip()
            confidence = prediction[0][idx]

            # Display on screen
            cv2.putText(frame, f"Your move: {player_move} ({int(confidence*100)}%)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: judge  Q: quit",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("RPS Game", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                computer_move = random.choice(choices)
                result = judge(player_move, computer_move)
                
                if "player" in result:
                    score["player"] += 1
                elif "computer" in result:
                    score["computer"] += 1

                print(f"Your move: {player_move} | Computer: {computer_move} | Result: {result}")
                print(f"Score - Player: {score['player']} | Computer: {score['computer']}\n")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    # Final result
    print("=== Final Result ===")
    print(f"Player: {score['player']} | Computer: {score['computer']}")
    if score["player"] > score["computer"]:
        print("You win!")
    elif score["player"] < score["computer"]:
        print("You lose...")
    else:
        print("Draw!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()