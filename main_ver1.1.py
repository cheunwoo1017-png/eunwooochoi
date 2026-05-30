# Rock Paper Scissors Game - AI Model
# Project #1  Version: 1.1
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
        return "Draw"
    win_cases = {"scissors": "paper", "rock": "scissors", "paper": "rock"}
    if win_cases[player] == computer:
        return "Player Wins!"
    return "Computer Wins!"

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
            cv2.putText(frame, f"Round {round_num} / {total_rounds}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "SPACE: judge  Q: quit",
                        (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("RPS Game", frame)

            key = cv2.waitKey(1) & 0xFF

            # 결과 처리
            if key == ord(' '):
                computer_move = random.choice(choices)
                result = judge(player_move, computer_move)
                print("player:", player_move)
                print("computer:", computer_move)
                print("result:", result)
                if "Player" in result:
                    score["player"] += 1
                elif "Computer" in result:
                    score["computer"] += 1

                # 검은 결과 화면 생성
                result_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(result_frame, f"You: {player_move}",
                            (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(result_frame, f"PC: {computer_move}",
                            (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(result_frame, result,
                            (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(result_frame, f"Score {score['player']} : {score['computer']}",
                            (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                cv2.imshow("RPS Game", result_frame)
                start = cv2.getTickCount()
                while True:
                    cv2.waitKey(1)
                    elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
                    if elapsed > 2.0:
                        break
                break

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    # Final result screen
    final_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print("=== Final Result ===")
    print(f"Player: {score['player']} | Computer: {score['computer']}")

    if score["player"] > score["computer"]:
        final_text = "You Win!"
        color = (0, 255, 0)
    elif score["player"] < score["computer"]:
        final_text = "You Lose..."
        color = (0, 0, 255)
    else:
        final_text = "Draw!"
        color = (255, 255, 0)

    cv2.putText(final_frame, "=== Final Result ===",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(final_frame, f"Player {score['player']} : {score['computer']} Computer",
                (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.putText(final_frame, final_text,
                (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    cv2.putText(final_frame, "Press any key to exit",
                (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    cv2.imshow("RPS Game", final_frame)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()