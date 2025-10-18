import cv2
import numpy as np
import os
import json
import tensorflow as tf
from datetime import datetime
from src.hand_detector import HandDetector
from typing import List, Optional

class SignLanguageDataCollector:
    def __init__(self, data_dir: str = "data"):
        """
        수어 데이터 수집 도구

        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = data_dir
        self.hand_detector = HandDetector()
        self.current_label = None
        self.recording = False
        self.sequence_data = []
        self.sequence_counter = 0

        # 디렉토리 생성
        os.makedirs(f"{data_dir}/train", exist_ok=True)
        os.makedirs(f"{data_dir}/test", exist_ok=True)

    def collect_data(self, label: str, num_sequences: int = 30):
        """
        특정 레이블의 수어 데이터 수집

        Args:
            label: 수어 레이블
            num_sequences: 수집할 시퀀스 개수
        """
        cap = cv2.VideoCapture(0)
        self.current_label = label
        collected_sequences = 0
        current_sequence = []

        print(f"\n=== '{label}' 수어 데이터 수집 시작 ===")
        print("스페이스바를 눌러 녹화 시작/정지")
        print("ESC를 눌러 종료")
        print(f"목표: {num_sequences}개 시퀀스\n")

        while collected_sequences < num_sequences:
            ret, frame = cap.read()
            if not ret:
                break

            # 손 인식
            annotated_frame, landmarks_list = self.hand_detector.detect_hands(frame)

            # 상태 표시
            status_text = f"Label: {label} | Collected: {collected_sequences}/{num_sequences}"
            if self.recording:
                status_text += " | RECORDING"
                cv2.putText(annotated_frame, "RECORDING", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(annotated_frame, status_text, (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 녹화 중이고 손이 감지되면 데이터 저장
            if self.recording and landmarks_list:
                normalized_landmarks = self.hand_detector.normalize_landmarks(landmarks_list[0])
                current_sequence.append(normalized_landmarks)

                # 시퀀스 길이가 충분하면 저장
                if len(current_sequence) >= 30:
                    self._save_sequence(current_sequence, label, collected_sequences)
                    collected_sequences += 1
                    current_sequence = []
                    self.recording = False
                    print(f"시퀀스 {collected_sequences} 저장 완료")

            cv2.imshow('Data Collection', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # Space
                self.recording = not self.recording
                if self.recording:
                    current_sequence = []
                    print("녹화 시작...")
                else:
                    print("녹화 중지")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n'{label}' 데이터 수집 완료: {collected_sequences}개 시퀀스")

    def _save_sequence(self, sequence: list, label: str, index: int):
        """시퀀스 데이터 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_dir}/train/{label}_{timestamp}_{index}.npy"

        # numpy 배열로 변환 후 저장
        sequence_array = np.array(sequence)
        np.save(filename, sequence_array)

        # 메타데이터 저장
        metadata = {
            "label": label,
            "timestamp": timestamp,
            "sequence_length": len(sequence),
            "index": index
        }
        metadata_file = filename.replace('.npy', '_meta.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def create_dataset_from_files(self, data_type: str = "train"):
        """
        저장된 파일들로부터 데이터셋 생성

        Args:
            data_type: "train" 또는 "test"

        Returns:
            X: 특징 데이터
            y: 레이블 데이터
            label_map: 레이블 매핑
        """
        data_path = f"{self.data_dir}/{data_type}"
        X = []
        y = []
        label_map = {}
        label_counter = 0

        # 모든 .npy 파일 읽기
        for filename in os.listdir(data_path):
            if filename.endswith('.npy'):
                # 레이블 추출
                label = '_'.join(filename.split('_')[:-3])

                if label not in label_map:
                    label_map[label] = label_counter
                    label_counter += 1

                # 데이터 로드
                file_path = os.path.join(data_path, filename)
                sequence = np.load(file_path)

                # 시퀀스 길이 정규화 (30 프레임으로)
                if len(sequence) > 30:
                    sequence = sequence[:30]
                elif len(sequence) < 30:
                    padding = np.zeros((30 - len(sequence), sequence.shape[1]))
                    sequence = np.vstack([padding, sequence])

                X.append(sequence)
                y.append(label_map[label])

        if X:
            X = np.array(X)
            y = tf.keras.utils.to_categorical(y, num_classes=len(label_map))

        return X, y, label_map

    def interactive_collection(self):
        """대화형 데이터 수집 인터페이스"""
        print("\n=== 수어 데이터 수집 프로그램 ===")
        print("수집할 수어 단어를 입력하고 Enter를 누르세요")
        print("'quit'을 입력하면 종료됩니다\n")

        while True:
            label = input("수어 단어 입력 (한글): ").strip()

            if label.lower() == 'quit':
                break

            if label:
                try:
                    num_sequences = int(input("수집할 시퀀스 개수 (기본값: 30): ") or 30)
                    self.collect_data(label, num_sequences)
                except ValueError:
                    print("올바른 숫자를 입력하세요")
                except KeyboardInterrupt:
                    print("\n수집 중단됨")

        print("\n데이터 수집 종료")

    def _get_padded_feature_vector(self, landmarks_list: List[List[float]]) -> np.ndarray:
        """
        감지된 랜드마크 리스트를 2손(128차원) 기준으로 패딩하여 반환합니다.
        """
        all_hand_features = []
        
        # 1. 감지된 손의 수만큼 특징 추출
        for landmarks in landmarks_list:
            # 단일 손의 64차원 특징 벡터 추출
            normalized_features = self.hand_detector.normalize_landmarks(landmarks)
            all_hand_features.append(normalized_features)
            
        # 2. 특징 벡터의 수를 2개로 강제 정규화 (128차원 보장)
        if len(all_hand_features) < 2:
            # 2손 미만인 경우, 나머지 손은 0으로 패딩 (64차원)
            num_missing_hands = 2 - len(all_hand_features)
            
            feature_dim = self.hand_detector.FEATURE_DIMENSION 

            padding_feature = np.zeros(feature_dim)

            for _ in range(num_missing_hands):
                all_hand_features.append(padding_feature)
                
        # 3. 두 손의 특징을 하나의 128차원 벡터로 합치기
        feature_vector = np.concatenate(all_hand_features[:2])
        
        return feature_vector