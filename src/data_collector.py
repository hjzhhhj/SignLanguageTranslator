import cv2
import numpy as np
import os
import json
import tensorflow as tf
from datetime import datetime
from src.hand_detector import HandDetector
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

class SignLanguageDataCollector:
    def __init__(self, data_dir: str = "data"):
        # 수어 데이터 수집 도구 (data_dir: 저장 디렉토리)
        self.data_dir = data_dir
        self.hand_detector = HandDetector() # 손 랜드마크 탐지기
        self.recording = False # 녹화 상태 여부 플래그

        # 학습/테스트용 디렉토리 생성
        os.makedirs(f"{data_dir}/train", exist_ok=True)
        os.makedirs(f"{data_dir}/test", exist_ok=True)

    def _put_korean_text(self, frame: np.ndarray, text: str, position: tuple,
                         font_size: int = 30, color: tuple = (255, 255, 255)):
        # OpenCV 이미지에 한글 텍스트 렌더링
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            # 한글 폰트 경로 설정
            # Mac용 폰트 경로
            font_paths = [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                "/Library/Fonts/AppleGothic.ttf",

                # Windows용 폰트 경로 (윈도우에서 사용 시 아래 경로 사용)
                # "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
                # "C:/Windows/Fonts/gulim.ttc",       # 굴림
                # "C:/Windows/Fonts/batang.ttc",      # 바탕
            ]

            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue

            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        color_rgb = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def collect_data(self, label: str, num_sequences: int = 30):
        # 특정 수어 레이블에 대한 데이터 시퀀스 수집 (label: 수어 단어, num_sequences: 수집 개수)
        cap = cv2.VideoCapture(0)  # 웹캠 연결
        collected_sequences = 0    # 수집된 시퀀스 개수 카운트
        current_sequence = []      # 현재 녹화 중인 시퀀스 버퍼

        # 사용자 안내 출력
        print(f"\n=== '{label}' 수어 데이터 수집 시작 ===")
        print("스페이스바: 녹화 시작/정지 | ESC: 종료")
        print(f"목표: {num_sequences}개 시퀀스\n")

        # 목표 수만큼 시퀀스를 수집할 때까지 반복
        while collected_sequences < num_sequences:
            ret, frame = cap.read()
            if not ret:
                break  # 카메라 입력이 없으면 종료

            # 손 인식 수행
            annotated_frame, landmarks_list = self.hand_detector.detect_hands(frame)

            # 녹화 중인 경우 화면에 빨간 텍스트로 표시
            if self.recording:
                cv2.putText(annotated_frame, "RECORDING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 상태 텍스트 표시 (한글 지원)
            status_text = f"레이블: {label} | 수집: {collected_sequences}/{num_sequences}"
            annotated_frame = self._put_korean_text(annotated_frame, status_text, (10, 60),
                                                     font_size=25, color=(255, 255, 255))

            # 녹화 중이며 손이 감지되면 데이터 수집
            if self.recording and landmarks_list:
                # 손 랜드마크를 2손 기준 128차원 특징 벡터로 정규화
                normalized_landmarks = self._get_padded_feature_vector(landmarks_list)
                current_sequence.append(normalized_landmarks)

                # 시퀀스가 30프레임 이상 쌓이면 저장
                if len(current_sequence) >= 30:
                    self._save_sequence(current_sequence, label, collected_sequences)
                    collected_sequences += 1
                    current_sequence = []   # 버퍼 초기화
                    self.recording = False  # 자동으로 녹화 종료
                    print(f"시퀀스 {collected_sequences} 저장 완료")

            # 영상 출력
            cv2.imshow('Data Collection', annotated_frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC → 종료
                break
            elif key == 32:  # Space → 녹화 시작/정지
                self.recording = not self.recording
                if self.recording:
                    current_sequence = []  # 새 시퀀스 시작
                    print("녹화 시작...")
                else:
                    print("녹화 중지")

        # 종료 처리
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n'{label}' 데이터 수집 완료: {collected_sequences}개 시퀀스")

    def _save_sequence(self, sequence: list, label: str, index: int):
        # 시퀀스 데이터를 파일로 저장 (.npy + .json 메타데이터)
        # 파일 이름에 시간 스탬프 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_dir}/train/{label}_{timestamp}_{index}.npy"

        # numpy 배열로 변환 후 저장
        sequence_array = np.array(sequence)
        np.save(filename, sequence_array)

        # 메타데이터 JSON으로 저장
        metadata = {
            "label": label,
            "timestamp": timestamp,
            "sequence_length": len(sequence),
            "index": index
        }
        metadata_file = filename.replace('.npy', '_meta.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def create_dataset_from_files(self,
                                  data_type: str = "train",
                                  label_map: Optional[Dict[str, int]] = None):
        # .npy 파일로부터 학습/테스트용 데이터셋 생성 (반환: X, y, label_map)
        data_path = f"{self.data_dir}/{data_type}"
        X, y = [], []
        allow_new_labels = label_map is None
        label_map = {} if label_map is None else dict(label_map)
        label_counter = len(label_map)

        # 해당 디렉토리의 모든 .npy 파일을 순회
        for filename in os.listdir(data_path):
            if filename.endswith('.npy'):
                # 파일 이름에서 레이블 추출
                # 예: hello_20231011_001.npy → "hello"
                label = '_'.join(filename.split('_')[:-3])

                # 새 레이블이면 인덱스 부여
                if label not in label_map:
                    if not allow_new_labels:
                        # 정의되지 않은 레이블은 건너뜀
                        continue
                    label_map[label] = label_counter
                    label_counter += 1

                # 데이터 로드
                file_path = os.path.join(data_path, filename)
                sequence = np.load(file_path).astype(np.float32)

                # 시퀀스 길이를 30프레임으로 정규화
                if len(sequence) > 30:
                    sequence = sequence[:30]
                elif len(sequence) < 30:
                    padding = np.zeros((30 - len(sequence), sequence.shape[1]), dtype=np.float32)
                    sequence = np.vstack([padding, sequence])

                X.append(sequence)
                y.append(label_map[label])

        # 배열로 변환
        if X:
            X = np.array(X, dtype=np.float32)
            y = tf.keras.utils.to_categorical(y, num_classes=len(label_map))

        return X, y, label_map

    def interactive_collection(self):
        # 터미널에서 사용자 입력을 받아 수어 데이터를 인터랙티브하게 수집
        print("\n=== 수어 데이터 수집 프로그램 ===")
        print("수집할 수어 단어를 입력하고 Enter를 누르세요.")
        print("'quit'을 입력하면 종료됩니다.\n")

        while True:
            # 수어 단어 입력
            label = input("수어 단어 입력 (한글): ").strip()

            # 종료 명령어
            if label.lower() == 'quit':
                break

            # 정상 입력 시 데이터 수집 시작
            if label:
                try:
                    num_sequences = int(input("수집할 시퀀스 개수 (기본값: 30): ") or 30)
                    self.collect_data(label, num_sequences)
                except ValueError:
                    print("올바른 숫자를 입력하세요.")
                except KeyboardInterrupt:
                    print("\n수집 중단됨.")

        print("\n데이터 수집 종료.")

    def _get_padded_feature_vector(self, landmarks_list: List[List[float]]) -> np.ndarray:
        # 손 랜드마크를 2손(128차원) 기준으로 패딩된 벡터로 변환 (손 1개만 감지시 0으로 채움)
        all_hand_features = []
        feature_dim = self.hand_detector.FEATURE_DIMENSION

        # 감지된 손마다 특징 추출
        for landmarks in landmarks_list:
            landmarks_array = np.array(landmarks, dtype=np.float32).reshape(-1, 3)
            wrist_x = landmarks_array[0, 0]
            normalized_features = self.hand_detector.normalize_landmarks(landmarks)
            all_hand_features.append((wrist_x, normalized_features))

        # 왼손 -> 오른손 순으로 정렬
        all_hand_features.sort(key=lambda item: item[0])
        sorted_features = [feat for _, feat in all_hand_features]

        # 손이 2개 미만이면 0벡터로 패딩
        if len(sorted_features) < 2:
            num_missing_hands = 2 - len(sorted_features)
            padding_feature = np.zeros(feature_dim, dtype=np.float32)

            for _ in range(num_missing_hands):
                sorted_features.append(padding_feature)

        # 두 손(좌, 우)의 특징을 합쳐 128차원 벡터로 생성
        feature_vector = np.concatenate(sorted_features[:2]).astype(np.float32)

        return feature_vector
