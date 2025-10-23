import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
from collections import deque


class SignLanguageModel:
    def __init__(self,
                 input_shape: Tuple[int, int] = (30, 128),
                 num_classes: int = 50,
                 model_path: Optional[str] = None):
        """
        수어 인식 AI 모델 클래스

        Args:
            input_shape: 입력 시퀀스의 shape (프레임 수, 특징 차원)
            num_classes: 분류할 수어 단어의 총 개수
            model_path: 사전 학습된 모델의 경로 (.h5 파일)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.sign_labels = {}  # {클래스인덱스: 수어단어} 매핑
        self.sequence_buffer = deque(maxlen=input_shape[0])  # 실시간 입력 버퍼

        # 기존 모델이 있으면 로드, 없으면 새로 생성
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()

        # 기본 수어 단어 사전 초기화
        self.initialize_sign_vocabulary()

    def initialize_sign_vocabulary(self):
        """
        기본 50개 수어 단어 사전 초기화 (한국 수어 예시)
        """
        self.sign_labels = {
            0: "안녕하세요", 1: "감사합니다", 2: "사랑해요", 3: "도와주세요", 4: "괜찮아요",
            5: "미안합니다", 6: "네", 7: "아니요", 8: "이름", 9: "좋아요",
            10: "싫어요", 11: "먹다", 12: "마시다", 13: "가다", 14: "오다",
            15: "앉다", 16: "서다", 17: "보다", 18: "듣다", 19: "말하다",
            20: "물", 21: "밥", 22: "집", 23: "학교", 24: "병원",
            25: "친구", 26: "가족", 27: "어머니", 28: "아버지", 29: "형제",
            30: "오늘", 31: "내일", 32: "어제", 33: "시간", 34: "돈",
            35: "일", 36: "공부", 37: "책", 38: "전화", 39: "컴퓨터",
            40: "아프다", 41: "피곤하다", 42: "배고프다", 43: "목마르다",
            44: "춥다", 45: "덥다", 46: "크다", 47: "작다", 48: "많다", 49: "적다"
        }

    def build_model(self):
        """
        LSTM 기반의 수어 인식 모델 구축

        입력: (30, 128) 시퀀스 데이터
        출력: 50 클래스 softmax 분류
        """
        model = models.Sequential([
            # 1️⃣ 첫 번째 LSTM 층 (시퀀스 전체 유지)
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.2),
            layers.BatchNormalization(),

            # 2️⃣ 두 번째 LSTM 층
            layers.LSTM(256, return_sequences=True),
            layers.Dropout(0.3),
            layers.BatchNormalization(),

            # 3️⃣ 세 번째 LSTM 층 (최종 출력만 사용)
            layers.LSTM(128, return_sequences=False),
            layers.Dropout(0.2),

            # 4️⃣ 완전연결(Dense) 계층 - 비선형 분류기
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.BatchNormalization(),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),

            # 5️⃣ 출력 계층 (클래스 수 만큼 Softmax)
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # 모델 컴파일 (Adam 옵티마이저 사용)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

        self.model = model

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32):
        """
        모델 학습 메서드

        Args:
            X_train: 학습 데이터 (N, 30, 128)
            y_train: 학습 레이블 (N, 50 one-hot)
            X_val: 검증 데이터
            y_val: 검증 레이블
            epochs: 학습 반복 횟수
            batch_size: 배치 크기
        """
        # 학습 안정화를 위한 콜백 설정
        callbacks = [
            # 1️⃣ EarlyStopping: 과적합 방지
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            # 2️⃣ ReduceLROnPlateau: 학습 정체 시 학습률 감소
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                monitor='val_loss'
            ),
            # 3️⃣ ModelCheckpoint: 최고 정확도 모델 자동 저장
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_sign_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]

        # 검증 데이터 존재 시 지정
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        # 학습 실행
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict_sign(self,
                     landmarks_sequence: np.ndarray,
                     threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """
        수어 동작 예측 (실시간 또는 시퀀스 입력)

        Args:
            landmarks_sequence: (T, 128) 형태의 랜드마크 시퀀스
            threshold: 예측 신뢰도 임계값

        Returns:
            predicted_sign: 예측된 수어 단어 (임계값 미달 시 None)
            confidence: 예측 확률값
        """
        if self.model is None:
            return None, 0.0

        # 입력 길이가 부족할 경우 패딩
        if landmarks_sequence.shape[0] < self.input_shape[0]:
            padding = np.zeros((self.input_shape[0] - landmarks_sequence.shape[0],
                                landmarks_sequence.shape[1]))
            landmarks_sequence = np.vstack([padding, landmarks_sequence])

        # 모델 입력 형태 맞추기 (1, 30, 128)
        input_data = np.expand_dims(landmarks_sequence[-self.input_shape[0]:], axis=0)
        predictions = self.model.predict(input_data, verbose=0)

        # 최고 확률 클래스와 confidence 추출
        max_idx = np.argmax(predictions[0])
        confidence = predictions[0][max_idx]

        # 임계값 이상이면 단어 반환, 아니면 None
        if confidence >= threshold:
            predicted_sign = self.sign_labels.get(max_idx, "Unknown")
            return predicted_sign, confidence

        return None, confidence

    def update_sequence_buffer(self, landmarks: np.ndarray):
        """
        실시간 프레임 입력 시, 버퍼에 랜드마크 추가

        Args:
            landmarks: 현재 프레임의 랜드마크 벡터 (128차원)
        """
        self.sequence_buffer.append(landmarks)

    def get_sequence_array(self) -> np.ndarray:
        """
        현재 누적된 시퀀스 버퍼를 numpy 배열로 반환

        Returns:
            np.ndarray: (len(buffer), 128)
        """
        if len(self.sequence_buffer) == 0:
            return np.zeros((0, 128))
        return np.array(self.sequence_buffer)

    def save_model(self, path: str):
        """
        모델 및 레이블 딕셔너리 저장

        Args:
            path: 저장할 .h5 모델 경로
        """
        if self.model:
            self.model.save(path)
            labels_path = path.replace('.h5', '_labels.json')

            with open(labels_path, 'w', encoding='utf-8') as f:
                json.dump(self.sign_labels, f, ensure_ascii=False, indent=2)

    def load_model(self, path: str):
        """
        저장된 모델 및 레이블 로드

        Args:
            path: 모델 경로 (.h5)
        """
        self.model = tf.keras.models.load_model(path)

        labels_path = path.replace('.h5', '_labels.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                # JSON 키는 문자열 → int로 변환
                self.sign_labels = {int(k): v for k, v in labels_data.items()}

    def create_augmented_data(self,
                              X: np.ndarray,
                              y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 증강 함수: 노이즈 추가 및 프레임 샘플링으로 다양성 향상

        Args:
            X: 입력 데이터 (N, 30, 128)
            y: 레이블 (N, 50 one-hot)

        Returns:
            augmented_X, augmented_y: 증강된 데이터
        """
        augmented_X, augmented_y = [], []

        for i in range(len(X)):
            # ✅ 원본 데이터
            augmented_X.append(X[i])
            augmented_y.append(y[i])

            # ✅ 노이즈 추가 (랜덤 잡음)
            noise = np.random.normal(0, 0.01, X[i].shape)
            augmented_X.append(X[i] + noise)
            augmented_y.append(y[i])

            # ✅ 시간 축 랜덤 샘플링 (90% 프레임만 유지)
            if len(X[i]) > 10:
                indices = np.random.choice(len(X[i]),
                                           size=int(len(X[i]) * 0.9),
                                           replace=False)
                indices.sort()
                augmented_X.append(X[i][indices])
                augmented_y.append(y[i])

        return np.array(augmented_X), np.array(augmented_y)
