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
        수어 인식 AI 모델

        Args:
            input_shape: 입력 시퀀스 shape (시퀀스 길이, 특징 차원)
            num_classes: 분류할 수어 단어 개수
            model_path: 저장된 모델 경로
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.sign_labels = {}
        self.sequence_buffer = deque(maxlen=input_shape[0])

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()

        # 기본 수어 단어 매핑 (한국 수어 기준)
        self.initialize_sign_vocabulary()

    def initialize_sign_vocabulary(self):
        """기본 수어 단어 사전 초기화"""
        self.sign_labels = {
            0: "안녕하세요",
            1: "감사합니다",
            2: "사랑해요",
            3: "도와주세요",
            4: "괜찮아요",
            5: "미안합니다",
            6: "네",
            7: "아니요",
            8: "이름",
            9: "좋아요",
            10: "싫어요",
            11: "먹다",
            12: "마시다",
            13: "가다",
            14: "오다",
            15: "앉다",
            16: "서다",
            17: "보다",
            18: "듣다",
            19: "말하다",
            20: "물",
            21: "밥",
            22: "집",
            23: "학교",
            24: "병원",
            25: "친구",
            26: "가족",
            27: "어머니",
            28: "아버지",
            29: "형제",
            30: "오늘",
            31: "내일",
            32: "어제",
            33: "시간",
            34: "돈",
            35: "일",
            36: "공부",
            37: "책",
            38: "전화",
            39: "컴퓨터",
            40: "아프다",
            41: "피곤하다",
            42: "배고프다",
            43: "목마르다",
            44: "춥다",
            45: "덥다",
            46: "크다",
            47: "작다",
            48: "많다",
            49: "적다"
        }

    def build_model(self):
        """LSTM 기반 수어 인식 모델 구축"""
        model = models.Sequential([
            # LSTM layers for sequence processing
            layers.LSTM(128, return_sequences=True,
                       input_shape=self.input_shape),
            layers.Dropout(0.2),
            layers.BatchNormalization(),

            layers.LSTM(256, return_sequences=True),
            layers.Dropout(0.3),
            layers.BatchNormalization(),

            layers.LSTM(128, return_sequences=False),
            layers.Dropout(0.2),

            # Dense layers for classification
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.BatchNormalization(),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),

            layers.Dense(self.num_classes, activation='softmax')
        ])

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
        모델 학습

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터
            y_val: 검증 레이블
            epochs: 학습 에폭
            batch_size: 배치 크기
        """
        # Early stopping and learning rate reduction
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_sign_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

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
        수어 동작 예측

        Args:
            landmarks_sequence: 랜드마크 시퀀스
            threshold: 예측 신뢰도 임계값

        Returns:
            predicted_sign: 예측된 수어 단어
            confidence: 예측 신뢰도
        """
        if self.model is None:
            return None, 0.0

        # 입력 데이터 전처리
        if landmarks_sequence.shape[0] < self.input_shape[0]:
            # 시퀀스가 짧으면 패딩
            padding = np.zeros((self.input_shape[0] - landmarks_sequence.shape[0],
                              landmarks_sequence.shape[1]))
            landmarks_sequence = np.vstack([padding, landmarks_sequence])

        # 예측
        input_data = np.expand_dims(landmarks_sequence[-self.input_shape[0]:], axis=0)
        predictions = self.model.predict(input_data, verbose=0)

        # 최고 확률 클래스 찾기
        max_idx = np.argmax(predictions[0])
        confidence = predictions[0][max_idx]

        if confidence >= threshold:
            predicted_sign = self.sign_labels.get(max_idx, "Unknown")
            return predicted_sign, confidence

        return None, confidence

    def update_sequence_buffer(self, landmarks: np.ndarray):
        """
        시퀀스 버퍼 업데이트

        Args:
            landmarks: 새로운 랜드마크 데이터
        """
        self.sequence_buffer.append(landmarks)

    def get_sequence_array(self) -> np.ndarray:
        """
        현재 시퀀스 버퍼를 numpy 배열로 반환

        Returns:
            시퀀스 배열
        """
        if len(self.sequence_buffer) == 0:
            return np.zeros((0, 64))
        return np.array(self.sequence_buffer)

    def save_model(self, path: str):
        """모델 저장"""
        if self.model:
            self.model.save(path)
            # 레이블 딕셔너리도 저장
            labels_path = path.replace('.h5', '_labels.json')
            with open(labels_path, 'w', encoding='utf-8') as f:
                json.dump(self.sign_labels, f, ensure_ascii=False, indent=2)

    def load_model(self, path: str):
        """모델 로드"""
        self.model = tf.keras.models.load_model(path)
        # 레이블 딕셔너리도 로드
        labels_path = path.replace('.h5', '_labels.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                self.sign_labels = {int(k): v for k, v in labels_data.items()}

    def create_augmented_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 증강

        Args:
            X: 원본 데이터
            y: 원본 레이블

        Returns:
            증강된 데이터와 레이블
        """
        augmented_X = []
        augmented_y = []

        for i in range(len(X)):
            # 원본 데이터
            augmented_X.append(X[i])
            augmented_y.append(y[i])

            # 노이즈 추가
            noise = np.random.normal(0, 0.01, X[i].shape)
            augmented_X.append(X[i] + noise)
            augmented_y.append(y[i])

            # 시간축 스케일링
            if len(X[i]) > 10:
                indices = np.random.choice(len(X[i]), size=int(len(X[i])*0.9), replace=False)
                indices.sort()
                augmented_X.append(X[i][indices])
                augmented_y.append(y[i])

        return np.array(augmented_X), np.array(augmented_y)