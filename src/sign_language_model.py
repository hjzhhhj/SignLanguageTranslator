import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os
from typing import Tuple, Optional
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
            # sign_labels는 학습 시 실제 데이터의 레이블로 설정됨

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
            padding = np.zeros(
                (self.input_shape[0] - landmarks_sequence.shape[0], landmarks_sequence.shape[1]),
                dtype=np.float32
            )
            landmarks_sequence = np.vstack([padding, landmarks_sequence])

        landmarks_sequence = landmarks_sequence.astype(np.float32)

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
        self.sequence_buffer.append(np.asarray(landmarks, dtype=np.float32))

    def get_sequence_array(self) -> np.ndarray:
        """
        현재 누적된 시퀀스 버퍼를 numpy 배열로 반환

        Returns:
            np.ndarray: (len(buffer), 128)
        """
        if len(self.sequence_buffer) == 0:
            return np.zeros((0, self.input_shape[1]), dtype=np.float32)
        return np.stack(self.sequence_buffer).astype(np.float32)

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
        print(f"✓ 모델 로드 성공: {path}")

        labels_path = path.replace('.h5', '_labels.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                # JSON 키는 문자열 → int로 변환
                self.sign_labels = {int(k): v for k, v in labels_data.items()}
                # 로드된 레이블 수에 맞춰 num_classes 업데이트
                self.num_classes = len(self.sign_labels)
                print(f"✓ 레이블 로드 성공: {list(self.sign_labels.values())}")
        else:
            print(f"⚠ 레이블 파일을 찾을 수 없습니다: {labels_path}")

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

        sequence_length = X.shape[1]
        feature_dim = X.shape[2]

        for sample, label in zip(X, y):
            sample = sample.astype(np.float32)
            label = label.astype(np.float32)

            # ✅ 원본 데이터
            augmented_X.append(sample)
            augmented_y.append(label)

            # ✅ 노이즈 추가 (랜덤 잡음)
            noise = np.random.normal(0, 0.01, sample.shape).astype(np.float32)
            augmented_X.append(sample + noise)
            augmented_y.append(label)

            # ✅ 시간 축 랜덤 샘플링 (프레임 드롭 후 패딩)
            if sequence_length > 10:
                keep_length = max(10, int(sequence_length * 0.9))
                indices = np.sort(np.random.choice(sequence_length, size=keep_length, replace=False))
                sampled = sample[indices]

                if sampled.shape[0] < sequence_length:
                    padding = np.zeros((sequence_length - sampled.shape[0], feature_dim), dtype=np.float32)
                    sampled = np.vstack([padding, sampled])

                augmented_X.append(sampled)
                augmented_y.append(label)

        return np.array(augmented_X, dtype=np.float32), np.array(augmented_y, dtype=np.float32)
