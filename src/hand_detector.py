import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple

class HandDetector:
    """
    MediaPipe를 이용한 손 랜드마크 감지 및 특징 추출 클래스
    """

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
        """
        손 검출기 초기화

        Args:
            static_image_mode: True이면 한 프레임마다 새로 검출 (정지 이미지용)
            max_num_hands: 감지할 최대 손 개수
            detection_confidence: 손 검출 최소 신뢰도
            tracking_confidence: 손 추적 최소 신뢰도
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 한 손의 랜드마크(21개 × 3좌표 = 63개) + 손 크기(scale) 1개 = 64차원
        self.FEATURE_DIMENSION = 64

    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
        """
        입력 프레임에서 손을 감지하고 랜드마크 좌표를 추출

        Args:
            frame: 입력 영상 (BGR, OpenCV 형식)

        Returns:
            annotated_frame: 시각화된 영상
            landmarks_list: 각 손의 랜드마크 좌표 리스트 (N손 × 63차원)
        """
        # MediaPipe는 RGB 영상 입력 필요
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 손 검출
        results = self.hands.process(rgb_frame)
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 21개 랜드마크 좌표 (x, y, z) → 63차원 벡터로 변환
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(landmarks)

                # 시각화 (선택 사항)
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame, landmarks_list

    def normalize_landmarks(self, landmarks: List[float]) -> np.ndarray:
        """
        손 랜드마크를 정규화하여 64차원 특징 벡터로 반환

        Args:
            landmarks: [x1, y1, z1, ..., x21, y21, z21] 형태의 리스트 (길이 63)

        Returns:
            normalized: 정규화된 64차원 numpy 벡터
        """
        # 21개의 (x, y, z) 좌표를 numpy 배열로 변환
        landmarks = np.array(landmarks, dtype=np.float32).reshape(-1, 3)

        # 손 중심(평균) 계산
        center = np.mean(landmarks, axis=0)

        # 중심 기준으로 이동한 좌표 (위치 불변성)
        centered_landmarks = landmarks - center

        # 손 크기 기준으로 스케일 정규화 (크기 불변성)
        max_distance = np.max(np.linalg.norm(centered_landmarks, axis=1))
        scale = max_distance if max_distance > 0 else 1.0
        normalized_landmarks = centered_landmarks / scale

        # (21×3)=63차원 벡터 + 손 크기(scale) 1개 = 64차원 특징 벡터
        flattened = normalized_landmarks.flatten()
        normalized = np.concatenate([flattened, np.array([scale], dtype=np.float32)]).astype(np.float32)

        return normalized

    def release(self):
        """
        MediaPipe 리소스 해제
        """
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
