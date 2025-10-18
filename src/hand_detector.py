import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class HandDetector:
    FEATURE_DIMENSION = 64
    
    def __init__(self,
                 max_hands: int = 2,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        """
        손 인식 모듈 초기화

        Args:
            max_hands: 최대 인식 손 개수
            detection_confidence: 탐지 신뢰도 임계값
            tracking_confidence: 추적 신뢰도 임계값
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
        """
        프레임에서 손 감지 및 랜드마크 추출

        Args:
            frame: 입력 이미지 프레임

        Returns:
            annotated_frame: 손 랜드마크가 그려진 프레임
            landmarks_list: 각 손의 랜드마크 좌표 리스트
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        results = self.hands.process(frame_rgb)

        frame_rgb.flags.writeable = True
        annotated_frame = frame.copy()
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # 랜드마크 그리기
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # 랜드마크 좌표 추출 (정규화된 좌표)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # 손 방향 정보 추가 (왼손/오른손)
                hand_label = hand_handedness.classification[0].label
                landmarks.append(1.0 if hand_label == "Right" else 0.0)

                landmarks_list.append(landmarks)

        return annotated_frame, landmarks_list

    def normalize_landmarks(self, landmarks: List[float]) -> np.ndarray:
        """
        랜드마크 좌표를 정규화

        Args:
            landmarks: 원본 랜드마크 좌표

        Returns:
            정규화된 랜드마크 배열
        """
        if not landmarks:
            return np.zeros(64)  # 21 points * 3 coords + 1 hand label

        # 손목(0번) 기준으로 정규화
        landmarks_array = np.array(landmarks[:-1]).reshape(-1, 3)
        wrist = landmarks_array[0]

        # 손목을 원점으로 이동
        normalized = landmarks_array - wrist

        # 거리 기반 스케일 정규화
        distances = np.linalg.norm(normalized, axis=1)
        max_distance = np.max(distances[distances > 0])

        if max_distance > 0:
            normalized = normalized / max_distance

        # 평탄화 및 손 방향 정보 추가
        result = normalized.flatten()
        result = np.append(result, landmarks[-1])

        return result

    def extract_motion_features(self,
                               current_landmarks: List[float],
                               previous_landmarks: Optional[List[float]] = None) -> np.ndarray:
        """
        동작 특징 추출 (속도, 각도 변화 등)

        Args:
            current_landmarks: 현재 프레임 랜드마크
            previous_landmarks: 이전 프레임 랜드마크

        Returns:
            동작 특징 벡터
        """
        if previous_landmarks is None or not current_landmarks:
            return np.zeros(63)  # 21 points * 3 coords for velocity

        current = np.array(current_landmarks[:-1]).reshape(-1, 3)
        previous = np.array(previous_landmarks[:-1]).reshape(-1, 3)

        # 속도 계산
        velocity = current - previous

        return velocity.flatten()

    def release(self):
        """리소스 해제"""
        self.hands.close()