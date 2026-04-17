"""
OpticalFlowEgomotion — Global background flow tracking (spec §Egomotion).

Uses Sparse Optical Flow (Lucas-Kanade) to track background features
and determine if the camera/user is moving, ignoring dynamic objects.
"""
import cv2
import numpy as np
from typing import List, Tuple

class OpticalFlowEgomotion:

    def __init__(self):
        self._prev_gray = None
        self._p0 = None
        
        # Lucas-Kanade params
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection params
        self.feature_params = dict(
            maxCorners=150,
            qualityLevel=0.1,
            minDistance=7,
            blockSize=7
        )
        
        # Classes that are likely to move independently
        self._dynamic_classes = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat", "horse", "bird"}
        
        self._last_speed_mps = 0.0
        self._last_state = "Stationary"

    def update(self, frame_gray: np.ndarray, boxes: List[Tuple[int, int, int, int]], labels: List[str]) -> Tuple[float, str]:
        """
        Calculates egomotion. 
        Returns (estimated_speed_mps, user_motion_state)
        """
        h, w = frame_gray.shape
        # Create mask: white (255) for background, black (0) for dynamic objects
        mask = np.ones_like(frame_gray) * 255
        for box, label in zip(boxes, labels):
            if label.lower() in self._dynamic_classes:
                x1, y1, x2, y2 = box
                # Ensure bounds
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0
                
        if self._prev_gray is None:
            self._prev_gray = frame_gray.copy()
            self._p0 = cv2.goodFeaturesToTrack(self._prev_gray, mask=mask, **self.feature_params)
            return 0.0, "Stationary"
            
        if self._p0 is None or len(self._p0) < 10:
            # Re-detect if we lost features
            self._p0 = cv2.goodFeaturesToTrack(self._prev_gray, mask=mask, **self.feature_params)
            if self._p0 is None:
                self._prev_gray = frame_gray.copy()
                return 0.0, "Stationary"
                
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self._prev_gray, frame_gray, self._p0, None, **self.lk_params)
        
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = self._p0[st == 1]
            
            if len(good_new) > 5:
                # Calculate median displacement
                dxs = good_new[:, 0] - good_old[:, 0]
                dys = good_new[:, 1] - good_old[:, 1]
                
                med_dx = np.median(dxs)
                med_dy = np.median(dys)
                
                # Update tracking points
                self._p0 = good_new.reshape(-1, 1, 2)
                
                # Interpret motion
                state = "Stationary"
                speed = 0.0
                
                # Calculate scale change (expansion)
                # Expansion vector dot product heuristic
                cx, cy = w / 2, h / 2
                expansions = []
                for (nx, ny), (ox, oy) in zip(good_new, good_old):
                    cx_vec = ox - cx
                    cy_vec = oy - cy
                    norm = np.sqrt(cx_vec**2 + cy_vec**2) + 1e-5
                    # Projection of motion vector onto vector originating from center
                    expansion = ((nx - ox) * cx_vec + (ny - oy) * cy_vec) / norm
                    expansions.append(expansion)
                    
                med_exp = np.median(expansions)
                
                # Thresholds to determine state from median pixel flows
                # These are heuristic values tuning walking movement approx to m/s
                if med_exp > 0.5:
                    state = "Walking forward"
                    speed = min(2.5, float(med_exp * 0.4)) # Approximate mps (expansion of 2.5px ~ 1m/s heuristically)
                elif med_exp < -0.5:
                    # Walking backward reduces approaching speed
                    state = "Walking backward"
                    speed = max(-2.5, float(med_exp * 0.4))
                elif med_dx > 3.0:
                    state = "Panning right"
                    speed = 0.0
                elif med_dx < -3.0:
                    state = "Panning left"
                    speed = 0.0
                elif med_dy > 3.0:
                    state = "Tilting down"
                    speed = 0.0
                elif med_dy < -3.0:
                    state = "Tilting up"
                    speed = 0.0
                    
                self._last_speed_mps = speed
                self._last_state = state
            else:
                self._p0 = None
        else:
             self._p0 = None

        # Periodically refresh features to keep tracking fresh
        if self._p0 is not None and len(self._p0) < 60:
             new_p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
             if new_p0 is not None:
                 self._p0 = new_p0

        self._prev_gray = frame_gray.copy()
        return self._last_speed_mps, self._last_state
