import cv2
import numpy as np
# https://www.youtube.com/watch?v=-eK58coLgIY
def calculate_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    fx = (24 * 1920) / 4000
    distance_to_ground = 100
    # Parameters for Lucas-Kanade method
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    old_corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

    # Initialize accumulated translation
    accumulated_translation = np.zeros(2)
    
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow using Lucas-Kanade method
            new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old_corners, None, **lk_params)

            if len(new_corners[status == 1])==0:
                old_corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
                continue

            # Select good points
            good_new = new_corners[status == 1]
            good_old = old_corners[status == 1]

            # Estimate translation
            if len(good_old) > 0 and len(good_new) > 0:
                translation = np.mean(good_new - good_old, axis=0)
                print("Estimated translation:", translation)
                # fx = (fmm * img_w)/ sensor_W
                
                translation_in_meters = translation * distance_to_ground / fx
                # Accumulate translation
                accumulated_translation += translation_in_meters
                # accumulated_translation += translation

                print("Accumulated translation:", accumulated_translation)

            # Update old corners
            old_corners = good_new.reshape(-1, 1, 2)

            # Draw optical flow tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

            img = cv2.add(frame, mask)
            cv2.imshow('Optical Flow', img)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            prev_gray = gray
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'bev_cam2.mp4'
calculate_optical_flow(video_path)
