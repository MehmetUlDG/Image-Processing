import argparse
import time
from cmath import phase
from typing import Tuple,Optional
from collections import deque,Counter

import cv2
import mediapipe as mp


mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
mp_styles=mp.solutions.drawing_styles

def parse_args():
    p=argparse.ArgumentParser(description='Real-time finger picking with MediaPipe')

    p.add_argument('--source',type=str,default='0',help='Source image:"0"(camera) or video file path')
    p.add_argument('--mirror',action='store_true',help='Turn the image horizontal')
    p.add_argument('--stable',type=int,default=12,help='Stability window')
    p.add_argument('--hold_sec',type=float,default=0.7,help='Same number + lower wait')
    p.add_argument('--cooldown_sec',type=float,default=0.7,help='After being locked out')
    p.add_argument('--process_stride',type=int,default=2,help='Every few frames again')
    p.add_argument('--min_conf',type=float,default=0.05,help='Detection threshold')
    p.add_argument('--min_track_conf',type=float,default=0.05,help='Watching threshold')
    p.add_argument('--move_thresh',type=float,default=0.012,help='the permitted landmark')
    p.add_argument('--min-rel-box',type=float,default=0.03,help='Hand box min')

    return p.parse_args()


def open_source(s:str):
    try:
        src=int(s)
    except ValueError:
        src=s

    cap=cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open source {s}')
    return cap


def finger_count(landmarks,handed_label:str,mirrored:bool)->int:
    TIP=[4,8,12,16,20]
    PIP=[None,6,10,14,18]

    is_right=(handed_label=='Right')
    if mirrored:
        is_right=not is_right

    thumb_open=(landmarks[4].x>landmarks[3].x) if is_right else(landmarks[4].x<landmarks[3].x)
    open_others=sum(1 for tip,pip in zip(TIP[1:],PIP[1:]) if landmarks[tip].y<landmarks[pip].y)

    return (1 if thumb_open else 0)+open_others

def bbox_area_and_rect(landmarks)->Tuple[float,Tuple[float,float,float,float]]:
    xs=[lm.x for lm in landmarks]
    ys=[lm.y for lm in landmarks]
    xmin,xmax=max(0.0,min(xs)),min(1.0,max(xs))
    ymin,ymax=max(0.0,min(ys)),min(1.0,max(ys))
    w=max(1e-6,xmax-xmin)
    h=max(1e-6,ymax-ymin)

    return w*h,(xmin,ymin,xmax,ymax)

def avg_motion(prev,curr)->float:
    if prev is None or curr is None:
        return None
    s=0.0
    n=min(len(prev),len(curr))
    for i in range(n):
        dx=curr[i].x-prev[i].x
        dy=prev[i].y-curr[i].y
        s += (dx*dx + dy*dy)**0.5

    return s/n

def majority(buffer:deque,need:int,ratio:0.8)->Tuple[Optional[int],float]:

    if len(buffer)<need:
        return None,0.0
    cnt=Counter(buffer)
    val,freq=cnt.most_common(1)[0]
    return (val if freq>need*ratio else None),(freq/len(buffer))



def main():
    args=parse_args()
    cap=open_source(args.source)

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=args.min_conf,
        min_tracking_confidence=args.min_track_conf
    )as hands:

        lm_style=mp_styles.get_default_hand_landmarks_style()
        cn_style=mp_styles.get_default_hand_connections_style()

        phase=0
        first_num=None
        second_num=None

        preds=deque(maxlen=args.stable)
        hold_start_t=None
        cooldown_util=0.0

        prev_landmarks=None
        last_motion=None
        last_bbox_area=None
        last_candidate=None
        last_vote_ratio=0.0

        frame_count=0
        prev_t=0.0

        while True:
            ok,frame_bgr=cap.read()
            if not ok:
                break
            if args.mirror:
                frame_bgr=cv2.flip(frame_bgr,1)

            now=time.time()
            h,w=frame_bgr.shape[:2]
            process_this=(frame_count % max(1,args.process_stride)==0)
            frame_count+=1

            current_count=None
            handed_count=None
            bbox_px=None

            if process_this:
                frame_rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                res=hands.process(frame_rgb)

                if res.multi_hand_landmarks and res.multi_handedness:
                    hand=res.multi_hand_landmarks[0]
                    handed_label=res.multi_handedness[0].classification[0].label
                    area,(xmin,ymin,xmax,ymax)=bbox_area_and_rect(hand.landmark)

                    if area >= args.min_rel_box:
                        mp_draw.draw_landmarks(
                            frame_bgr,hand,mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=lm_style,connection_drawing_spec=cn_style
                        )
                        current_count=finger_count(hand.landmark,handed_label,args.mirror)

                        m=avg_motion(prev_landmarks,hand.landmark)
                        if m is not None:
                            last_motion=m
                            prev_landmarks=hand.landmark

                        x1,y1=int(xmin*w),int(ymin*h)
                        x2,y2=int(xmax*w),int(ymax*h)
                        bbox_px=(x1,y1,x2,y2)
                    else:
                        current_count=None
                        prev_landmarks=None
                else:
                    current_count=None
                    prev_landmarks=None


                if current_count is not None:
                    preds.append(current_count)

                candidate,vote_ratio=majority(preds,args.stable,ratio=0.8)
                last_candidate=candidate
                last_vote_ratio=vote_ratio


                can_lock=(now>=cooldown_util)

                if candidate is not None and can_lock and phase < 2:
                    if hold_start_t is None or candidate != last_candidate:

                        hold_start_t = now


                    if now - hold_start_t >= args.hold_sec:


                        if phase == 0:
                            first_num = candidate
                            phase = 1
                            preds.clear()
                        elif phase == 1:
                            second_num = candidate
                            phase = 2

                        cooldown_util = now + args.cooldown_sec
                        hold_start_t = None
                else:
                    if hold_start_t is not None and candidate is None:
                        hold_start_t = None



            phase_text = f'Phase: {phase} ({"First" if phase == 0 else "Second" if phase == 1 else "DONE"})'
            cv2.putText(frame_bgr, phase_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)


            selections_text = f'Selections: {first_num or "?"} | {second_num or "?"}'
            cv2.putText(frame_bgr, selections_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)


            pred_text = f'Current/Candidate: {current_count or "N/A"} / {last_candidate or "N/A"}'
            cv2.putText(frame_bgr, pred_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)


            motion_text = f'Motion: {last_motion:0.4f}' if last_motion is not None else 'Motion: N/A'
            cooldown_text = f'Cooldown: {max(0.0, cooldown_util - now):0.2f}s'

            cv2.putText(frame_bgr, motion_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)
            cv2.putText(frame_bgr, cooldown_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)
            cv2.putText(frame_bgr, "Quit: Press Q", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(frame_bgr, "Reset: Press esc", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                        cv2.LINE_AA)
            final_result = None
            result_color = (255, 255, 255)

            if phase == 2 and first_num is not None and second_num is not None:
                final_result = first_num + second_num
                result_color = (0, 255, 0)


            result_text = f'Result: {final_result if final_result is not None else "Waiting..."}'
            cv2.putText(frame_bgr, result_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2,
                        cv2.LINE_AA)

            t = time.time()
            fps = 1.0 / (t - prev_t) if prev_t != 0.0 else 0.0
            prev_t = t
            cv2.putText(frame_bgr, f'FPS: {fps:0.1f}', (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Finger Picker', frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                phase = 0
                first_num = None
                second_num = None
                preds.clear()
                hold_start_t = None
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
            main()
