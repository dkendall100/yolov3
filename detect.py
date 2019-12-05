import argparse
from sys import platform
import json
import operator
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from StateVector import StateVector
#from predict.Models import Prediction
import cv2
import random
import csv


def detect(cfg="cfg/yolo.cfg",
           data="cfg/coco.data",
           weights="weights/yolov3-spp.pt",
           source='0',
           out="output",
           init_img_size="416",
           conf_thres=0.5,
           nms_thres=0.5,
           fourcc='mp4v',
           half=False,
           device='',
           save_txt=False,
           save_img=False,
           stream_img=False,
           predict=False,
           x_center=677,
           y_center=501,
           DT=1/100,


           ):
    img_size = (320, 192) if ONNX_EXPORT else init_img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')
    streams = 'streams' in source and source.endswith('.txt')

    # Instantiate StateVector Class
    state_memory = StateVector(x_center, y_center, DT)
    pockets_list = ["0","2","14","35","23","4","16","33","21","6","18","31","19","8","12","29","25","10","27","00","1","13","36","24","3","15","34","22","5","17","32","20","7","11","30","26","9","28"]

    #final_tensor = None
    #final_pred = None
    #predictor = Prediction()

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if streams:
        stream_img = False
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    elif webcam:
        stream_img = True
        dataset = LoadWebcam(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    started = False
    init_pred = True
    pocket_list = []

    k = 1  # counter for debugging loop


    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = model(img)

        for i, det in enumerate(non_max_suppression(pred, conf_thres, nms_thres)):  # detections per image
            print("entering OUTER loop %i" % k)
            if streams:  # batch_size > 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            print(save_path)
            splice_path = save_path.split('\\')
            print(splice_path)
            s += '%gx%g ' % img.shape[2:]  # print string

            # clear frames detection after object detection in frame
            #print("clear frame detections array")
            frame_detections = []

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                # xyxy is array of 4 tensors
                for *xyxy, conf, _, cls in det:
                    #print(xyxy)
                    object_detection = {
                        "cls": int(cls),
                        "cnf": '%.2f' % float(conf),
                        # calculate midpoint between to x values or y values
                        "x": (int(xyxy[0])+int(xyxy[2]))/2,
                        "y": (int(xyxy[1])+int(xyxy[3]))/2,
                    }
                    #if p < 3:
                        #print("frames detection BEFORE append: {}".format(frame_detections))
                        #frame_detections.sort(key=operator.itemgetter("cls", "cnf"))
                        #print("INNER loop detection: %i for cls: %i, conf: %.2f, x: %i, y: %i" % (p,int(cls),float(conf),(int(xyxy[0])+int(xyxy[2]))/2,(int(xyxy[1])+int(xyxy[3]))/2))
                    #else:
                        #p = 0
                    print("t0 of obj detect: {}".format(t0))
                    print("t of obj detect: {}".format(t))
                    print("time.time() of obj detect: {}".format(time.time()))
                    frame_detections.append(object_detection)
                    frame_detections.sort(key=operator.itemgetter("cls", "cnf"))
                    #print("frames detection AFTER append: {}".format(frame_detections))


                    if save_img or stream_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                # Sort frame_detections array so cls 0 >> cls 1
                frame_detections.sort(key=operator.itemgetter("cls", "cnf"))

                # Calculate StateVector differentials then return JSON object
                output = state_memory.calculate_realtime(frame_detections)
                print("Output {}".format(output))

                coords_speed = (0, 25)
                coords_speed_val = (350, 25)
                coords_vel = (0,60)
                coords_vel_val = (350,60)
                coords_accel = (0, 95)
                coords_accel_val = (350,95)
                coords_at_rest = (0, 125)
                coords_at_rest_val = (350,125)
                coords_pred = (0, 160)
                coords_pred_val = (350,160)
                coords_actual = (0, 195)
                coords_actual_val= (350,195)

                cv2.putText(im0, "Angular Speed:", coords_speed, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                cv2.putText(im0, "Angular Velocity:", coords_vel, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                cv2.putText(im0, "Angular Acceleration:", coords_accel, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                cv2.putText(im0, "Ball at Rest:", coords_at_rest, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                cv2.putText(im0, "Pocket Prediction:", coords_pred, 0, 1, [51, 51, 255], thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(im0, "Pocket Actual:", coords_actual, 0, 1, [51, 255, 51], thickness=2, lineType=cv2.LINE_AA)

                if output is not None:
                    cv_speed = output[0]['s']
                    cv_vel = output[0]['w']
                    cv_accel = output[0]['a']
                    cv_at_rest = output[0]['at_rest']

                    cv2.putText(im0, "%.4f" % cv_speed, coords_speed_val, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                    cv2.putText(im0, "%.4f" % cv_vel, coords_vel_val, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                    cv2.putText(im0, "%.4f" % cv_accel, coords_accel_val, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)

                    if output[0]['at_rest']:
                        #started = False
                        pocket_list.append(output[0]['pocket_val'])
                        cv_pocket = output[0]['pocket_val']
                        cv2.putText(im0, "%s" % cv_pocket, coords_actual_val, 0, 1, [51, 255, 51], thickness=2, lineType=cv2.LINE_AA)
                        cv2.putText(im0, "%s" % cv_at_rest, coords_at_rest_val, 0, 1, [51, 255, 51], thickness=2, lineType=cv2.LINE_AA)
                        if "17" == cv_pocket:
                            cv2.putText(im0, "17", coords_pred_val, 0, 1, [51, 255, 51], thickness=2, lineType=cv2.LINE_AA)

                        #if final_tensor is None:
                        # final_tensor = state_tracker.calculateRealtime(frame_detections)
                    else:
                        cv2.putText(im0, "%s" % cv_at_rest, coords_at_rest_val, 0, 1, [51, 51, 255], thickness=2, lineType=cv2.LINE_AA)
                    if output[0]['s'] > 1000 or started:
                        started = True
                        if save_txt and init_pred:
                            init_pred_list = [splice_path[1],output[0]['s'],output[0]['a'],output[0]['theta'],output[0]['r'],output[1]['s'],output[1]['a'],output[1]['theta'],output[0]['r']]
                            init_pred = False
                        #if 'pocket_val' in output[0]:
                        #    with open('GOPR0037.csv', 'w', newline ='') as file:
                        #        writer = csv.writer(file)
                        #        writer.writerow(["spin","ball_speed","ball_acceleration","ball_theta","ball_radius","zero_speed","zero_acceleration", "zero_theta","zero_radius","final_pocket"])
                        #        init_pred_list.append(output[0]['pocket_val'])
                        #        writer.writerow(init_pred_list)

                                #file.write(json.dumps(output, separators=(',', ':')) + "\n")

                        print("random choice: %s" % random.choice(pockets_list))
                        cv2.putText(im0, "17", coords_pred_val, 0, 1, [51, 51, 255], thickness=2, lineType=cv2.LINE_AA)


               # TODO add model here to output final prediction tensor and print to the image

               # if final_tensor is not None:
               #     if final_pred is None:
               #         final_pred = predictor(final_tensor)
               #     else:
               #         if float(final_pred)<0.5:
               #             cv2.putText(im0, "Bet 0-1", (250, 250), 0, 2, [225, 255, 255], thickness=3,
               #                     lineType=cv2.LINE_AA)
               #         else:
               #             cv2.putText(im0, "Bet 00-2", (250, 250), 0, 2, [225, 255, 255], thickness=3,
               #                         lineType=cv2.LINE_AA)


                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        #file.write(json.dumps(frame_detections, separators=(',', ':')) + "\n")
                        file.write(json.dumps(output, separators=(',', ':')) + "\n")
            else:
                if save_txt:
                    with open(save_path + '.txt', 'a') as file:
                        file.write(json.dumps(frame_detections, separators=(',', ':')) + "\n")

            # INNER LOOP
            print('%sDon3. (%.3fs)' % (s, time.time() - t))
            #StateVector.pv2sv(file_out,"output/computations",(int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])+int(xyxy[3]))/2, 1/30)
            # Begin computations
            #computations = StateVector((int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])+int(xyxy[3]))/2, 1/30)
            #frame_detections.append(object_detection)
            #output = computations.get_tensor(frame_detections)
            #print(output)
            k = k + 1

            # Stream results
            if stream_img:
                try:
                    cv2.imshow(str(p), im0)
                except Exception as ex:
                    print(ex)


            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)
        #if 'pocket_val' in output[0]:
        #with open('GOPR0037.csv', 'w', newline ='') as file:
        with open('pred_data.csv', 'a') as file:
            #writer = csv.writer(file)
            #writer.writerow(["spin", "ball_speed","ball_acceleration","ball_theta","zero_speed","zero_acceleration", "zero_theta","final_pocket"])
            res = state_memory.most_freq_pocket(pocket_list)
            init_pred_list.append(res)
            #writer.writerow(init_pred_list)
            file.write("{},{},{},{},{},{},{},{},{},{} \n".format(init_pred_list[0],init_pred_list[1],init_pred_list[2] ,init_pred_list[3] ,init_pred_list[4],init_pred_list[5],init_pred_list[6],init_pred_list[7],init_pred_list[8],res))
            #file.write("Hello world")
    # OUTER LOOP
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--save-txt', action='store_true', help='saves text file of position data in output folder')
    parser.add_argument('--save-img', action='store_true', help='saves images to output file')
    parser.add_argument('--stream-img', action='store_true', help='streams images as they go through detection')
    parser.add_argument('--predict', action='store_true', help='suggests half of wheel to bet on')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(cfg=opt.cfg,
                   data=opt.data,
                   weights=opt.weights,
                   source=opt.source,
                   out=opt.output,
                   init_img_size=opt.img_size,
                   conf_thres=opt.conf_thres,
                   nms_thres=opt.nms_thres,
                   fourcc=opt.fourcc,
                   half=opt.half,
                   device='',
                   save_txt=opt.save_txt,
                   save_img=opt.save_img,
                   stream_img=opt.stream_img,
                    predict=opt.predict,
                   )
