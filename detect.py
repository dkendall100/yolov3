import argparse
from sys import platform
import json
import operator
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from calculate import StateVector
#from predict.Models import Prediction
import cv2


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
           x_center=567,
           y_center=567,
           DT=1/100,


           ):
    img_size = (320, 192) if ONNX_EXPORT else init_img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')
    streams = 'streams' in source and source.endswith('.txt')

    state_tracker = StateVector(x_center, y_center, DT)

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
    # counter for debugging loop
    k = 0
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = model(img)

        for i, det in enumerate(non_max_suppression(pred, conf_thres, nms_thres)):  # detections per image
            print("entering loop %i" % k)
            if streams:  # batch_size > 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            frame_detections = []
            objs_array = [] # flush after 2
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                p = 0
                for *xyxy, conf, _, cls in det:
                    print("xyxy is: {}".format(*xyxy))
                    object_detection = {
                        "cls": int(cls),
                        "cnf": '%.2f' % float(conf),
                        "x": (int(xyxy[0])+int(xyxy[2]))/2,
                        "y": (int(xyxy[1])+int(xyxy[3]))/2,
                    }
                    if p < 2:
                        #frame_detections.sort(key=operator.itemgetter("cls", "cnf"))
                        print("inner loop count: %i for %i with %.2f at x: %i and y: %i" % (p,int(cls),float(conf),(int(xyxy[0])+int(xyxy[2]))/2,(int(xyxy[1])+int(xyxy[3]))/2))
                    else:
                        p = 0
                    p = p + 1
                    computations = StateVector((int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])+int(xyxy[3]))/2)
                    frame_detections.append(object_detection)

                    output = computations.get_tensor(frame_detections)
                    print(output)


                    if save_img or stream_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

#                       frame_detections.sort(key=operator.itemgetter("cls", "cnf"))

#                        if final_tensor is None:
#                            final_tensor = state_tracker.get_tensor(frame_detections)

               #     # TODO add model here to output final prediction tensor and print to the image

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
                        file.write(json.dumps(frame_detections, separators=(',', ':')) + "\n")
                        #file.write(output)
            else:
                if save_txt:
                    with open(save_path + '.txt', 'a') as file:
                        file.write(json.dumps(frame_detections, separators=(',', ':')) + "\n")

            print('%sDone. (%.3fs)' % (s, time.time() - t))
            #StateVector.pv2sv(file_out,"output/computations",(int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])+int(xyxy[3]))/2, 1/30)
            # Begin computations
            #computations = StateVector((int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])+int(xyxy[3]))/2, 1/30)
            #frame_detections.append(object_detection)
            #output = computations.get_tensor(frame_detections)
            #print(output)
            k = k + 1

            # Stream results
            if stream_img:
                cv2.imshow(p, im0)

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

    print('Done. (%.3fs)' % (time.time() - t0))
    #output = computations.get_tensor(frame_detections)
    #print(output)


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
