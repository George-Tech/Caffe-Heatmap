import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/CN/liang.luo/code/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


net_file='deploy.prototxt'  
caffe_model='snapshot/slot_mbv2deep_1029_iter_120000.caffemodel'  
test_dir = "test_img"
out_dit = "test_out"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background', 'slot', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def preprocess(src):
    img = cv2.resize(src, (240,288))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def getheatmap(conv_out, c, j, i, thr):
    temp = conv_out.data[0,c,j,i]
    if temp < thr :
        return 0
    for jj in range(3):
        for ii in range(3):
            idxj = max(0,min(j+1-jj,72-1))
            idxi = max(0,min(i+1-ii,60-1))
            if temp < conv_out.data[0,c,idxj,idxi]:
                return 0
    return temp

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    resizeimg = cv2.resize(origimg, (240,288))
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    
    net.blobs['data'].data[...] = img
    out = net.forward()  
    
    #############################################################
    ch, h, w = net.blobs['conv_out_f'].data[0,:,:,:].shape
    print ("conv_out_f:", ch, w, h)

    svtxt = imgfile.replace('test_img','test_out')
    svtxt = svtxt.replace('.jpg','.txt')
    fp = open(svtxt, 'w')
    for c in range(ch):
        for j in range (h):
            for i in range (w):
                temp = net.blobs['conv_out_f'].data[0,c,j,i]   
                if j==0 and i==0:
                    maxv = temp
                    minv = temp
                else:
                    maxv=max(temp, maxv)
                    minv=min(temp, minv)
            
        fmap = np.zeros((h, w, 1), np.uint8)
        print("max = ", maxv,"min = ", minv)
        for j in range (h):
            for i in range (w):
                #temp = net.blobs['conv_out'].data[0,c,j,i]
                #temp = 255*(temp - minv)/(maxv-minv+0.001)
                fvalue = net.blobs['conv_out_f'].data[0,c,j,i]
                fvalue = max(0,min(255, fvalue*255))
                fmap[j][i]=int(fvalue)
                temp = getheatmap(net.blobs['conv_out_f'], c, j, i, 0.3)
                temp = min(255,255*temp)
                if (temp > 0.3*255):
                    cv2.circle(resizeimg, (i*4,j*4), 2, (0,255,0), -1)
                    print >> fp, "%d %d" % (i*4, j*4)
        svnametail = '_fmap' + str(c) + '.jpg'
        svfmap=imgfile.replace('test_img','test_out')
        svfmap=svfmap.replace('.jpg',svnametail)
        svdrawmap = svfmap.replace('fmap','draw_map')
        #cv2.imwrite(svfmap, fmap)
        cv2.imwrite(svdrawmap, resizeimg)
    fp.close()
        #mc.toimage(fmap).save(svfmap)
############################################################
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
