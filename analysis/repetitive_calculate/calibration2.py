import numpy as np
import cv2

class calib:
    #Note: cam0 should be cam1 in folders. cam1 should be cam0 in folders
    P1 = [2537.51445, 0, 1598.422058, -231.135868, 0, 2537.51445, 1367.049011, 0, 0, 0, 1, 0]
    P1 = np.array(P1).reshape(3,4)
    C1 = [2406.275208, 0, 1714.237176, 0, 2408.324856, 1361.612116, 0, 0, 1]
    C1 = np.array(C1).reshape(3,3)
    R1 = [0.999664, -0.00166, 0.025849, 0.001631, 0.9999980000000001, 0.001165, -0.025851, -0.001122, 0.999665]
    R1 = np.array(R1).reshape(3,3)
    D1 = [-0.004589, 0.003641, -0.000279, -0.000584, 0]
    D1 = [0,0,0,0, 0]
    D1 = np.array(D1)


    P0 = [2537.51445, 0, 1598.422058, 0, 0, 2537.51445, 1367.049011, 0, 0, 0, 1, 0]
    P0 = np.array(P0).reshape(3,4)
    C0 = [2438.620986, 0, 1701.022095, 0, 2443.895697, 1375.522774, 0, 0, 1]
    C0 = np.array(C0).reshape(3,3)
    R0 = [0.999449, -0.004905, 0.032821, 0.004942, 0.999987, -0.001063, -0.032815, 0.001224, 0.999461]
    R0 = np.array(R0).reshape(3,3)
    D0 = [-0.004538, 0.004316, -0.000377, -0.001485, 0]
    D0 = [0,0,0,0, 0]
    D0 = np.array(D0)
    #aov_x = np.arctan(P0[0,2]/P0[0,0])*2
    aov_y = np.arctan(P0[1,2]/P0[1,1])*2
    #aov_y = np.arctan(C0[1,2]/C0[1,1])*2

    @staticmethod
    def disparity2ptcloud(disparity):
        w,h = disparity.shape
        print calib.P0
        print calib.P1
        cx = calib.P0[0,2]
        cy =calib.P0[1,2] 
        f = calib.P0[0,0]
        Tx = calib.P1[0,3]
        cx2 = calib.P1[0,2]
        Q = np.array([[1, 0,     0,         -cx],
                      [0, 1,     0,         -cy],
                      [0, 0,     0,           f],
                      [0, 0, -1/Tx, (cx-cx2)/Tx]])
        
        Q = np.float32([[1, 0, 0, -0.5*w],
               [0, -1, 0, 0.5*h],
               [0, 0, f*0.001, 0],
               [0, 0, 0, 1]])
        
        print Q
        return cv2.reprojectImageTo3D(disparity, Q)





    @staticmethod
    def project_disp_to_points(disp,rescale_ratio):
        cx = calib.P0[0,2]*rescale_ratio
        cy =calib.P0[1,2] *rescale_ratio
        f = calib.P0[0,0] *rescale_ratio
        fy = calib.P0[1,1] *rescale_ratio
        Tx = calib.P1[0,3] *rescale_ratio
        cx2 = calib.P1[0,2] *rescale_ratio
        fx = f
        disp_min = np.min(disp)
        disp[disp < 0] = 0
        baseline = -1*calib.P1[0,3]/calib.P0[0,0]
        mask = disp > 0
        print "details:"
        print calib.P0[0,0]
        print baseline
        depth = calib.P0[0,0] * baseline / (disp + 1. - mask)
        depth = depth
        eff_depth = depth[depth<1.3]
        depth[disp == disp_min] = np.mean(eff_depth)
        #depth[depth < 0.7] = 0.7
        #depth[depth > 3] = 3


        depth = cv2.blur(depth, (11,11))
        print np.unique(depth)
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        c = c - cx
        r = r - cy
        c = c*depth /fx
        r = r*depth /fy


        c = c#/rescale_ratio
        r = r#/rescale_ratio
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]
        return points,c,r,depth

    @staticmethod
    def rectifyImage(img1,img2):
        assert img1.shape == img2.shape
        h,w,c = img1.shape
        
        map1x, map1y = cv2.initUndistortRectifyMap(
            cameraMatrix=calib.C1,
            distCoeffs=calib.D1,
            R=calib.R1,
            newCameraMatrix=calib.P1,
            size=(w, h),
            m1type=cv2.CV_32FC1)

        map2x, map2y = cv2.initUndistortRectifyMap(
            cameraMatrix=calib.C0,
            distCoeffs=calib.D0,
            R=calib.R0,
            newCameraMatrix=calib.P0,
            size=(w, h),
            m1type=cv2.CV_32FC1)

        img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        return img1_rect,img2_rect



