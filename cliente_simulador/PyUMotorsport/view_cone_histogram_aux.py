import time, os, cv2
import utils
from os import listdir
import matplotlib.pyplot as plt
# # import PyQt4
# import matplotlib
# # matplotlib.use('qt4agg')

intialTracbarVals = [[73, 67, 27], [129, 255, 255], [29, 27, 57], [46, 255, 255], [5, 46, 32], [14, 180, 204]]
utils.initializeTrackbars_hsv(intialTracbarVals[0], intialTracbarVals[1], intialTracbarVals[2], intialTracbarVals[3],
                              intialTracbarVals[4], intialTracbarVals[5])

if __name__ == '__main__':

    # Image processing
    images = []
    for file in os.listdir("/home/shernandez/PycharmProjects/UMotorsport/Deteccion_conos/cliente_simulador/PyUMotorsport/imagenes_auxiliares"):
        images.append(cv2.imread("/home/shernandez/PycharmProjects/UMotorsport/Deteccion_conos/cliente_simulador/PyUMotorsport/imagenes_auxiliares/"+file))

    cont = 0
    for image in images:

        cond = True
        # if cont > 5:
        #     cond =False
        # cont += 1
        while True:
            src = utils.valTrackbars_hsv()
            cv2.imshow('cono', image)
            l_b, u_b, l_y, u_y, l_o, u_o = utils.valTrackbars_hsv()
            masked_blue = utils.hsv_filter(image, l_b, u_b)
            masked_yell = utils.hsv_filter(image, l_y, u_y)
            masked_oran = utils.hsv_filter(image, l_o, u_o)
            cv2.imshow('masked blue', masked_blue)
            cv2.imshow('masked yellow', masked_yell)
            cv2.imshow('masked orange', masked_oran)
            cv2.waitKey(1)
            print('Blue thres: ', l_b, u_b)
            print('Yellow thres: ', l_y, u_y)
            print('Orange thres: ', l_o, u_o)
            if cond:
                break

        hsv_im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w = image.shape[1]
        h = image.shape[0]
        r_cols = image[:,:,0].sum(axis=0)
        g_cols = image[:, :, 1].sum(axis=0)
        b_cols = image[:, :, 2].sum(axis=0)
        r_rows = image[:, :, 0].sum(axis=1)
        g_rows = image[:, :, 1].sum(axis=1)
        b_rows = image[:, :, 2].sum(axis=1)

        plt.subplot(221)
        plt.imshow(image)

        plt.subplot(222)
        plt.plot(r_cols, color='r')
        plt.plot(g_cols, color='g')
        plt.plot(b_cols, color='b')

        plt.subplot(223)
        plt.plot(r_rows, color='r')
        plt.plot(g_rows, color='g')
        plt.plot(b_rows, color='b')

        plt.subplot(224)
        h_hist = cv2.calcHist(hsv_im[:, :, 0], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist(hsv_im[:, :, 1], [0], None, [256], [0, 256])
        v_hist = cv2.calcHist(hsv_im[:, :, 2], [0], None, [256], [0, 256])
        plt.plot(h_hist, color='b')
        plt.plot(s_hist, color='g')
        plt.plot(v_hist, color='r')



        plt.xlim([0, 256])
        plt.show()

    # cv2.imshow("cone", image)
    #
    # cv2.waitKey()


