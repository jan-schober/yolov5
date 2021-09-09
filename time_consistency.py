import glob
import cv2
from scipy import linalg
import numpy as np
from LPIPSmodels import util
import LPIPSmodels.dist_model as dm
import matplotlib.pyplot as plt
import seaborn as sns

#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim
gt_list = sorted(glob.glob('/home/schober/carla/for_yolov5/images/carla_rendered_01/*.jpg'))
aug_list= sorted(glob.glob('/home/schober/carla/for_yolov5/images/carla_01_fest_z/*.png'))
out_folder = 'output_time_final/'
model = dm.DistModel()
model.initialize(model='net-lin', net='alex', use_gpu=True)
## calculate the tOF

def calculate_tOF(gt_0, gt_1, ag_0, ag_1):
    OF_gt = cv2.calcOpticalFlowFarneback(gt_0, gt_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    OF_ag = cv2.calcOpticalFlowFarneback(ag_0, ag_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    OF_diff = np.absolute(OF_gt - OF_ag)
    TOF = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # l1 vector norm
    TOF = TOF.mean()

    return TOF, OF_gt, OF_ag, OF_diff

def calculate_tLP(gt_0, gt_1, ag_0, ag_1):
    img_gt_0 = util.im2tensor(gt_0)
    img_ag_0 = util.im2tensor(ag_0)
    img_gt_1 = util.im2tensor(gt_1)
    im_ag_1 = util.im2tensor(ag_1)

    dist0t = model.forward(img_gt_0, img_gt_1)
    dist1t = model.forward(img_ag_0, im_ag_1)

    dist01t = np.absolute(dist0t - dist1t) * 100.0

    return dist01t[0]

def convert_optical_flow(flow, frame):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = np.zeros_like(frame)
    mask[..., 1] = 255
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return rgb

def main():

    tof_array = []
    tlp_array = []


    for i in range(0, len(gt_list)-1):
        gt_0 = cv2.imread(gt_list[i])
        gt_0_gray = cv2.cvtColor(gt_0, cv2.COLOR_BGR2GRAY)
        gt_1 = cv2.imread(gt_list[i+1])
        gt_1_gray = cv2.cvtColor(gt_1, cv2.COLOR_BGR2GRAY)
        ag_0 = cv2.imread(aug_list[i])
        ag_0_gray = cv2.cvtColor(ag_0, cv2.COLOR_BGR2GRAY)
        ag_1 = cv2.imread(aug_list[i+1])
        ag_1_gray = cv2.cvtColor(ag_1, cv2.COLOR_BGR2GRAY)

        tof, OF_gt, OF_ag, OF_diff = calculate_tOF(gt_0_gray, gt_1_gray, ag_0_gray, ag_1_gray)
        file_name = gt_list[i].split('/')[-1]
        file_name = file_name.split('.')[0]

        rgb_gt = convert_optical_flow(OF_gt, gt_0)
        rgb_ag = convert_optical_flow(OF_ag, gt_0)
        rgb_diff = convert_optical_flow(OF_diff, gt_0)

        cv2.imwrite(out_folder + file_name + 'flow_gt.png', rgb_gt)
        cv2.imwrite(out_folder + file_name + 'flow_ag.png', rgb_ag)
        cv2.imwrite(out_folder + file_name + 'flow_diff.png', rgb_diff)

        tlp = calculate_tLP(gt_0, gt_1, ag_0, ag_1)

        tof_array.append(tof)
        tlp_array.append(tlp)

    np.save('tof_arr_final.npy' ,tof_array)
    np.save('tlp_arr_final.npy', tlp_array)

    tof_mean = "{:.3f}".format(np.mean(tof_array))
    tlp_mean = "{:.3f}".format(np.mean(tlp_array))


    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('images')
    ax1.set_ylabel('tOF-Score', color = color)
    ax1.plot(tof_array, color = color, lw = 0.5)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel('tLP-Score', color = color)
    ax2.plot(tlp_array, color = color, lw= 0.5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid()
    plt.title('Mean tOF = ' + str(tof_mean) + '   Mean tLP = ' + str(tlp_mean))
    plt.savefig('time_consistency.eps', format = 'eps', dpi = 200)
    '''
    plt.plot(tof_array, label='tOF', lw=0.5)
    plt.plot(tlp_array, label='tLP', lw=0.5)
    plt.xlabel('images')
    plt.legend(loc="upper right")
    plt.title( 'Mean tOF = ' + str(tof_mean) + '   Mean tLP = ' + str(tlp_mean))
    plt.savefig('time_consistency_rendered_augmented.png', dpi=500)
    '''

if __name__ == "__main__":
    main()
