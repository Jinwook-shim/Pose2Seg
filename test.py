import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils
import cv2
import torch
from datasets.RingDataset import RingDataset

import matplotlib

matplotlib.use('TkAgg')


# from pathlib import Path
def test(model, dataset='cocoVal', logger=print):
    if dataset == 'OCHumanVal':
        ImageRoot = r'\\fs01\Algo\ML\Datasets\Pose2Seg\data\OCHuman\images'
        AnnoFile = r'\\fs01\Algo\ML\Datasets\Pose2Seg\data\OCHuman\annotations\ochuman_coco_format_val_range_0.00_1.00.json'
    elif dataset == 'OCHumanTest':
        ImageRoot = r'\\fs01\Algo\ML\Datasets\Pose2Seg\data\OCHuman\images'
        AnnoFile = r'\\fs01\Algo\ML\Datasets\Pose2Seg\data\OCHuman\annotations\ochuman_coco_format_test_range_0.00_1.00.json'
    elif dataset == 'cocoVal':
        ImageRoot = r'\\fs01\Algo\ML\Datasets\Pose2Seg\data\coco2017\val2017'
        AnnoFile = r'\\fs01\Algo\ML\Datasets\Pose2Seg\data\coco2017\annotations\person_keypoints_val2017_pose2seg.json'

    use_cocoDataset = 0

    if use_cocoDataset:
        datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    else:
        datainfos = RingDataset(path_of_rgbs=r'\\fs01\Algo\ML\Datasets\NeuralPCL\Data\Adaya_2019-11-05-16.41.21_joined\Components\RGB')
        datainfos = torch.utils.data.DataLoader(datainfos, batch_size=1,
                                                shuffle=True, num_workers=0)

    model.eval()

    results_segm = []
    imgIds = []

    for i in tqdm(range(0, len(datainfos))):
        if use_cocoDataset:
            rawdata = datainfos[i]
            img = rawdata['data']
            image_id = ID =rawdata['id']
            gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1)  # (N, 17, 3)
            gt_segms = rawdata['segms']
            height, width = img.shape[0:2]
            image_id = rawdata['id']
            gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
            valid = True
        else:
            image_id = i
            rawdata = next(iter(datainfos))
            img = rawdata[0][0].numpy()
            ID = rawdata[3][0]
            valid = rawdata[4][0].numpy()
            gt_kpts = rawdata[1].numpy()
            gt_masks = rawdata[2][0].numpy()
            height, width = img.shape[0:2]
        if not valid:
            continue
        # region test openpose data
        if False:
            fig, ax = plt.subplots(1, 2)

            colors_kp = np.array(list(range(0, kp.shape[0])))
            ax[1].imshow(img)

            ax[0].imshow(img)
            ax[0].scatter(kp[:, 0], kp[:, 1], c=colors_kp, s=5, cmap='hsv')
            for i in range(kp.shape[0]):
                ax[0].plot(kp[i, 0], kp[i, 1], '*r')
                ax[0].text(kp[i, 0], kp[i, 1], i, color='white')
            colors_gt = np.array(list(range(0, gt_kpts[0].shape[0])))
            ch = 0
            ax[1].scatter(gt_kpts[ch][:, 0], gt_kpts[ch][:, 1], c=colors_gt, s=5, cmap='hsv')
            for i in range(gt_kpts[0].shape[0]):
                ax[1].plot(gt_kpts[ch][i, 0], gt_kpts[ch][i, 1], '*r')
                ax[1].text(gt_kpts[ch][i, 0], gt_kpts[ch][i, 1], i, color='white')
            plt.show()
        # endregion

        # output = model([img], [gt_kpts], [gt_masks])
        # mask = np.all(img > 0, 2)
        # import time
        iter1 = 1
        # start = time.time()

        for ii in range(iter1):
            output = model([img], [gt_kpts], [gt_masks])

        # print('It took', (time.time() - start)/iter, 'seconds.')

        img = img[..., ::-1]
        # plt.switch_backend("TkAgg")
        # plt.imshow(gt_masks[0,...])
        # plt.show()

        # fig.savefig('C:\\Users\\erez\\Projects\\Pose2Seg\\demo.png', bbox_inches='tight')

        MASKS = np.zeros(output[0][0].shape)
        for id, mask in enumerate(output[0]):
            from skimage.color import label2rgb
            MASKS += (id + 1) * mask
            maskencode = maskUtils.encode(np.asfortranarray(mask))
            maskencode['counts'] = maskencode['counts'].decode('ascii')
            results_segm.append({
                "image_id": image_id,
                "category_id": 1,
                "score": 1.0,
                "segmentation": maskencode
            })
        imgIds.append(image_id)
        plt.switch_backend("TkAgg")
        image_label_overlay = label2rgb(MASKS, image=img, alpha=0.3, bg_label=0)
        plt.imshow(image_label_overlay)
        plt.savefig(r'out\e\{}.png'.format(ID))
        plt.close('all')
        # plt.show()

    def do_eval_coco(image_ids, coco, results, flag):
        from pycocotools.cocoeval import COCOeval
        assert flag in ['bbox', 'segm', 'keypoints']
        # Evaluate
        coco_results = coco.loadRes(results)
        cocoEval = COCOeval(coco, coco_results, flag)
        cocoEval.params.imgIds = image_ids
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval
    if use_cocoDataset:
        cocoEval = do_eval_coco(imgIds, datainfos.COCO, results_segm, 'segm')
        logger('[POSE2SEG]          AP|.5|.75| S| M| L|    AR|.5|.75| S| M| L|')
        _str = '[segm_score] %s ' % dataset
        for value in cocoEval.stats.tolist():
            _str += '%.3f ' % value
        logger(_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pose2Seg Testing")
    parser.add_argument(
        "--weights",
        help="path to .pkl model weight",
        type=str,
    )
    parser.add_argument(
        "--coco",
        help="Do test on COCOPersons val set",
        action="store_true",
    )
    parser.add_argument(
        "--OCHuman",
        help="Do test on OCHuman val&test set",
        action="store_true",
    )

    args = parser.parse_args()

    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(args.weights)

    print('===========>   testing    <===========')
    if args.coco:
        test(model, dataset='cocoVal')
    if args.OCHuman:
        test(model, dataset='OCHumanVal')
        test(model, dataset='OCHumanTest')
