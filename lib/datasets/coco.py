# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import h5py
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class coco(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        # coco_train2014/val2014
        datasets.imdb.__init__(self, 'coco' + '_' + image_set + year)
        self._year = year
        # image_set for coco is 'train', 'val' or 'test'
        self._image_set = image_set
        self._image_set_year = image_set + year
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'images')
        self._image_path = self._get_image_path()
        self._classes = ('__background__', # always index 0
                         'person', 'bicycle', 'car', 'motorcycle',
                         'airplane', 'bus', 'train', 'truck',
                         'boat', 'traffic light', 'fire hydrant', 'stop sign',
                         'parking meter', 'bench', 'bird', 'cat',
                         'dog', 'horse', 'sheep', 'cow',
                         'elephant', 'bear', 'zebra', 'giraffe',
                         'backpack', 'umbrella', 'handbag', 'tie',
                         'suitcase', 'frisbee', 'skis', 'snowboard',
                         'sports ball', 'kite', 'baseball bat', 'baseball glove',
                         'skateboard', 'surfboard', 'tennis racket', 'bottle',
                         'wine glass', 'cup', 'fork', 'knife',
                         'spoon', 'bowl', 'banana', 'apple',
                         'sandwich', 'orange', 'broccoli', 'carrot',
                         'hot dog', 'pizza', 'donut', 'cake',
                         'chair', 'couch', 'potted plant', 'bed',
                         'dining table', 'toilet', 'tv', 'laptop',
                         'mouse', 'remote', 'keyboard', 'cell phone',
                         'microwave', 'oven', 'toaster', 'sink',
                         'refrigerator', 'book', 'clock', 'vase',
                         'scissors', 'teddy bear', 'hair drier', 'toothbrush')
        self._image_ext = '.jpg'
        self._annotations = self._load_annotations()
        self._cat_id_to_class_index = self._load_category_ids()
        self._image_index = self._load_image_set_index()
        self._image_to_id = dict(zip(*self._load_image_ids_names()))
        # Default to edgeboxes roidb handler, consider adding constructor param for this
        self._roidb_handler = self.edgeboxes_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'MSCOCO devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

        # clean image list from images with no gt or no proposals
        self.clean_imdb()


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        # return os.path.abspath(os.path.join(self._image_path, self._image_index[i]))
        return os.path.join(self._image_path, self._image_index[i])

    def _load_annotations(self):
        """
        Load MSCOCO annotations to memory using the python API.
        """
        # coco/annotations/instances_train2014.json
        filename = os.path.join(self._devkit_path, 'annotations',
                                'instances_' + self._image_set_year + '.json')
        assert os.path.exists(filename)
        return COCO(filename)


    def _load_category_ids(self):
        """
        Returns a map from MSCOCO category (class) ids to our class ids
        """
        classes = list(self._classes[1:])
        cat_ids = self._annotations.getCatIds(classes)
        # don't include background class 0
        return dict(zip(cat_ids, xrange(1,self.num_classes)))

    def _load_image_ids_names(self):
        """
        Loads image ids and names from COCO annotations
        Returns image name list and image id list
        """
        image_ids = self._annotations.getImgIds()
        image_infos = self._annotations.loadImgs(image_ids)
        image_list = [image_info['file_name'] for image_info in image_infos]
        return image_list, image_ids

    def _clean_list(self, lst, bad_indices):

        # in case manual indices need to be added
        # don't change orig list
        #bad_indices = list(bad_indices)

        for j in reversed(bad_indices):
            del lst[j]
        return

    # def _clean_list(self, lst, bad_indices):
    #     """
    #     alt. implementation that 'cleans' lists down to 1 item,
    #     for debugging purposes.
    #     """
    #     n = len(lst)
    #     for i in reversed(xrange(1,n)):
    #         del lst[i]
    #     return

    def _load_image_set_index(self):
        image_list = self._load_image_ids_names()[0]
        # sort to match matlab order
        image_list.sort()
        return image_list

    def _get_default_path(self):
        """
        Return the default path where MSCOCO is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'coco')

    def _get_image_path(self):
        """
        Return the path where the train\val\test image set is (Ex: train2014)
        """
        return os.path.join(self._data_path, self._image_set_year)

    def clean_imdb(self):
        """
        Clean image list and roidb from images with no gt or proposals
        Update cache files accordingly.
        :return:
        """
        bad_index_file = os.path.abspath(
            os.path.join(self.cache_path, self._name + '_bad_indices.pkl'))

        print("{} cleaning roidb and image list...".format(self._name))
        # if cache file exists, that means that a clean roidb has been saved
        if os.path.exists(bad_index_file):
            with open(bad_index_file, 'rb') as f:
                bad_indices = cPickle.load(f)
                self._clean_list(self._image_index, bad_indices)
        else:
            gt_roidb = self.gt_roidb()
            proposals_roidb = self.roidb

            # clean image index and roidbs
            bad_indices = self._clean_roidbs(gt_roidb, proposals_roidb)
            self._clean_list(self._image_index, bad_indices)

            # update cache and save index list
            gt_file = os.path.join(self.cache_path,
                          self.name + '_gt_roidb.pkl')
            roidb_file = os.path.join(self.cache_path,
                          self.name + '_roidb.pkl')
            with open(bad_index_file, 'wb') as f:
                cPickle.dump(bad_indices, f)
            with open(gt_file, 'wb') as f:
                cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)
            with open(roidb_file, 'wb') as f:
                cPickle.dump(proposals_roidb, f, cPickle.HIGHEST_PROTOCOL)
        print("done.")
        return

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_image_info(image)
                    for image in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_image_info(self, image_path):
        """
        Load image and bounding boxes info from JSON file in the MSCOCO
        format.

        Bboxes are 0-indexed absolute coordinates, see issue:
        https://github.com/pdollar/coco/issues/5#issuecomment-129899109
        """
        img_id = self._image_to_id[image_path]
        ann_id = self._annotations.getAnnIds(img_id)
        anns = self._annotations.loadAnns(ann_id)

        num_objs = len(anns)
        if num_objs == 0:
            return None

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, ann in enumerate(anns):
            bb = ann['bbox']
            # convert coco style bbox annotation to pixel coordinates
            # dirty hack: some bounding boxes are less than a pixel wide
            x, y, width, height = [bb[0], bb[1], bb[2], bb[3]]
            x1, y1, x2, y2 = [x, y,
                              x+width-min(1,width),
                              y+height-min(1,height)]
            # sanity check
            assert x1 <= x2 and y1 <= y2
            cls = self._cat_id_to_class_index[ann['category_id']]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def edgeboxes_roidb(self):
        """
        Return the database of edgeboxes regions of interest.
        Ground-truth ROIs are also included.

        Side effect: cleans self._image_index from images with no gt or proposals

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
                print("{} loaded roidb from {}".format(self._name, cache_file))
            return roidb

        else:
            if self._image_set != 'test':
                gt_roidb = self.gt_roidb()
            else:
                gt_roidb = None

            print('{} generating edgeboxes roidb'.format(self._name))
            eb_roidb = self._load_edgeboxes_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, eb_roidb)

            with open(cache_file, 'wb') as fid:
                cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
                print 'Wrote roidb to {}'.format(cache_file)

            return roidb

    def _load_edgeboxes_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'edgeboxes_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
                'Edgeboxes search data not found at: {}'.format(filename)

        with h5py.File(filename, 'r') as f:
            box_list=[]
            for tmp_ref in f['boxes']:
                # transpose and 0-index proposals
                # because they are stored by matlab in hdf5 format
                ref = tmp_ref[0]
                boxes = f[ref].value.T - 1
                # matlab stores empty boxes as shape (2,)
                if boxes.shape == (2,):
                    box_list.append(None)
                else:
                    box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _clean_roidbs(self, a, b):
        """
        Remove roi info of images with no proposals OR no gt boxes.

        Assume a or b are not None
        """
        if a and b:
            assert len(a) == len(b)
            bad_indices = [i for i in xrange(len(a)) if a[i] is None or b[i] is None]
        elif a is None:
            bad_indices = [i for i in xrange(len(b)) if b[i] is None]
        elif b is None:
            bad_indices = [i for i in xrange(len(a)) if a[i] is None]

        if a:
            self._clean_list(a,bad_indices)
        if b:
            self._clean_list(b,bad_indices)

        return bad_indices

    def _write_coco_results_file(self, all_boxes):
        """
        Results must be written to file in JSON format.
        one dict per instance:
        [{
            "image_id" : int,
            "category_id" : int,
            "bbox : [x,y,width,height],
            "score" : float,
        }]
        :param all_boxes:
        :return:
        """
        use_salt = self.config['use_salt']
        prefix = 'instances_'
        if use_salt:
            prefix = '{}_{}'.format(os.getpid(), prefix)

        path = os.path.abspath(os.path.join(self._devkit_path, 'results',
                                            prefix))

        filename = path + self._image_set_year + '_results.json'
        print 'Writing results to file: {}'.format(filename)
        results = []
        with open(filename, 'wt') as f:
            for im_ind, image_name in enumerate(self.image_index):
                for cls_ind, cls in enumerate(self.classes):

                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind].tolist()
                    if dets == []:
                        continue

                    image_id = self._image_to_id[image_name]
                    cat_id = self._annotations.getCatIds(cls)[0]
                    for k in xrange(len(dets)):
                        x1,y1,x2,y2 = [dets[k][0], dets[k][1],
                                       dets[k][2], dets[k][3]]
                        # convert back to coco style bbox annotation
                        x,y,width,height = [x1-1, y1-1, x2-x1+1, y2-y1+1]
                        bbox = [x, y, width, height]
                        score = dets[k][-1]
                        res = {'image_id' : image_id,
                               'category_id' : cat_id,
                               'bbox' : bbox,
                               'score' : score}
                        results.append(res)
            json.dump(results,f)
        return filename

    def evaluate_detections(self, all_boxes, output_dir=None):
        resFile = self._write_coco_results_file(all_boxes)
        cocoGt = self._annotations
        cocoDt = cocoGt.loadRes(resFile)
        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt)
        # useSegm should default to 0
        #cocoEval.params.useSegm = 0
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.coco('val', '2014')
    res = d.roidb
    from IPython import embed; embed()
