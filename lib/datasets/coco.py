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
from pycocotools.coco import COCO

class coco(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        # coco_train2014/val2014
        datasets.imdb.__init__(self, 'coco' + '_' + image_set + year)
        self._year = year
        # image_set for coco is 'train', 'val' or 'test'
        self._image_set = image_set
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
        self._image_to_id = dict(zip(self._load_image_ids_names()))
        # Default to roidb handler
        self._roidb_handler = self.edgeboxes_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'MSCOCO devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return os.path.join(self._image_path, self._image_index[i])

    def _load_annotations(self):
        """
        Load MSCOCO annotations to memory using the python API.
        """
        # coco/annotations/instances_train2014.json
        filename = os.path.join(self._devkit_path, 'annotations',
                                'instances_' + self._image_set + '.json')
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

    def _load_image_set_index(self):
        image_list = self._load_image_ids_names()[0]
        # sort to match matlab order
        image_list.sort()
        return image_list

    def _clean_list(self, lst, bad_indices):

        # in case manual indices need to be added
        # don't change orig list
        #bad_indices = list(bad_indices)

        for j in reversed(bad_indices):
            del lst[j]
        return

    def _get_default_path(self):
        """
        Return the default path where MSCOCO is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'coco')

    def _get_image_path(self):
        """
        Return the path where the train\val\test image set is (Ex: train2014)
        """
        return os.path.join(self._data_path, self._image_set + self._year)

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

    def edgeboxes_roidb(self):
        """
        Return the database of edgeboxes regions of interest.
        Ground-truth ROIs are also included.

        Side effect: cleans self._image_index from images with no gt or proposals

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_edgeboxes_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:

                roidb, bad_indices = cPickle.load(fid)
                print("{} loaded clean roidb from {}.\nCleaning image list...".format(self._name, cache_file))
                self._clean_list(self._image_index, bad_indices)
            return roidb

        else:
            if self._image_set != 'test':
                gt_roidb = self.gt_roidb()
            else:
                gt_roidb = None

            print('{} generating edgeboxes roidb'.format(self._name))
            eb_roidb = self._load_edgeboxes_roidb(gt_roidb)

            # clean bad indices off image index AND roidbs here
            print("{} cleaning roidb and image list...".format(self._name))
            bad_indices = self._clean_roidbs(gt_roidb, eb_roidb)
            self._clean_list(self._image_index, bad_indices)

            roidb = datasets.imdb.merge_roidbs(gt_roidb, eb_roidb)

            with open(cache_file, 'wb') as fid:
                cPickle.dump((roidb, bad_indices), fid, cPickle.HIGHEST_PROTOCOL)
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

                ref = tmp_ref[0]
                boxes = f[ref].value.T - 1
                # matlab reshapes empty boxes as (2,)
                if boxes.shape == (2,):
                    box_list.append(None)
                else:
                    box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_image_info(self, image_path):
        """
        Load image and bounding boxes info from JSON file in the MSCOCO
        format.

        ONLY NEED bbox indices and class indices from annotation object
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
            x = ann['bbox'][0]
            y = ann['bbox'][1]
            width = ann['bbox'][2]
            height = ann['bbox'][3]
            # convert (xmax,ymax,width,height) to (xmin,ymin,xmax,ymax)
            x1 = x
            y1 = y
            x2 = x + width - 1
            y2 = y + height - 1
            cls = self._cat_id_to_class_index[ann['category_id']]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    # TODO: write evaluation functions
    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        prefix = ''
        if use_salt:
            prefix += '{}_'.format(os.getpid())

        print('Building reverse lookup table for ' + self._image_set + ' set')
        full_img_list = self._load_image_set_index()
        index = dict( zip(full_img_list, xrange(len(full_img_list))) )

        # ILSVRC2014_devkit/results/44503_det_test_aeroplane.txt
        path = os.path.abspath(os.path.join(self._devkit_path, 'results',
                                            prefix))

        filename = path + 'det_' + self._image_set + '_results.txt'
        print 'Writing results to file: {}'.format(filename)
        with open(filename, 'wt') as f:
            for im_ind, image_name in enumerate(self.image_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices (does the ilsvrc expect 1-based indices??)
                    for k in xrange(dets.shape[0]):
                        # f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        f.write('{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.
                                format(index[image_name] + 1,
                                       cls_ind, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return filename

    def _do_matlab_eval(self, filename): # prefix, output_dir='output'):
        # rm_results = self.config['cleanup']

        cache_filename = self._name + '_evaluation_cache.mat'
        cache_filepath = os.path.join(self.cache_path, cache_filename)

        gt_dir = os.path.join(self._data_path, 'bbox', self._image_set)

        path = os.path.join(self._devkit_path, 'evaluation')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        # cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               # .format(self._devkit_path, prefix,
               #         self._image_set, output_dir, int(rm_results))
        cmd += 'evaluate_frcn(\'{:s}\', \'{:s}\', \'{:s}\'); quit;"' \
                .format(filename, gt_dir, cache_filepath)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir=None):
        filename = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(filename) #, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
