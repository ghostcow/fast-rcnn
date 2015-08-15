# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet
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


class imagenet(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        # datasets.imdb.__init__(self, 'voc_' + year + '_' + image_set)
        # ILSVRC2014_train/val
        datasets.imdb.__init__(self, 'ILSVRC' + year + '_' + image_set)
        self._year = year
        # image_set for imagenet is 'train', 'val' or 'test'
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        # self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._image_path = self._get_image_path()
        self._classes = ('__background__', # always index 0
                         'accordion', 'airplane', 'ant', 'antelope',
                         'apple', 'armadillo', 'artichoke', 'axe',
                         'baby bed', 'backpack', 'bagel', 'balance beam',
                         'banana', 'band aid', 'banjo', 'baseball',
                         'basketball', 'bathing cap', 'beaker', 'bear',
                         'bee', 'bell pepper', 'bench', 'bicycle',
                         'binder', 'bird', 'bookshelf', 'bow tie',
                         'bow', 'bowl', 'brassiere', 'burrito',
                         'bus', 'butterfly', 'camel', 'can opener',
                         'car', 'cart', 'cattle', 'cello',
                         'centipede', 'chain saw', 'chair', 'chime',
                         'cocktail shaker', 'coffee maker', 'computer keyboard', 'computer mouse',
                         'corkscrew', 'cream', 'croquet ball', 'crutch',
                         'cucumber', 'cup or mug', 'diaper', 'digital clock',
                         'dishwasher', 'dog', 'domestic cat', 'dragonfly',
                         'drum', 'dumbbell', 'electric fan', 'elephant',
                         'face powder', 'fig', 'filing cabinet', 'flower pot',
                         'flute', 'fox', 'french horn', 'frog',
                         'frying pan', 'giant panda', 'goldfish', 'golf ball',
                         'golfcart', 'guacamole', 'guitar', 'hair dryer',
                         'hair spray', 'hamburger', 'hammer', 'hamster',
                         'harmonica', 'harp', 'hat with a wide brim', 'head cabbage',
                         'helmet', 'hippopotamus', 'horizontal bar', 'horse',
                         'hotdog', 'iPod', 'isopod', 'jellyfish',
                         'koala bear', 'ladle', 'ladybug', 'lamp',
                         'laptop', 'lemon', 'lion', 'lipstick',
                         'lizard', 'lobster', 'maillot', 'maraca',
                         'microphone', 'microwave', 'milk can', 'miniskirt',
                         'monkey', 'motorcycle', 'mushroom', 'nail',
                         'neck brace', 'oboe', 'orange', 'otter',
                         'pencil box', 'pencil sharpener', 'perfume', 'person',
                         'piano', 'pineapple', 'ping-pong ball', 'pitcher',
                         'pizza', 'plastic bag', 'plate rack', 'pomegranate',
                         'popsicle', 'porcupine', 'power drill', 'pretzel',
                         'printer', 'puck', 'punching bag', 'purse',
                         'rabbit', 'racket', 'ray', 'red panda',
                         'refrigerator', 'remote control', 'rubber eraser', 'rugby ball',
                         'ruler', 'salt or pepper shaker', 'saxophone', 'scorpion',
                         'screwdriver', 'seal', 'sheep', 'ski',
                         'skunk', 'snail', 'snake', 'snowmobile',
                         'snowplow', 'soap dispenser', 'soccer ball', 'sofa',
                         'spatula', 'squirrel', 'starfish', 'stethoscope',
                         'stove', 'strainer', 'strawberry', 'stretcher',
                         'sunglasses', 'swimming trunks', 'swine', 'syringe',
                         'table', 'tape player', 'tennis ball', 'tick',
                         'tie', 'tiger', 'toaster', 'traffic light',
                         'train', 'trombone', 'trumpet', 'turtle',
                         'tv or monitor', 'unicycle', 'vacuum', 'violin',
                         'volleyball', 'waffle iron', 'washer', 'water bottle',
                         'watercraft', 'whale', 'wine bottle', 'zebra')

        self._wnids = ('__background__', # also index 0?
                       'n02672831', 'n02691156', 'n02219486', 'n02419796',
                       'n07739125', 'n02454379', 'n07718747', 'n02764044',
                       'n02766320', 'n02769748', 'n07693725', 'n02777292',
                       'n07753592', 'n02786058', 'n02787622', 'n02799071',
                       'n02802426', 'n02807133', 'n02815834', 'n02131653',
                       'n02206856', 'n07720875', 'n02828884', 'n02834778',
                       'n02840245', 'n01503061', 'n02870880', 'n02883205',
                       'n02879718', 'n02880940', 'n02892767', 'n07880968',
                       'n02924116', 'n02274259', 'n02437136', 'n02951585',
                       'n02958343', 'n02970849', 'n02402425', 'n02992211',
                       'n01784675', 'n03000684', 'n03001627', 'n03017168',
                       'n03062245', 'n03063338', 'n03085013', 'n03793489',
                       'n03109150', 'n03128519', 'n03134739', 'n03141823',
                       'n07718472', 'n03797390', 'n03188531', 'n03196217',
                       'n03207941', 'n02084071', 'n02121808', 'n02268443',
                       'n03249569', 'n03255030', 'n03271574', 'n02503517',
                       'n03314780', 'n07753113', 'n03337140', 'n03991062',
                       'n03372029', 'n02118333', 'n03394916', 'n01639765',
                       'n03400231', 'n02510455', 'n01443537', 'n03445777',
                       'n03445924', 'n07583066', 'n03467517', 'n03483316',
                       'n03476991', 'n07697100', 'n03481172', 'n02342885',
                       'n03494278', 'n03495258', 'n03124170', 'n07714571',
                       'n03513137', 'n02398521', 'n03535780', 'n02374451',
                       'n07697537', 'n03584254', 'n01990800', 'n01910747',
                       'n01882714', 'n03633091', 'n02165456', 'n03636649',
                       'n03642806', 'n07749582', 'n02129165', 'n03676483',
                       'n01674464', 'n01982650', 'n03710721', 'n03720891',
                       'n03759954', 'n03761084', 'n03764736', 'n03770439',
                       'n02484322', 'n03790512', 'n07734744', 'n03804744',
                       'n03814639', 'n03838899', 'n07747607', 'n02444819',
                       'n03908618', 'n03908714', 'n03916031', 'n00007846',
                       'n03928116', 'n07753275', 'n03942813', 'n03950228',
                       'n07873807', 'n03958227', 'n03961711', 'n07768694',
                       'n07615774', 'n02346627', 'n03995372', 'n07695742',
                       'n04004767', 'n04019541', 'n04023962', 'n04026417',
                       'n02324045', 'n04039381', 'n01495701', 'n02509815',
                       'n04070727', 'n04074963', 'n04116512', 'n04118538',
                       'n04118776', 'n04131690', 'n04141076', 'n01770393',
                       'n04154565', 'n02076196', 'n02411705', 'n04228054',
                       'n02445715', 'n01944390', 'n01726692', 'n04252077',
                       'n04252225', 'n04254120', 'n04254680', 'n04256520',
                       'n04270147', 'n02355227', 'n02317335', 'n04317175',
                       'n04330267', 'n04332243', 'n07745940', 'n04336792',
                       'n04356056', 'n04371430', 'n02395003', 'n04376876',
                       'n04379243', 'n04392985', 'n04409515', 'n01776313',
                       'n04591157', 'n02129604', 'n04442312', 'n06874185',
                       'n04468005', 'n04487394', 'n03110669', 'n01662784',
                       'n03211117', 'n04509417', 'n04517823', 'n04536866',
                       'n04540053', 'n04542943', 'n04554684', 'n04557648',
                       'n04530566', 'n02062744', 'n04591713', 'n02391049')
        self._wnid_to_ind = dict(zip(self._wnids, xrange(self.num_classes)))
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.edgeboxes_roidb
        # self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
        # clean imdb
        self.clean_imdb()

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return os.path.join(self._image_path, self._image_index[i])

    def _load_image_list(self):
        """
        Enumerate picture list relative to image path.
        Assuming images are at most one level deep.

        Loads from cache to speed things up.
        """
        cache_file = os.path.join(self.cache_path,
                                  self._name + '_image_list.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                image_list = cPickle.load(f)
                print('Loaded image list from: {}'.format(cache_file))
        else:
            image_list = []
            for root,_,files in os.walk(self._image_path):
                folder = '' if (root == self._image_path) else os.path.basename(root)
                images = [os.path.join(folder,file) for file in files]
                image_list.extend(images)

            with open(cache_file, 'wb') as f:
                cPickle.dump(image_list, f)
                print('Saved image list to: {}'.format(cache_file))
        return image_list

    def _load_image_set_index(self):
        image_list = self._load_image_list()
        # sort to match matlab order
        image_list.sort()
        return image_list

    def _clean_list(self, lst, bad_indices):
        for j in reversed(bad_indices):
            del lst[j]
        return

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed. ILSVRC2014_devkit
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'ILSVRC2014_devkit')

    def _get_image_path(self):
        """
        Return the path where the train\val\test image set is
        """
        return os.path.join(self._data_path, 'images', self._image_set)

    def clean_imdb(self):
        """
        Clean image list and roidb from images with no gt or proposals.

        This function uses a cache to speed up it's calls.
        :return:
        """
        gt_file = os.path.abspath(
            os.path.join(self.cache_path, self.name + '_gt_roidb.pkl'))
        prop_file = os.path.abspath(
            os.path.join(self.cache_path, self.name + '_roidb.pkl'))
        bad_index_file = os.path.abspath(
            os.path.join(self.cache_path, self._name + '_bad_indices.pkl'))

        print("{} cleaning imdb...".format(self.name))
        if os.path.exists(bad_index_file):
            with open(bad_index_file, 'rb') as f:
                bad_indices = cPickle.load(f)
            if self._image_set != 'test':
                gt_roidb = self.gt_roidb()
                self._clean_list(gt_roidb, bad_indices)
            prop_roidb = self.roidb
            self._clean_list(prop_roidb, bad_indices)
            self._clean_list(self._image_index, bad_indices)
            print("done.")
            return

        # check for bad indices in roidbs
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            prop_roidb = self.roidb
            assert len(gt_roidb) == len(prop_roidb)
            bad_indices = [i for i in xrange(len(gt_roidb))
                           if gt_roidb[i] is None or prop_roidb[i] is None]
            self._clean_list(gt_roidb, bad_indices)
            self._clean_list(prop_roidb, bad_indices)
            # update gt_roidb cache
            with open(gt_file, 'wb') as f:
                cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)
        else:
            prop_roidb = self.roidb
            bad_indices = [i for i in xrange(len(prop_roidb)) if prop_roidb[i] is None]
            self._clean_list(prop_roidb,bad_indices)

        # finish cleaning
        self._clean_list(self._image_index, bad_indices)

        # update rest of cache
        with open(prop_file, 'wb') as f:
            cPickle.dump(prop_roidb, f, cPickle.HIGHEST_PROTOCOL)
        with open(bad_index_file, 'wb') as f:
            cPickle.dump(bad_indices, f)
        print("done.")

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return gt_roidb

        gt_roidb = [self._load_ilsvrc_annotation(image)
                    for image in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'Wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

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

        print('{} generating edgeboxes roidb'.format(self._name))
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            eb_roidb = self._load_edgeboxes_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, eb_roidb)
        else:
            roidb = self._load_edgeboxes_roidb(None)
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
                ref = tmp_ref[0]
                boxes = f[ref].value.T - 1
                # matlab reshapes empty boxes as (2,)
                if boxes.shape == (2,):
                    box_list.append(None)
                else:
                    box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_ilsvrc_annotation(self, image_path):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'bbox',
                                self._image_set,
                                os.path.splitext(image_path)[0] + '.xml')

        if not os.path.exists(filename):
            return None
        else:
            def get_data_from_tag(node, tag):
                return node.getElementsByTagName(tag)[0].childNodes[0].data

            with open(filename) as f:
                data = minidom.parseString(f.read())
            objs = data.getElementsByTagName('object')
            num_objs = self._count_objects_in_xml(objs, get_data_from_tag)
            height,width = self._get_size_from_xml(data, get_data_from_tag)

            # don't create info if no objects in the image
            if num_objs == 0:
                return None

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = 0
        for obj in objs:
            # ILSVRC pixels are already 0-based (hrmph)
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            # skip bad boxes
            if x1>x2 or y1>y2:
                continue
            # correct about 1000 bboxes with invalid edge indices
            if x2 == width:
                x2 -= 1
            if y2 == height:
                y2 -= 1
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            # update object index
            ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _count_objects_in_xml(self, objs, get_data_from_tag):
        count=0
        for ix, obj in enumerate(objs):
            # ILSVRC pixels are already 0-based (hrmph)
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            if x1<=x2 and y1<=y2:
                count += 1
        return count

    def _get_size_from_xml(self, data, get_data_from_tag):
        s = data.getElementsByTagName('size')[0]
        height = float(get_data_from_tag(s, 'height'))
        width = float(get_data_from_tag(s, 'width'))
        return height, width

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        prefix = ''
        if use_salt:
            prefix += '{}_'.format(os.getpid())

        # ILSVRC2014 devkit compares image index to val.txt indices
        print('Building reverse lookup table for ' + self._image_set + ' set')
        full_img_list = self._load_image_set_index()
        index = dict( zip(full_img_list, xrange(1, len(full_img_list)+1)) )

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
                    # the ILSVRC devkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.
                                format(index[image_name],
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
        cmd += 'evaluate_frcn(\'{:s}\', \'{:s}\', \'{:s}\'); quit;"' \
                .format(filename, gt_dir, cache_filepath)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir=None):
        filename = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.imagenet('train', '2014')
    res = d.roidb
    from IPython import embed; embed()
