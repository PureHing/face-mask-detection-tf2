#!/usr/bin/env python
# -*- coding: utf8 -*-

from xml.etree import ElementTree

from lxml import etree
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# rootPath = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
# sys.path.append(rootPath)
# print(rootPath)
XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'


# pascalVocReader readers the voc xml files parse it
class PascalVocReader:
    """
    this class will be used to get transfered width and height from voc xml files
    """

    def __init__(self, filepath, width, height):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        self.width = width
        self.height = height

        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, bndbox, width, height):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        width_trans = (xmax - xmin) / width *self.width
        height_trans = (ymax - ymin) / height*self.height
        points = [width_trans, height_trans]
        self.shapes.append((points))

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        pic_size = xmltree.find('size')
        size = (int(pic_size.find('width').text), int(pic_size.find('height').text))
        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            self.addShape(bndbox, *size)
        return True


class create_w_h_txt:
    def __init__(self, vocxml_path, txt_path,IMAGE_W,IMAGE_H):

        self.voc_path = vocxml_path
        self.txt_path = txt_path
        self.image_w=IMAGE_W
        self.image_h=IMAGE_H
        # print(self.voc_path,self.txt_path)

    def _gether_w_h(self):
        pass

    def _write_to_txt(self):
        pass

    def process_file(self):
        file_w = open(self.txt_path, 'a')
        # print (self.txt_path)
        for file in os.listdir(self.voc_path):
            file_path = os.path.join(self.voc_path, file)
            xml_parse = PascalVocReader(file_path, self.image_w,self.image_h)
            data = xml_parse.getShapes()
            for w, h in data:
                txtstr = str(w) + ' ' + str(h) + '\n'
                # print (txtstr)
                file_w.write(txtstr)
        file_w.close()


class kMean_parse:
    def __init__(self, path_txt, k):
        self.path = path_txt
        self.km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=3000000, tol=1e-5, random_state=0)
        self._load_data()

    def _load_data(self):
        self.data = np.loadtxt(self.path)

    def _iou(self, box, clusters):
        """
        Calculates the Intersection over Union (IoU) between a box and k clusters.
        :param box: tuple or array, shifted to the origin (i. e. width and height)
        :param clusters: numpy array of shape (k, 2) where k is the number of clusters
        :return: numpy array of shape (k, 0) where k is the number of clusters
        """
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("Box has no area")

        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = intersection / (box_area + cluster_area - intersection)

        return iou_

    def _avg_iou(self, boxes, clusters):
        """
        Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
        :param boxes: numpy array of shape (r, 2), where r is the number of rows
        :param clusters: numpy array of shape (k, 2) where k is the number of clusters
        :return: average IoU as a single float
        """
        return np.mean([np.max(self._iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

    def parse_data(self):
        self.y_k = self.km.fit_predict(self.data)
        sort_arr = np.sort(self.km.cluster_centers_, axis=0)
        acc = self._avg_iou(self.data, sort_arr) * 100
        sort_arr[:, [0, 1]] = sort_arr[:, [1, 0]]
        filename =  'priors_h_w_anchors.txt'
        np.savetxt(filename, sort_arr, fmt='%d', delimiter=',', newline=' , ')
        print(f'Accuracy: {acc:.2f}% , Anchors file save into {os.getcwd()+"/"+filename} ')

    def plot_data(self):
        plt.scatter(self.data[self.y_k == 0, 0], self.data[self.y_k == 0, 1], s=50, c="orange", marker="o",
                    label="cluster 1")
        plt.scatter(self.data[self.y_k == 1, 0], self.data[self.y_k == 1, 1], s=50, c="green", marker="s",
                    label="cluster 2")
        plt.scatter(self.data[self.y_k == 2, 0], self.data[self.y_k == 2, 1], s=50, c="blue", marker="^",
                    label="cluster 3")
        plt.scatter(self.data[self.y_k == 3, 0], self.data[self.y_k == 3, 1], s=50, c="gray", marker="*",
                    label="cluster 4")
        plt.scatter(self.data[self.y_k == 4, 0], self.data[self.y_k == 4, 1], s=50, c="yellow", marker="d",
                    label="cluster 5")
        # draw the centers
        plt.scatter(self.km.cluster_centers_[:, 0], self.km.cluster_centers_[:, 1], s=250, marker="*", c="red",
                    label="cluster center")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':

    xml_path = "Maskdata/Annotations"

    whtxt = create_w_h_txt(xml_path, 'data.txt',320,240)
    whtxt.process_file()
    kmean_parse = kMean_parse('data.txt', k=16)
    kmean_parse.parse_data()
    kmean_parse.plot_data()
