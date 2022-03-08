#!/usr/bin/env python
# encoding: utf-8
"""
    @File       : RemoveBedFilter.py
    @Time       : 2022/3/2 16:27
    @Author     : Haoran Jia
    @license    : Copyright(c) 2022 Haoran Jia. All rights reserved.
    @contact    : 21211140001@m.fudan.edu.cn
    @Description：
"""

import SimpleITK as sitk
import numpy as np


class RemoveBedFilter(object):
    """
    作用: 给定ct, 去掉床, 将背景设置为-1024
    """
    def __init__(self, fpath):
        self.img_origin = sitk.ReadImage(fpath)
        self.img_main_body = None
        self.img_bed = None

        self.mask_main_body = None
        self.mask_lung = None

        self.thresh1 = None
        self.thresh2 = None

    def Execute(self) -> sitk.Image:
        """
        通过将人主体和肺的mask结合, 得到完整人体遮罩, 进而得到去除床和背景的人体
        :return:
        """
        # sitk.Show(self.img_origin)
        mask_main_body = self.CreateMainBodyMask()
        mask_lung = self.CreateLungMask()
        mask_body = mask_main_body + mask_lung

        # sitk.Show(sitk.LabelOverlay(sitk.Cast(sitk.RescaleIntensity(self.img_origin), sitk.sitkUInt8), mask_body))

        mask_solid = self.CreateSolidMask(mask_body)

        img_body = self.ApplyMask(self.img_origin, mask_solid)

        # sitk.Show(sitk.LabelOverlay(sitk.Cast(sitk.RescaleIntensity(self.img_origin), sitk.sitkUInt8), mask_solid))
        # sitk.Show(img_body)
        return img_body

    def CreateMainBodyMask(self):
        """
        生成人主体的mask
        :return: 生成的mask, uint8, 背景0, 前景255
        """
        # 为节省内存, 将原始图像区间转到0-255, 修改数据类型
        img = sitk.Cast(sitk.RescaleIntensity(self.img_origin), sitk.sitkUInt8)

        # 进行自动阈值分割, 进行形态学修补
        mask, self.thresh1 = self.SimpleThreshold(img, isUpperThresh=False, isOpining=False, isClosing=False)
        # 选择最大连通域作为主要人体的mask
        self.mask_main_body = self.ConnectedComponent(mask, 1)
        # sitk.Show(sitk.LabelOverlay(img, self.mask_main_body))

        return self.mask_main_body

    def CreateLungMask(self):
        """
        生成肺部的mask
        :return: 生成的mask, uint8, 背景0, 前景255
        """
        # 为节省内存, 将原始图像区间转到0-255, 修改数据类型
        img = sitk.Cast(sitk.RescaleIntensity(self.img_origin), sitk.sitkUInt8)
        # 进行自动阈值分割, 进行形态学修补
        mask_all, self.thresh1 = self.SimpleThreshold(img, isOpining=False, isClosing=False)
        # 将mask用在原始图像中
        img_masked = self.ApplyMask(self.img_origin, mask_all)
        # sitk.Show(img_masked)

        # 原始图像减去遮罩掉的部分, 得到剩余的肺和其他部分
        img_lung_other = sitk.ShiftScale(self.img_origin, 1024) - sitk.ShiftScale(img_masked, 1024)
        img_lung_other = sitk.Cast(sitk.RescaleIntensity(img_lung_other), sitk.sitkUInt8)
        # sitk.Show(img_lung_other, "lung + other")

        mask, self.thresh2 = self.SimpleThreshold(img_lung_other)
        # sitk.Show(mask)
        # lung = self.ConnectedComponent(mask, 2)
        # sitk.Show(lung)
        self.mask_lung = self.ConnectedComponent_FindLung(mask)

        return self.mask_lung

    @staticmethod
    def SimpleThreshold(img, threshold=None, isUpperThresh: bool = False,
                        isOpining: bool = False, isClosing: bool = False):
        """
        对输入的原始图像进行阈值二值化, 可以自动阈值或给定阈值
        :param img: 原始图像
        :param threshold: int或str
        :param isUpperThresh: 是否设置一个最高的上限
        :param isOpining: 是否进行开操作
        :param isClosing: 是否进行闭操作
        :return: 阈值处理后的mask
        """
        if isUpperThresh:   # 对最大值进行限制
            img = sitk.Threshold(img, lower=0, upper=80, outsideValue=0)

        # 根据threshold类型选择不同的阈值方法
        if isinstance(threshold, int):
            img_thresh = sitk.BinaryThreshold(img, lowerThreshold=threshold, upperThreshold=255,
                                              insideValue=255, outsideValue=0)
        elif isinstance(threshold, str) or (threshold is None):
            if (threshold == "otsu") or (threshold is None):
                otsu_filter = sitk.OtsuThresholdImageFilter()
                otsu_filter.SetInsideValue(0)
                otsu_filter.SetOutsideValue(255)
                img_thresh = otsu_filter.Execute(img)
                threshold = otsu_filter.GetThreshold()
            else:
                raise TypeError("Unsupported method!")
        else:
            raise TypeError("Please input int or str threshold")

        if isOpining:    # 是否进行形态学操作
            img_thresh = sitk.BinaryMorphologicalOpening(img_thresh)
        if isClosing:
            img_thresh = sitk.BinaryMorphologicalClosing(img_thresh, kernelRadius=(10, 10, 10))

        return img_thresh, threshold

    @staticmethod
    def ConnectedComponent(img: sitk.Image, size_order: int) -> sitk.Image:
        """
        寻找二值化得到的mask的连通域, 按照物理体积
        :param img: 二值化得到的mask
        :param size_order: 体积的排序
        :return: 第size_order大的连通域的mask
        """
        # 生成按照连通域标记的图像
        connected_component_filter = sitk.ConnectedComponentImageFilter()
        img_label = connected_component_filter.Execute(img)
        # object_count = connected_component_filter.GetObjectCount()
        # sitk.Show(img_label)

        # 对标记图像进行统计
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.SetNumberOfThreads(8)     # 设置运算的线程数(用于并行运算?)
        stats.Execute(img_label, img)
        labels = stats.GetLabels()  # 获取全部的标签值(tuple)
        size = [stats.GetPhysicalSize(label) for label in labels]  # 计算相应的物理大小

        size_sort = np.argsort(size)    # 大小坐标排序, 升序
        max_label = labels[size_sort[-size_order]]    # 找到最大值的标签

        # 得到对应标签的mask (array)
        arr_label = sitk.GetArrayFromImage(img_label)
        arr_mask = np.zeros_like(arr_label)
        arr_mask[arr_label == max_label] = 255

        # 转换为Image
        img_mask = sitk.GetImageFromArray(arr_mask)
        img_mask.CopyInformation(img)
        img_mask = sitk.Cast(img_mask, sitk.sitkUInt8)

        return img_mask

    @staticmethod
    def ConnectedComponent_FindLung(img: sitk.Image) -> sitk.Image:
        """
        利用二次阈值处理后的图像, 找到肺部的mask
        :param img: 二次阈值处理后的图像, 在一次阈值中已去掉了人和床等大部分区域
        :return: 肺部mask
        """
        # 生成按照连通域标记的图像
        img_label = sitk.ConnectedComponent(img)

        # 对标记图像进行统计
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.SetNumberOfThreads(8)  # 设置运算的线程数(用于并行运算?)
        stats.Execute(img_label, img)

        labels = stats.GetLabels()  # 获取全部的标签值(tuple)
        sizes = [stats.GetPhysicalSize(label) for label in labels]  # 计算相应的物理大小
        sizes_sort = np.argsort(sizes)  # 大小坐标排序, 升序

        labels = [labels[index] for index in sizes_sort[-5:]]  # 找到前五个最大值的标签
        ellipsoid_diameters = [stats.GetEquivalentEllipsoidDiameter(label) for label in labels]

        lung_index = []
        for i, d in enumerate(ellipsoid_diameters):
            # print(d)
            if (max(d) < 250) and (min(d) > 30) and (max(d) / min(d) < 3):
                lung_index.append(i)

        if len(lung_index) == 0:     # 没找到连通域, 抛出问题
            raise IndexError("found no possible lung")
        elif len(lung_index) == 1:
            labels = [labels[lung_index[0]]]
            print(f"Waring!!!!Found only 1 component!!! Please check MANUALLY!!")
        elif len(lung_index) == 2:  # 正好找到两个连通域, 大概率是两个肺
            labels = [labels[i] for i in lung_index]
        else:   # 找到多于两个连通域, 取体积最大的两个
            labels = [labels[i] for i in lung_index][-2:]
            print(f"Waring!!!!Found {len(lung_index)} components!!! Please check MANUALLY!!")

        # 得到对应标签的mask (array)
        arr_label = sitk.GetArrayFromImage(img_label)   # 原始标记img的arr
        arr_mask = np.zeros_like(arr_label)
        for i in range(len(labels)):
            arr_mask[arr_label == labels[i]] = 255

        # 转换为Image
        img_mask = sitk.GetImageFromArray(arr_mask)
        img_mask.CopyInformation(img)
        img_mask = sitk.Cast(img_mask, sitk.sitkUInt8)
        # sitk.Show(img_mask)

        return img_mask

    @staticmethod
    def CreateSolidMask(img_mask):
        """
        从外部对背景进行生长, 得到封闭无空洞的前景
        :param img_mask: 有空洞的mask
        :return: 没有肺部, 但是补全了其他低密度器官的mask
        """
        # 通过背景的生长分割, 实现对身体内部空洞的填补
        solid_mask = sitk.ConnectedThreshold(img_mask, seedList=[(0, 0, 0)], lower=0, upper=0)
        solid_mask = sitk.ShiftScale(solid_mask, shift=-1, scale=-255)
        # sitk.Show(solid_mask)
        return solid_mask

    @staticmethod
    def ApplyMask(img: sitk.Image, mask: sitk.Image):
        """
        利用mask对原图进行遮罩
        :param img: 完整的原图像, 要求取值在(-1024-3072)
        :param mask: 遮罩图
        :return: 处理后的图片, 数据类型等与img完全相同
        """
        img = sitk.ShiftScale(img, shift=1024, scale=1)
        mask = sitk.Cast(sitk.RescaleIntensity(mask, 0, 1), img.GetPixelID())
        img_masked = sitk.Cast(sitk.Multiply(img, mask), img.GetPixelID())
        img_masked = sitk.ShiftScale(img_masked, shift=-1024, scale=1)

        return img_masked


if __name__ == '__main__':
    remover = RemoveBedFilter(r"E:\internal_dosimetry\dataset\patient1\hdr\ct.hdr")
    # remover.CreateBasicSeg()
    # remover.CreateLungMask()
    remover.Execute()

    pass
