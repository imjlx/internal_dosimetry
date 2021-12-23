import SimpleITK as sitk
import numpy as np


def GetLargestConnectedComponent(binarysitk_image):
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(4)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(binarysitk_image.GetDirection())
    outmask_sitk.SetSpacing(binarysitk_image.GetSpacing())
    outmask_sitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmask_sitk


# 逻辑与操作
def GetMaskImage(sitk_src, sitk_mask, replacevalue=0):
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_mask = sitk.GetArrayFromImage(sitk_mask)
    array_out = array_src.copy()
    array_out[array_mask < 200] = replacevalue
    array_out[array_mask > 200] = 255
    outmask_sitk = sitk.GetImageFromArray(array_out)
    outmask_sitk.SetDirection(sitk_src.GetDirection())
    outmask_sitk.SetSpacing(sitk_src.GetSpacing())
    outmask_sitk.SetOrigin(sitk_src.GetOrigin())
    return outmask_sitk


def removefat(body, low, up):
    sitk_seg = sitk.BinaryThreshold(body, lowerThreshold=low, upperThreshold=up, insideValue=255, outsideValue=0)
    sitk_open = sitk.BinaryMorphologicalOpening(sitk_seg)
    sitk_open = GetLargestConnectedComponent(sitk_open)

    ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
    ConnectedThresholdImageFilter.SetLower(0)
    ConnectedThresholdImageFilter.SetUpper(0)
    ConnectedThresholdImageFilter.SetSeedList(
        [(0, 0, 0), (sitk_open.GetSize()[0] - 1, sitk_open.GetSize()[1] - 1, sitk_open.GetSize()[2] - 1)])
    bodymask = ConnectedThresholdImageFilter.Execute(sitk_open)
    bodymask = sitk.ShiftScale(bodymask, -1, -1)

    bodymask = sitk.Cast(bodymask, sitk.sitkFloat32)
    sitk_src = sitk.Cast(body, sitk.sitkFloat32)
    result = bodymask * sitk_src
    return result


def bedmask(vol):
    sitk_seg = sitk.BinaryThreshold(vol, lowerThreshold=1, upperThreshold=3000, insideValue=255, outsideValue=0)
    # 利用种子生成算法，填充空气
    ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
    ConnectedThresholdImageFilter.SetLower(0)
    ConnectedThresholdImageFilter.SetUpper(0)
    ConnectedThresholdImageFilter.SetSeedList([(0, 0, 0), (sitk_seg.GetSize()[0] - 1, sitk_seg.GetSize()[1] - 1, 0)])

    # 得到body的mask，此时body部分是0，所以反转一下
    bedmask = ConnectedThresholdImageFilter.Execute(sitk_seg)
    bedmask = sitk.ShiftScale(bedmask, -1, -1)
    bedmask = GetLargestConnectedComponent(bedmask)

    # 用bodymask减去threshold，得到初步的lung的mask
    result = sitk.Cast(bedmask, sitk.sitkFloat32)
    vol = sitk.Cast(vol, sitk.sitkFloat32)
    result = result * vol
    return result


def remove_bed(read_path, save_path):
    sitk_src = sitk.ReadImage(read_path)
    sitk_src = sitk_src + 1024
    sitk_seg = sitk.BinaryThreshold(sitk_src, lowerThreshold=600, upperThreshold=3000, insideValue=255, outsideValue=0)

    sitk_open = sitk.BinaryMorphologicalOpening(sitk_seg)
    sitk_open = GetLargestConnectedComponent(sitk_open)

    ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
    ConnectedThresholdImageFilter.SetLower(0)
    ConnectedThresholdImageFilter.SetUpper(0)
    ConnectedThresholdImageFilter.SetSeedList([(0, 0, 0), (sitk_open.GetSize()[0] - 1, sitk_open.GetSize()[1] - 1, sitk_open.GetSize()[2] - 1)])
    bodymask = ConnectedThresholdImageFilter.Execute(sitk_open)
    bodymask = sitk.ShiftScale(bodymask, -1, -1)

    bodymask = sitk.Cast(bodymask, sitk.sitkFloat32)
    sitk_src = sitk.Cast(sitk_src, sitk.sitkFloat32)
    result = bodymask * sitk_src
    Resampledimage = sitk.Resample(result, result.GetSize(),
                               sitk.TranslationTransform(3), sitk.sitkNearestNeighbor,
                               result.GetOrigin(), result.GetSpacing(),
                               result.GetDirection(), 0.0, sitk.sitkFloat32)
    sitk.WriteImage(Resampledimage-1024, save_path)



# sitk_src = sitk.ReadImage('F://PET-CT_segment//Patient10_CT.nii')
# sitk_src = sitk_src + 1024
# sitk_seg = sitk.BinaryThreshold(sitk_src, lowerThreshold=600, upperThreshold=3000, insideValue=255, outsideValue=0)
#
# sitk_open = sitk.BinaryMorphologicalOpening(sitk_seg)
# sitk_open = GetLargestConnectedCompont(sitk_open)
#
# ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
# ConnectedThresholdImageFilter.SetLower(0)
# ConnectedThresholdImageFilter.SetUpper(0)
# ConnectedThresholdImageFilter.SetSeedList([(0, 0, 0), (sitk_open.GetSize()[0] - 1, sitk_open.GetSize()[1] - 1, sitk_open.GetSize()[2] - 1)])
# bodymask = ConnectedThresholdImageFilter.Execute(sitk_open)
# bodymask = sitk.ShiftScale(bodymask, -1, -1)
#
# bodymask = sitk.Cast(bodymask, sitk.sitkFloat32)
# sitk_src = sitk.Cast(sitk_src, sitk.sitkFloat32)
# result = bodymask * sitk_src
# Resampledimage = sitk.Resample(result, result.GetSize(),
#                                sitk.TranslationTransform(3), sitk.sitkNearestNeighbor,
#                                result.GetOrigin(), result.GetSpacing(),
#                                result.GetDirection(), 0.0, sitk.sitkFloat32)
# bed = sitk_src - Resampledimage
# bed = bedmask(bed)
# allimage = sitk.ReadImage("sitchtransform.nii")
# bed = sitk.Cast(bed, allimage.GetPixelID())
# bed.SetOrigin(allimage.GetOrigin())
# bed = sitk.Resample(bed, allimage.GetSize(),
#                                sitk.TranslationTransform(3), sitk.sitkNearestNeighbor,
#                                allimage.GetOrigin(), allimage.GetSpacing(),
#                                allimage.GetDirection(), 0.0, allimage.GetPixelID())
# allimage = allimage + bed
# position = 502.5 - 30.45
# for i in range(5):
#     position += i * 40
#     print(position + 20)
#     colli = int(position/allimage.GetSpacing()[2])
#     lenth = int(40/allimage.GetSpacing()[2])
#     allimage = allimage - 1024
#     array = sitk.GetArrayFromImage(allimage)
#     x = np.linspace(-allimage.GetSize()[0]*allimage.GetSpacing()[0]/2, allimage.GetSize()[0]*allimage.GetSpacing()[0]/2, allimage.GetSize()[0])
#     y = np.linspace(-allimage.GetSize()[1]*allimage.GetSpacing()[1]/2, allimage.GetSize()[1]*allimage.GetSpacing()[1]/2, allimage.GetSize()[1])
#     [xx, yy] = np.meshgrid(x, y)
#     distance = np.sqrt(xx**2+yy**2)
#     new = np.zeros((allimage.GetSize()[1], allimage.GetSize()[0]))
#     new[distance > 270] = 3000
#     new = np.expand_dims(new, axis=0)
#     new = np.repeat(new, allimage.GetSize()[2], axis=0)
#     new[colli:colli+lenth+1] = 0
#     collimater = sitk.GetImageFromArray(new)
#     collimater.SetOrigin(allimage.GetOrigin())
#     collimater.SetSpacing(allimage.GetSpacing())
#     collimater.SetDirection(allimage.GetDirection())
#     allimage = sitk.Cast(allimage, sitk.sitkInt16)
#     collimater = sitk.Cast(collimater, sitk.sitkInt16)
#     allimage = allimage + collimater
#     allimage.SetOrigin((0, 0, 0))
#     sitk.WriteImage(allimage, "sitchtransformwithcollimator"+str(i)+".nii")