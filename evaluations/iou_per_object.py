import numpy as np

def iou_per_object(pred_labels, GT_labels):
    # a function to evaluate segmentation using IOU per object   
    # plot_flag: 0 no plotting
    #            1: plot each cell in segmentation to make sure calculation is correct 

    # change the label since some times cell count start at 0 and background as nan
    pred_labels = (pred_labels+1)*(pred_labels>0)
    GT_labels = (GT_labels+1)*(GT_labels>0)

    # get the unique label IDs
    pred_label_list = np.unique(pred_labels)

    # initialize metrics, all object iou as -1
    iou_array = np.zeros([max(pred_label_list)+1,])  - 1
    
    # for each lable ID == each segmented cell in prediction:   
    for label_i in pred_label_list:
        # ignore label ==0, which is the background        
        if(label_i==0):
            continue

        # get the segmented region of cell label_i
        pred_mask = pred_labels == label_i

        # find the ground truth labels of these pixels
        map_list = GT_labels[pred_mask>0]
        map_list = map_list[map_list>0]
        
        if(map_list.shape[0]==0):
            # if there is no overlapping pixels, it means there is no such cell in ground truth, a false positive 
            iou_array[label_i] = 0
        else:
            # if there are overlapping pixels, identify the label value with most counts in ground truth and use that to do IOU calculation            
            values, counts = np.unique(map_list, return_counts=True)
            ind = np.argmax(counts)
            gt_label = values[ind]     
            true_mask = GT_labels == gt_label
            
            # get intersection over union for this object
            iou_array[label_i] = (np.sum(np.logical_and(true_mask, pred_mask)) /
                np.sum(np.logical_or(true_mask, pred_mask)))

    return iou_array