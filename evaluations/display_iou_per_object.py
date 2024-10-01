import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import os

def display_iou_per_object(seg_labels, GT_labels, result_folder):
    # a function to display IOU per object and plot to help understanding
    # seg_labels: cell segmentation labels
    # GT_labels: ground truth labels
    # note: these labels could be and usually are in different orders and might also have different number of objects

    # change the label since some times cell count start at 0 and background as nan
    seg_labels = (seg_labels+1)*(seg_labels>0)
    GT_labels = (GT_labels+1)*(GT_labels>0)

    # get the unique label IDs
    seg_label_list = np.unique(seg_labels)

    # build the colormap with iterative tab10
    tab20 = cm.get_cmap('tab20', 10)
    for i in range(10):
        tab20.colors = np.concatenate([tab20.colors,tab20.colors],axis=0)    
    tab20.colors = np.concatenate([np.zeros([1,4]),tab20.colors],axis=0)

    newmap = cm.get_cmap('tab20', 4000+1)
    newmap.colors = tab20.colors[0:4000+1,:]
    newmap.colors[0,:]  = np.zeros([1,4])
    newmap.colors[0,3] = 1

    # for each lable ID == each segmented cell in prediction:   
    for label_i in seg_label_list:
        # ignore label ==0, which is the background        
        if(label_i==0):
            continue

        # get the segmented region of cell label_i
        pred_mask = seg_labels == label_i

        # find the ground truth labels of these pixels
        map_list = GT_labels[pred_mask>0]
        map_list = map_list[map_list>0]

        # if there are overlapping pixels
        if(map_list.shape[0]>0):
            
            values, counts = np.unique(map_list, return_counts=True)

            ind = np.argmax(counts)
            gt_label = values[ind]     

            true_mask = GT_labels == gt_label

            union = pred_mask.astype(int) + 2 * true_mask.astype(int)

            [ind_x, ind_y] = np.where(union>0)

            min_x = max(min(ind_x)-2,0)
            max_x = min(max(ind_x)+2,pred_mask.shape[0])
            min_y = max(min(ind_y)-2,0)
            max_y = min(max(ind_y)+2,pred_mask.shape[1])
            

            this_object_iou = (np.sum(np.logical_and(true_mask[min_x:max_x, min_y:max_y], pred_mask[min_x:max_x, min_y:max_y])) /
                np.sum(np.logical_or(true_mask[min_x:max_x, min_y:max_y], pred_mask[min_x:max_x, min_y:max_y])))
            

            fig, ax = plt.subplots(2,3, figsize=(12,6), dpi=256, facecolor='w', edgecolor='k')
            ax[0,0].imshow(pred_mask[min_x:max_x, min_y:max_y].astype(int)*1, cmap=newmap,interpolation='none',vmax = 4001,vmin = 0)
            ax[0,0].axis('off')
            ax[0,0].title.set_text('seg_label='+"%d" % label_i)

            ax[0,1].imshow(true_mask[min_x:max_x, min_y:max_y].astype(int)*2, cmap=newmap,interpolation='none',vmax = 4001,vmin = 0)
            ax[0,1].axis('off')
            ax[0,1].title.set_text('GT_label='+"%d" % gt_label)
            
            ax[0,2].imshow( pred_mask[min_x:max_x, min_y:max_y].astype(int)*1 + 2 * true_mask[min_x:max_x, min_y:max_y].astype(int), cmap=newmap,interpolation='none',vmax = 4001,vmin = 0)
            ax[0,2].axis('off')
            ax[0,2].title.set_text('iou='+"%.2f" % this_object_iou)

            X, Y = np.meshgrid(np.arange(0,seg_labels.shape[1]), np.arange(0,seg_labels.shape[0]))
            
            ax[1,0].imshow(seg_labels, cmap=newmap,interpolation='none',vmax = 4001,vmin = 0)
            ax[1,0].axis('off')
            ax[1,0].plot( [min_y,max_y,max_y,min_y,min_y],[min_x,min_x,max_x,max_x,min_x])
            ax[1,0].contour(X, Y, pred_mask,1,colors=('yellow'),linewidths=0.7)
                                    
            ax[1,1].imshow(GT_labels, cmap=newmap,interpolation='none',vmax = 4001,vmin = 0)
            ax[1,1].axis('off')
            ax[1,1].contour(X, Y, true_mask,1,colors=('yellow'),linewidths=0.7)
            ax[1,1].plot( [min_y,max_y,max_y,min_y,min_y],[min_x,min_x,max_x,max_x,min_x])
            ax[1,2].axis('off')

            # save the plot
            fig.savefig(os.path.join(result_folder,'seg'+str(label_i) + '_GT'+str(gt_label)+'_iou.png'))

            plt.close(fig)      

       