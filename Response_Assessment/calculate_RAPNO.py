import os
import numpy as np
import nibabel as nib
from skimage.morphology import label
from skimage.measure import regionprops
from glob import glob
import pandas as pd
from tqdm import tqdm
from rapno_function import rapno

def get_rapno(mask,vox_x,vox_z,pic_dir,background, num_lesions, make_csv=True, all_lesions = False):
	#split connected components_____________________________________________
	#points are considered connected if they have the same value and are neighbors
	#with connectivity =3, neighbors are defined like as a 3x3 cube where the point is
	#center cube. All connected points are labeled with the same value
	conn_comp = label(np.round(mask),connectivity=3).astype(np.float)

	#creates a array that has length of the number of connected components
	all_rano = np.zeros(len(np.unique(conn_comp)))
	j_idx = np.zeros(len(np.unique(conn_comp)))

	num_j = len(list(np.sum(conn_comp,(0,1)).argsort()[::-1]))
	num_i = len(np.unique(conn_comp)) - 1
	
	for i in range(1,len(np.unique(conn_comp))):
		#make a mask with only the current connected component
		curr_mask = np.zeros(conn_comp.shape)
		idx = np.where(conn_comp==np.unique(conn_comp)[i]) #index where the connected component mask equals the component of interest
		curr_mask[idx] = 1.0
		curr_mask = curr_mask.astype(np.int)

		#will store length of major axis and minor axis of each axial slice
		curr_maj = np.zeros(curr_mask.shape[2]) #size of z dimension
		curr_min = np.zeros(curr_mask.shape[2])

		#threshold is twice the slice thickness + gap spacing
		thres = 2*vox_z
		
		# if thres < 10: #for rano
		# 	thres = 10

		#looks at each axial slice in the mask with the current connected component
		#and orders the axial slices from largest to smallest area. 'j' is the index
		#of the axial slice. This helps to start with the slice containing the largest
		#area of the lesion and proceed from there
		jx = 0
		area_sorted_curr_mask_ix = list(np.sum(curr_mask,(0,1)).argsort()[::-1])
		has_area_ix = [np.sum(curr_mask[...,j] > 0) > 0 for j in area_sorted_curr_mask_ix]

		for j in list(np.sum(curr_mask,(0,1)).argsort()[::-1]):
			#if the area of the lesion of the current connected component at the current slice
			#is greater than 0, calculate the rano score
			if np.sum(curr_mask[...,j]>0):
				jx += 1
				#at each slice, gets the length of the longest diameter and the orthogonal diameter
				#Note: 
				# - swapaxes is basically transpose
				# - pic file name: base_pic_dir + index of slice + index of connected component
				if pic_dir == None:
					curr_maj[j], curr_min[j] = rapno(np.swapaxes(curr_mask[...,j],0,1), output_file=None, background_image=None, vox_x = vox_x, thres = thres)
				else:
					curr_maj[j], curr_min[j] = rapno(np.swapaxes(curr_mask[...,j],0,1), output_file=pic_dir+str(i)+'_'+str(j)+'.png', background_image=np.swapaxes(background[...,j],0,1), vox_x = vox_x, thres = thres)

		curr_maj = curr_maj * vox_x
		curr_min = curr_min * vox_x

		#remove all diameters if the lesion is not 'measurable'
		curr_maj[np.where(curr_maj<thres)] = 0
		curr_min[np.where(curr_min<thres)] = 0
		curr_rano = np.multiply(curr_maj,curr_min)

		all_rano[i] = curr_rano[np.argmax(curr_rano)] #store largest rapno score of particular current connected mask 
		j_idx[i] = np.argmax(curr_rano)#index of the axial slice with the largest rapno score

	#sum up the top x lesions
	idx2 = np.nonzero(all_rano)[0]
	
	sum_lesions = np.zeros(num_lesions) - 1
	if all_lesions:
		for i in range(num_lesions):
			ii = i + 1
			if len(idx2) < ii:
				sum_rano = np.sum(all_rano)
				i_idx = idx2
			else:
				sum_rano = np.sum(all_rano[np.argsort(all_rano)[-ii:]])
				i_idx = np.argsort(all_rano)[-ii:]
			sum_lesions[i] = sum_rano
		#save the number of the connected component and the axial slice of used in the final RANO measurement

		if make_csv:
			df2 = pd.DataFrame(np.vstack((i_idx,j_idx[i_idx])).T)
			df2.to_csv(pic_dir+'rano_idx.csv', index=False, header=False, sep=' ')
		return sum_lesions

	if len(idx2) < num_lesions:
		sum_rano = np.sum(all_rano)
		i_idx = idx2
	else:
		sum_rano = np.sum(all_rano[np.argsort(all_rano)[-num_lesions:]])
		i_idx = np.argsort(all_rano)[-num_lesions:]

	#save the number of the connected component and the axial slice of used in the final RAPNO measurement
	if make_csv:
		df2 = pd.DataFrame(np.vstack((i_idx,j_idx[i_idx])).T)
		df2.to_csv(pic_dir+'rano_idx.csv', index=False, header=False, sep=' ')
	return sum_rano