import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import find_contours
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.spatial.distance import cdist
import pandas as pd
from collections import namedtuple

class Point(namedtuple('Point', 'x y')):
	__slots__ = ()
	@property
	def length(self):
		return (self.x ** 2 + self.y ** 2) ** 0.5 #length from the origin
	def __sub__(self, p):
		return Point(self.x - p.x, self.y - p.y) #subtract self.x, self.y by coordinates of Point p
	def __str__(self):
		return 'Point: x=%6.3f  y=%6.3f  length=%6.3f' % (self.x, self.y, self.length)
		#print the coordinates and the length of the Point

def plot_contours(contours, lw=4, alpha=0.5):#展示找到的轮廓
	for n, contour in enumerate(contours):
		plt.plot(contour[:, 1], contour[:, 0], linewidth=lw, alpha=alpha, c='g')

def vector_norm(p):#正则化？
	length = p.length#计算长度
	return Point(p.x / length, p.y / length)

"""
Returns a (1) 2D matrix that represent the distances between points in P1 and P2 and (2) the
array containing [Point using coordinates from P1, Point using coordinates from P2, distance
between points] sorted in order of distance, largest to smallest

Inputs:
- P1: list of 2D coordinates
- P2: list of 2D coordinates
- min_length: int, minimum distance between points to be included in the second array returned
"""
def compute_pairwise_distances(P1, P2, min_length=10):
	#creates a 2D array where the element correlate with the location between the point from P1 and point from P2
	euc_dist_matrix = cdist(P1, P2, metric='euclidean')

	#scipy.spatial.distance.cdist(XA, XB, metric='euclidean', p=None, V=None, VI=None, w=None)
	indices = []
	for x in range(euc_dist_matrix.shape[0]):
		for y in range(euc_dist_matrix.shape[1]): 
			#astrick unpacks the contents, effectively sending a value for x and a value for y
			p1 = Point(*P1[x])
			p2 = Point(*P1[y])
			d = euc_dist_matrix[x, y]

			if p1 == p2 or d < min_length:
				continue
			indices.append([p1, p2, d])
	return euc_dist_matrix, sorted(indices, key=lambda x: x[2], reverse=True)#返回距离最大

"""
Returns np.array of coordinates that connect p1 and p2. The number of coordinates is specified
by d

Inputs:
- p1: Point
- p2: Point
- d: Distance between P1 and P2
"""
def interpolate(p1, p2, d):
	#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
	X = np.linspace(p1.x, p2.x, int(round(d))).astype(int)
	Y = np.linspace(p1.y, p2.y, int(round(d))).astype(int)
	XY = np.asarray(list(set(zip(X, Y))))
	return XY
#
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
#
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
#
def find_largest_orthogonal_cross_section(pairwise_distances, img, tolerance=0.1):#找到最大的正交横截面
	for i, (p1, p2, d1) in enumerate(pairwise_distances):
		# Compute intersections with background pixels
		XY = interpolate(p1, p2, d1)
		intersections = np.sum(img[x, y] == 0 for x, y in XY) #calculate instances where tumor not present
															  #in between line
		if intersections/float(len(XY)) < .1: #if the number of non-present pixels is less than threshold
			V = vector_norm(Point(p2.x - p1.x, p2.y - p1.y))
			# Iterate over remaining line segments
			for j, (q1, q2, d2) in enumerate(pairwise_distances[i:]):
				W = vector_norm(Point(q2.x - q1.x, q2.y - q1.y))
				if abs(np.dot(V, W)) < tolerance:
					XY = interpolate(q1, q2, d2)
					intersections = np.sum(img[x, y] == 0 for x, y in XY)
					if intersections/float(len(XY)) < .1 and intersect(p1,p2,q1,q2):
						return p1, p2, q1, q2

"""
Return length of longest diameter and longest length of orthagonal lesion

Input:
- binary_image: 2D numpy array representing an axial slice with mask of current connected component
- tol: float, cos(radian) allowed to satisfy orthogonal condition. 0.1 allows for 85-95 degree condition
"""
def rapno(binary_image, tol=0.1, output_file=None, background_image=None, vox_x = 1, thres = 10):
	#finds the location that would make an outline of the lesion at that axial slice
	binary_image2 = binary_image.astype('uint8') * 255
	contours = find_contours(binary_image2, level=1)

	#find_contours finds the contour at each axial slice and separates as individual element
	#combine contours from all slices into one array
	comb_contours = contours[0]
	for i in range(1,len(contours)):
		comb_contours = np.concatenate((comb_contours,contours[i]))
	comb_contours = comb_contours.astype(int)

	if len(contours) == 0:
		print("No lesion contours > 1 pixel detected.")
		return 0.0, 0.0

	# Calculate pairwise distances over boundary
	euc_dist_matrix, ordered_diameters = compute_pairwise_distances(comb_contours, comb_contours, min_length=thres/np.float(vox_x))

	# Exhaustive search for longest valid line segment and its orthogonal counterpart
	try:
		p1, p2, q1, q2 = find_largest_orthogonal_cross_section(ordered_diameters, binary_image, tolerance=tol)
		#multiply maximum bidiemnsional diameters
		rano_measure = ((p2 - p1).length * (q2 - q1).length) * vox_x * vox_x
	except TypeError:
		return 0.0, 0.0

	if output_file is not None:
		fig = plt.figure(figsize=(10, 10), frameon=False)
		plt.margins(0,0)
		plt.gca().set_axis_off()
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		if background_image is not None:
			plt.imshow(background_image, cmap='gray')
		else:
			plt.imshow(binary_image, cmap='gray')
		plot_contours(contours, lw=1, alpha=1.)
		D1 = np.asarray([[p1.x, p2.x], [p1.y, p2.y]])
		D2 = np.asarray([[q1.x, q2.x], [q1.y, q2.y]])

		plt.plot(D1[1, :], D1[0, :], lw=2, c='k')
		plt.plot(D2[1, :], D2[0, :], lw=2, c='k')
		plt.text(20, 20, 'RANO: {:.2f}'.format(rano_measure), {'color': 'r', 'fontsize': 20})
		plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0, dpi=100)#保存图片
		plt.close(fig)
	return (p2 - p1).length, (q2 - q1).length
