import numpy as np

def form_voxel_slice_img(img_h, img_w, point_cloud, num_slices=35, zmin=-2.5, zmax=1):
	"""
	We follow PIXOR:
	http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf
	
	keep the height information as channels along the 3rd dimension (like the RGB channels of 2D images)
	we can apply 2D convolution to the BEV representation. keeps the metric space.
	Projection and discretization: Define the 3D physical dimension L x W x H of the scene
	The 3D points within this 3D rectangular space are then discretized with a resolution of
	dL x dW x dH per cell.
	The value for each cell is encoded as occupancy (i.e., 1 if there exist points within this cell,
	and 0 otherwise). 
	After discretization, we get a 3D occupancy tensor of shape L/dL x W/dW x H/dH
	We also encode the reflectance (real value normalized to be within [0, 1]) of the LIDAR point in 
	a similar way. The only difference is that for reflectance we set dH = H. Our final representation
	is a combination of the 3D occupancy tensor and the 2D reflectance image, 
	whose shape is L/dL x W/dW x ( (H/dH) + 1).
	[-35,35] x [-40,40]
	discretization resolution 0.1 meter
	We set the height range to [−2.5, 1] meters in LIDAR coordinates and divide all points into 
	35 slices with bin size of 0.1 meter. One reflectance channel is also computed
	As a result, our input representation has the dimension of 800 × 700 × 36.
	"""
	x_vals = np.round(point_cloud[:,0]).astype(np.int64)
	y_vals = np.round(point_cloud[:,1]).astype(np.int64)
	x_vals = np.logical_and(0 <= x_vals, x_vals <= img_w)
	y_vals = np.logical_and(0 <= y_vals, y_vals <= img_h)
	xy_vals = np.logical_and(x_vals, y_vals)
	point_cloud = point_cloud[xy_vals]
	x_vals = np.round(point_cloud[:,0]).astype(np.int64)
	y_vals = np.round(point_cloud[:,1]).astype(np.int64)
	z_vals = point_cloud[:,2]
	img = np.zeros((img_h, img_w, num_slices), dtype=np.uint8)

	slice_width = (zmax - zmin) / num_slices

	# one more z_plane then slices we will use. Plane just denotes interval bin endpoints
	z_planes = np.linspace(zmin, zmax, num_slices+1)

	for z_plane_idx in range(num_slices):
		z_start = zmin + z_plane_idx * slice_width
		z_end = z_start + slice_width
		# zmin -> (zmax-1), 
		z_is_valid = np.logical_and(z_vals >= z_start, z_vals < z_end)
		y = y_vals[z_is_valid]
		x = x_vals[z_is_valid]
		img[y,x,z_plane_idx] = 1
	return img



def form_voxel_slice_img_vectorized(img_h, img_w, point_cloud, num_slices=35, zmin=-2.5, zmax=1):
	"""
		Args:
		-	img_h
		-	img_w
		-	point_cloud
		-	num_slices
		-	zmin
		-	zmax
		Returns:
		-	img
	"""
	x_vals = np.round(point_cloud[:,0]).astype(np.int64)
	y_vals = np.round(point_cloud[:,1]).astype(np.int64)
	z_vals = point_cloud[:,2]
	img = np.zeros((img_h, img_w, num_slices), dtype=np.uint8)

	slice_width = (zmax - zmin) / num_slices

	# Plane just denotes interval bin endpoints. One more z_plane then `slices`.
	z_planes = np.linspace(zmin, zmax, num_slices+1)

	z_plane_indices = np.digitize(z_vals, z_planes)
	# convert to 0-indexed per bin
	z_plane_indices -= 1

	# TODO: discard all that fall out of range.
	is_valid = np.logical_and(z_plane_indices >= 0, z_plane_indices < num_slices)
	x = x_vals[is_valid]
	y = y_vals[is_valid]
	z_plane_indices = z_plane_indices[is_valid]

	img[y,x,z_plane_indices] = 1
	return img


def test_form_voxel_slice_img():
	"""
	"""
	point_cloud1 = np.array([
		[1,1,2],
		[1,1,6],
		[3,3,0],
		[3,3,4]
	])

	point_cloud2 = np.array([
		[1,1,3.9],
		[1,1,6.01],
		[3,3,1.9],
		[3,3,5.9]
	])

	img_h = 4
	img_w = 4
	num_slices = 4
	zmin = 0
	zmax = 8
	img1 = form_voxel_slice_img(img_h, img_w, point_cloud1, num_slices, zmin, zmax)
	img2 = form_voxel_slice_img(img_h, img_w, point_cloud1, num_slices, zmin, zmax)


	img1_v = form_voxel_slice_img_vectorized(img_h, img_w, point_cloud1, num_slices, zmin, zmax)
	img2_v = form_voxel_slice_img_vectorized(img_h, img_w, point_cloud1, num_slices, zmin, zmax)

	assert np.allclose(img1,img2)
	assert np.allclose(img1,img1_v)
	assert np.allclose(img2,img2_v)




def test_form_voxel_slice_img_all_out_of_bounds():
	"""
	"""
	point_cloud1 = np.array([
		[1,1,-2],
		[1,1,-6],
		[3,3,2.0],
		[3,3,14]
	])

	point_cloud2 = np.array([
		[1,1,-200],
		[1,1,-100],
		[3,3,2.01],
		[3,3,100]
	])

	img_h = 4
	img_w = 4
	num_slices = 4
	zmin = 0
	zmax = 8

	img1 = form_voxel_slice_img(img_h, img_w, point_cloud1, num_slices, zmin, zmax)
	img2 = form_voxel_slice_img(img_h, img_w, point_cloud1, num_slices, zmin, zmax)

	img1_v = form_voxel_slice_img_vectorized(img_h, img_w, point_cloud1, num_slices, zmin, zmax)
	img2_v = form_voxel_slice_img_vectorized(img_h, img_w, point_cloud1, num_slices, zmin, zmax)

	assert np.allclose(img1,img2)
	assert np.allclose(img1,img1_v)
	assert np.allclose(img2,img2_v)